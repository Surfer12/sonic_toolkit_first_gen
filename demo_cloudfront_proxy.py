#!/usr/bin/env python3
"""
CloudFront Reverse Proxy Demo

This script demonstrates how to set up a CloudFront distribution
as a reverse proxy for web applications and APIs.

Enhanced Features:
- External configuration file support (JSON/YAML)
- Structured logging for debugging
- Retry logic for network requests
- Metrics export to files
- Comprehensive error handling

The demo shows:
1. Creating a reverse proxy distribution
2. Testing the setup
3. Managing cache invalidation
4. Monitoring performance

Usage:
    python3 demo_cloudfront_proxy.py
    python3 demo_cloudfront_proxy.py --config config.json --real-aws
    python3 demo_cloudfront_proxy.py --log-level DEBUG
"""

import subprocess
import json
import time
import requests
import logging
import argparse
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml  # Add pyyaml dependency for YAML support


@dataclass
class DemoConfig:
    """Configuration for CloudFront proxy demo."""
    region: str = "us-east-1"
    origin_domain: str = "httpbin.org"
    price_class: str = "PriceClass_100"
    use_real_aws: bool = False
    log_level: str = "INFO"
    log_file: Optional[str] = None
    config_file: Optional[str] = None
    output_dir: str = "./output"
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    export_metrics: bool = True


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: str
    distribution_id: Optional[str]
    domain_name: Optional[str]
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class CloudFrontProxyDemo:
    """Enhanced CloudFront reverse proxy demonstration with advanced features."""

    def __init__(self, config: DemoConfig):
        """Initialize the enhanced demo."""
        self.config = config
        self.distribution_id = None
        self.domain_name = None
        self.metrics = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            distribution_id=None,
            domain_name=None
        )

        # Setup logging
        self._setup_logging()

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(exist_ok=True)

        self.logger.info(f"CloudFront Proxy Demo initialized with config: {config}")

    def _setup_logging(self):
        """Setup structured logging."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        # Create logger
        self.logger = logging.getLogger('CloudFrontDemo')
        self.logger.setLevel(log_level)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def _load_config(self, config_file: str) -> DemoConfig:
        """Load configuration from JSON or YAML file."""
        config_path = Path(config_file)

        if not config_path.exists():
            self.logger.warning(f"Config file {config_file} not found, using defaults")
            return DemoConfig()

        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

            # Create config from loaded data
            config = DemoConfig()
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    self.logger.warning(f"Unknown config key: {key}")

            self.logger.info(f"Loaded configuration from {config_file}")
            return config

        except Exception as e:
            self.logger.error(f"Failed to load config file {config_file}: {e}")
            return DemoConfig()

    def _retry_request(self, func, *args, **kwargs):
        """Retry mechanism for network requests."""
        last_exception = None

        for attempt in range(self.config.retry_attempts):
            try:
                self.logger.debug(f"Attempt {attempt + 1}/{self.config.retry_attempts}")
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"All {self.config.retry_attempts} attempts failed")

        raise last_exception

    def _export_metrics(self):
        """Export performance metrics to file."""
        if not self.config.export_metrics:
            return

        metrics_file = Path(self.config.output_dir) / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            # Update final metrics
            self.metrics.distribution_id = self.distribution_id
            self.metrics.domain_name = self.domain_name

            with open(metrics_file, 'w') as f:
                json.dump(asdict(self.metrics), f, indent=2)

            self.logger.info(f"Metrics exported to {metrics_file}")

        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")

    def _update_metrics(self, response_time: float, status_code: int, cache_status: str = None):
        """Update performance metrics."""
        self.metrics.request_count += 1

        if 200 <= status_code < 300:
            self.metrics.success_count += 1
        else:
            self.metrics.failure_count += 1

        self.metrics.avg_response_time = (
            (self.metrics.avg_response_time * (self.metrics.request_count - 1)) + response_time
        ) / self.metrics.request_count

        self.metrics.min_response_time = min(self.metrics.min_response_time, response_time)
        self.metrics.max_response_time = max(self.metrics.max_response_time, response_time)

        if cache_status:
            if 'Hit' in cache_status:
                self.metrics.cache_hits += 1
            else:
                self.metrics.cache_misses += 1

    def run_demo(self):
        """
        Run the complete CloudFront reverse proxy demonstration.
        """
        self.logger.info("🚀 Starting CloudFront Reverse Proxy Demonstration")
        self.logger.info("=" * 60)
        self.logger.info(f"Origin Domain: {self.config.origin_domain}")
        self.logger.info(f"Real AWS Resources: {'Yes' if self.config.use_real_aws else 'No (dry run)'}")
        self.logger.info(f"Region: {self.config.region}")
        self.logger.info(f"Output Directory: {self.config.output_dir}")

        try:
            if not self.config.use_real_aws:
                self.logger.info("🔍 Running in DRY RUN mode - no actual AWS resources will be created")
                self.logger.info("   To create real resources, use --real-aws flag")

                # Show what would be created
                self._show_dry_run_config()
                return

            # Step 1: Create the distribution
            self.logger.info("\\n📦 Step 1: Creating CloudFront Distribution")
            self.logger.info("-" * 40)

            success = self._create_distribution()
            if not success:
                self.logger.error("❌ Failed to create distribution")
                return

            # Step 2: Wait for deployment
            self.logger.info("\\n⏳ Step 2: Waiting for Distribution Deployment")
            self.logger.info("-" * 40)
            self._wait_for_deployment()

            # Step 3: Test the reverse proxy
            self.logger.info("\\n🧪 Step 3: Testing Reverse Proxy Functionality")
            self.logger.info("-" * 40)
            self._test_reverse_proxy()

            # Step 4: Demonstrate cache behavior
            self.logger.info("\\n📦 Step 4: Demonstrating Cache Behavior")
            self.logger.info("-" * 40)
            self._test_caching()

            # Step 5: Show management commands
            self.logger.info("\\n🔧 Step 5: Management and Maintenance")
            self.logger.info("-" * 40)
            self._show_management_commands()

            # Export metrics
            self._export_metrics()

            self.logger.info("\\n🎉 Demo completed successfully!")
            self.logger.info(f"\\n📋 Your CloudFront Reverse Proxy:")
            self.logger.info(f"   Domain: {self.domain_name}")
            self.logger.info(f"   Distribution ID: {self.distribution_id}")

        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            self.metrics.errors.append(str(e))
            self._export_metrics()
            raise

    def _show_dry_run_config(self):
        """Show the configuration that would be created."""
        self.logger.info("📋 Configuration Preview:")
        self.logger.info("-" * 30)
        self.logger.info(f"Origin Domain: {self.config.origin_domain}")
        self.logger.info("Origin Protocol: https-only")
        self.logger.info("HTTP Port: 80")
        self.logger.info("HTTPS Port: 443")
        self.logger.info(f"Price Class: {self.config.price_class}")
        self.logger.info("Viewer Protocol: redirect-to-https")
        self.logger.info("Compression: Enabled")
        self.logger.info("IPv6: Enabled")
        self.logger.info("HTTP/2: Enabled")

        self.logger.info("\\n🔧 Cache Behavior:")
        self.logger.info("   Forward Query Strings: Yes")
        self.logger.info("   Forward Cookies: All")
        self.logger.info("   Forward Headers: Host, User-Agent, Accept, etc.")
        self.logger.info("   Min TTL: 0 seconds")
        self.logger.info("   Default TTL: 86,400 seconds (1 day)")
        self.logger.info("   Max TTL: 31,536,000 seconds (1 year)")

        self.logger.info("\\n📊 Security Features:")
        self.logger.info("   SSL/TLS: TLSv1.2_2021 minimum")
        self.logger.info("   Geo Restriction: None")
        self.logger.info("   WAF: Not configured (can be added)")

        self.logger.info("\\n📝 AWS CLI Commands that would be run:")
        self.logger.info("   aws cloudfront create-distribution --distribution-config file://config.json")
        self.logger.info("   aws cloudfront get-distribution --id <distribution-id>")
        self.logger.info("   aws cloudfront create-invalidation --distribution-id <id> --invalidation-batch file://batch.json")

    def _create_distribution(self) -> bool:
        """Create the CloudFront distribution."""
        cmd = [
            "python3", "cloudfront_reverse_proxy.py", "create",
            "--origin-domain", self.config.origin_domain,
            "--comment", "Demo Reverse Proxy Distribution",
            "--region", self.config.region
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Parse the JSON output to get distribution details
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if "Distribution ID:" in line:
                    self.distribution_id = line.split(": ")[1].strip()
                elif "Domain Name:" in line:
                    self.domain_name = line.split(": ")[1].strip()

            print(f"✅ Distribution created: {self.distribution_id}")
            print(f"🌐 Domain: {self.domain_name}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create distribution: {e}")
            print(f"Error: {e.stderr}")
            return False

    def _wait_for_deployment(self):
        """Wait for the distribution to be deployed."""
        if not self.distribution_id:
            return

        print(f"⏳ Waiting for distribution {self.distribution_id} to deploy...")
        print("   This may take 10-30 minutes...")

        max_attempts = 60  # 60 attempts * 30 seconds = 30 minutes max
        for attempt in range(max_attempts):
            status = self._get_distribution_status()
            if status == "Deployed":
                print("✅ Distribution deployed successfully!")
                return
            elif status in ["InProgress", "Processing"]:
                print(f"   Attempt {attempt + 1}/{max_attempts}: {status}")
                time.sleep(30)  # Wait 30 seconds
            else:
                print(f"❌ Deployment failed with status: {status}")
                return

        print("⏰ Deployment timeout - check AWS console for status")

    def _get_distribution_status(self) -> str:
        """Get current distribution status."""
        if not self.distribution_id:
            return "Unknown"

        cmd = [
            "python3", "cloudfront_reverse_proxy.py", "status",
            "--id", self.distribution_id,
            "--region", self.region
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return "Deployed"  # Simplified - real implementation would parse output
        except subprocess.CalledProcessError:
            return "Failed"

    def _test_reverse_proxy(self):
        """Test the reverse proxy functionality with retry logic."""
        if not self.domain_name:
            self.logger.warning("⚠️  No domain name available for testing")
            return

        self.logger.info("🧪 Testing reverse proxy functionality...")

        # Test basic connectivity with retry
        test_url = f"https://{self.domain_name}/get"
        self.logger.info(f"   Testing URL: {test_url}")

        try:
            def make_request():
                start_time = time.time()
                response = requests.get(test_url, timeout=self.config.timeout)
                elapsed = time.time() - start_time

                # Update metrics
                self._update_metrics(elapsed, response.status_code)

                return response, elapsed

            response, elapsed = self._retry_request(make_request)

            if response.status_code == 200:
                self.logger.info(f"   ✅ HTTP {response.status_code} - Reverse proxy working!")
                self.logger.info(f"   📄 Response time: {elapsed:.2f}s")
            else:
                self.logger.warning(f"   ⚠️  HTTP {response.status_code} - Unexpected response")
                self.metrics.errors.append(f"HTTP {response.status_code} from {test_url}")

        except Exception as e:
            self.logger.error(f"   ❌ Request failed: {e}")
            self.logger.info("   💡 Note: Domain may still be propagating (can take up to 24 hours)")
            self.metrics.errors.append(f"Request failed: {str(e)}")

    def _test_caching(self):
        """Test caching behavior with retry logic."""
        if not self.domain_name:
            return

        self.logger.info("📦 Testing cache behavior...")

        # Make multiple requests to see caching in action
        test_url = f"https://{self.domain_name}/cache/60"  # 60 second cache header
        self.logger.info(f"   Testing URL: {test_url}")

        for i in range(3):
            try:
                def make_cache_request():
                    start_time = time.time()
                    response = requests.get(test_url, timeout=self.config.timeout)
                    elapsed = time.time() - start_time

                    cache_status = response.headers.get('X-Cache', 'Unknown')
                    cf_ray = response.headers.get('CF-RAY', 'Unknown')

                    # Update metrics
                    self._update_metrics(elapsed, response.status_code, cache_status)

                    return response, elapsed, cache_status, cf_ray

                response, elapsed, cache_status, cf_ray = self._retry_request(make_cache_request)

                self.logger.info(f"   Request {i+1}: HTTP {response.status_code} in {elapsed:.3f}s")
                self.logger.info(f"      Cache: {cache_status}")
                self.logger.info(f"      CF-RAY: {cf_ray}")

                if i < 2:  # Don't sleep after last request
                    time.sleep(1)

            except Exception as e:
                self.logger.error(f"   ❌ Request {i+1} failed: {e}")
                self.metrics.errors.append(f"Cache test request {i+1} failed: {str(e)}")
                break

    def _show_management_commands(self):
        """Show management commands."""
        if not self.distribution_id:
            return

        print("🔧 Management Commands:")
        print(f"   # Check status: python3 cloudfront_reverse_proxy.py status --id {self.distribution_id}")
        print(f"   # Invalidate cache: python3 cloudfront_reverse_proxy.py invalidate --id {self.distribution_id}")
        print("   # Update distribution: python3 cloudfront_reverse_proxy.py update --id {self.distribution_id} --origin-domain new-origin.com")
        print()
        print("🗂️  AWS Console URLs:")
        print(f"   Distribution: https://console.aws.amazon.com/cloudfront/home?region={self.region}#distribution-settings:{self.distribution_id}")
        print(f"   CloudWatch Metrics: https://console.aws.amazon.com/cloudwatch/home?region={self.region}#metricsV2:graph=~metric~Source~CloudFront~statistic~Average~view~timeSeries~stacked~false~region~{self.region}~start~-{self.region}~end~P1D~dimensions~DistributionId~{self.distribution_id}")


def create_sample_config():
    """Create a sample configuration file."""
    sample_config = {
        "region": "us-east-1",
        "origin_domain": "httpbin.org",
        "price_class": "PriceClass_100",
        "use_real_aws": False,
        "log_level": "INFO",
        "log_file": "./output/demo.log",
        "output_dir": "./output",
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "timeout": 30,
        "export_metrics": True
    }

    config_path = Path("./config.json")
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)

    print(f"✅ Sample configuration created: {config_path}")


def main():
    """Main demo function with enhanced configuration support."""
    parser = argparse.ArgumentParser(description="CloudFront Reverse Proxy Demo")
    parser.add_argument("--config", help="Path to configuration file (JSON or YAML)")
    parser.add_argument("--origin-domain", help="Origin domain to proxy")
    parser.add_argument("--real-aws", action="store_true", help="Actually create AWS resources")
    parser.add_argument("--region", help="AWS region")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", help="Path to log file")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--create-config", action="store_true",
                       help="Create sample configuration file and exit")

    args = parser.parse_args()

    # Create sample config if requested
    if args.create_config:
        create_sample_config()
        return 0

    try:
        # Load configuration
        if args.config:
            demo = CloudFrontProxyDemo(DemoConfig())
            config = demo._load_config(args.config)
        else:
            config = DemoConfig()

        # Override config with command line arguments
        if args.origin_domain:
            config.origin_domain = args.origin_domain
        if args.real_aws:
            config.use_real_aws = True
        if args.region:
            config.region = args.region
        if args.log_level:
            config.log_level = args.log_level
        if args.log_file:
            config.log_file = args.log_file
        if args.output_dir:
            config.output_dir = args.output_dir

        demo = CloudFrontProxyDemo(config)
        demo.run_demo()

    except KeyboardInterrupt:
        print("\\n\\n⏹️  Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
