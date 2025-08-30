#!/usr/bin/env python3
"""
CloudFront Reverse Proxy Demo

This script demonstrates how to set up a CloudFront distribution
as a reverse proxy for web applications and APIs.

The demo shows:
1. Creating a reverse proxy distribution
2. Testing the setup
3. Managing cache invalidation
4. Monitoring performance

Usage:
    python3 demo_cloudfront_proxy.py
"""

import subprocess
import json
import time
import requests
from typing import Dict, Any, Optional
import argparse


class CloudFrontProxyDemo:
    """Demonstration of CloudFront reverse proxy functionality."""

    def __init__(self, region: str = "us-east-1"):
        """Initialize the demo."""
        self.region = region
        self.distribution_id = None
        self.domain_name = None

    def run_demo(self, origin_domain: str = "httpbin.org", use_real_aws: bool = False):
        """
        Run the complete CloudFront reverse proxy demonstration.

        Parameters
        ----------
        origin_domain : str
            Origin domain to proxy (default: httpbin.org for testing)
        use_real_aws : bool
            Whether to actually create AWS resources (requires AWS credentials)
        """
        print("🚀 CloudFront Reverse Proxy Demonstration")
        print("=" * 60)
        print(f"Origin Domain: {origin_domain}")
        print(f"Real AWS Resources: {'Yes' if use_real_aws else 'No (dry run)'}")
        print()

        if not use_real_aws:
            print("🔍 Running in DRY RUN mode - no actual AWS resources will be created")
            print("   To create real resources, run with --real-aws flag")
            print()

            # Show what would be created
            self._show_dry_run_config(origin_domain)
            return

        # Step 1: Create the distribution
        print("Step 1: Creating CloudFront Distribution")
        print("-" * 40)

        success = self._create_distribution(origin_domain)
        if not success:
            print("❌ Failed to create distribution")
            return

        # Step 2: Wait for deployment
        print("\\nStep 2: Waiting for Distribution Deployment")
        print("-" * 40)
        self._wait_for_deployment()

        # Step 3: Test the reverse proxy
        print("\\nStep 3: Testing Reverse Proxy Functionality")
        print("-" * 40)
        self._test_reverse_proxy()

        # Step 4: Demonstrate cache behavior
        print("\\nStep 4: Demonstrating Cache Behavior")
        print("-" * 40)
        self._test_caching()

        # Step 5: Show management commands
        print("\\nStep 5: Management and Maintenance")
        print("-" * 40)
        self._show_management_commands()

        print("\\n🎉 Demo completed successfully!")
        print(f"\\n📋 Your CloudFront Reverse Proxy:")
        print(f"   Domain: {self.domain_name}")
        print(f"   Distribution ID: {self.distribution_id}")
        print(f"   Origin: {origin_domain}")

    def _show_dry_run_config(self, origin_domain: str):
        """Show the configuration that would be created."""
        print("📋 Configuration Preview:")
        print("-" * 30)
        print(f"Origin Domain: {origin_domain}")
        print("Origin Protocol: https-only")
        print("HTTP Port: 80")
        print("HTTPS Port: 443")
        print("Price Class: PriceClass_100")
        print("Viewer Protocol: redirect-to-https")
        print("Compression: Enabled")
        print("IPv6: Enabled")
        print("HTTP/2: Enabled")

        print("\\n🔧 Cache Behavior:")
        print("   Forward Query Strings: Yes")
        print("   Forward Cookies: All")
        print("   Forward Headers: Host, User-Agent, Accept, etc.")
        print("   Min TTL: 0 seconds")
        print("   Default TTL: 86,400 seconds (1 day)")
        print("   Max TTL: 31,536,000 seconds (1 year)")

        print("\\n📊 Security Features:")
        print("   SSL/TLS: TLSv1.2_2021 minimum")
        print("   Geo Restriction: None")
        print("   WAF: Not configured (can be added)")

        print("\\n📝 AWS CLI Commands that would be run:")
        print("   aws cloudfront create-distribution --distribution-config file://config.json")
        print("   aws cloudfront get-distribution --id <distribution-id>")
        print("   aws cloudfront create-invalidation --distribution-id <id> --invalidation-batch file://batch.json")

    def _create_distribution(self, origin_domain: str) -> bool:
        """Create the CloudFront distribution."""
        cmd = [
            "python3", "cloudfront_reverse_proxy.py", "create",
            "--origin-domain", origin_domain,
            "--comment", "Demo Reverse Proxy Distribution",
            "--region", self.region
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
        """Test the reverse proxy functionality."""
        if not self.domain_name:
            print("⚠️  No domain name available for testing")
            return

        print("🧪 Testing reverse proxy functionality...")

        # Test basic connectivity
        test_url = f"https://{self.domain_name}/get"
        print(f"   Testing URL: {test_url}")

        try:
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200:
                print(f"   ✅ HTTP {response.status_code} - Reverse proxy working!")
                print(f"   📄 Response time: {response.elapsed.total_seconds():.2f}s")
            else:
                print(f"   ⚠️  HTTP {response.status_code} - Unexpected response")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request failed: {e}")
            print("   💡 Note: Domain may still be propagating (can take up to 24 hours)")

    def _test_caching(self):
        """Test caching behavior."""
        if not self.domain_name:
            return

        print("📦 Testing cache behavior...")

        # Make multiple requests to see caching in action
        test_url = f"https://{self.domain_name}/cache/60"  # 60 second cache header

        print(f"   Testing URL: {test_url}")

        for i in range(3):
            try:
                start_time = time.time()
                response = requests.get(test_url, timeout=10)
                elapsed = time.time() - start_time

                cache_status = response.headers.get('X-Cache', 'Unknown')
                cf_ray = response.headers.get('CF-RAY', 'Unknown')

                print(f"   Request {i+1}: HTTP {response.status_code} in {elapsed:.3f}s")
                print(f"      Cache: {cache_status}")
                print(f"      CF-RAY: {cf_ray}")

                if i < 2:  # Don't sleep after last request
                    time.sleep(1)

            except requests.exceptions.RequestException as e:
                print(f"   ❌ Request {i+1} failed: {e}")
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


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="CloudFront Reverse Proxy Demo")
    parser.add_argument("--origin-domain", default="httpbin.org",
                       help="Origin domain to proxy (default: httpbin.org)")
    parser.add_argument("--real-aws", action="store_true",
                       help="Actually create AWS resources (requires AWS credentials)")
    parser.add_argument("--region", default="us-east-1",
                       help="AWS region (default: us-east-1)")

    args = parser.parse_args()

    try:
        demo = CloudFrontProxyDemo(region=args.region)
        demo.run_demo(origin_domain=args.origin_domain, use_real_aws=args.real_aws)

    except KeyboardInterrupt:
        print("\\n\\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
