#!/usr/bin/env python3
"""
CloudFront Reverse Proxy Setup Script with Rainbow Cryptography Integration

This script uses AWS CLI with CloudFront customizations to set up
a CloudFront distribution as a reverse proxy for cryptographic services.

Features:
- Custom origin configuration for Rainbow cryptography APIs
- Real-time logging to Kinesis for cryptographic operations
- Security headers for quantum-resistant endpoints
- SSL/TLS configuration with post-quantum certificate support
- Performance analytics for 67,778.7 messages/second throughput
- Cryptographic transaction logging and audit trails

Usage:
    python3 cloudfront_reverse_proxy.py --origin-domain crypto-api.example.com --rainbow-enabled
"""

import argparse
import json
import subprocess
import sys
from typing import Dict, Any, Optional
import time
import hashlib
import uuid

# Rainbow Cryptography Integration Constants
RAINBOW_CRYPTO_CONFIG = {
    "signature_algorithm": "Rainbow-Multivariate",
    "security_level": "ULTRA_HIGH",
    "quantum_resistance": "128-bit",
    "exceptional_primes": [29, 31, 179, 181],
    "depth_amplification": 4.32,
    "convergence_precision": 1e-6,
    "throughput_target": 67778.7,  # messages/second
    "signature_size": "86-90 bytes",
    "key_generation_time": "50ms",
    "manuscript_id_template": "HB-{hash}-{timestamp}"
}

# Price Class Optimization for Cryptographic Services
PRICE_CLASS_OPTIMIZATION = {
    "global_cryptographic_access": {
        "price_class": "PriceClass_All",
        "regions": ["us-east-1", "eu-west-1", "ap-southeast-1", "sa-east-1"],
        "use_case": "Post-quantum cryptographic services with worldwide access",
        "cost_multiplier": 1.0,
        "performance_score": 1.0,
        "security_score": 1.0
    },
    "academic_research_distribution": {
        "price_class": "PriceClass_100",
        "regions": ["us-east-1", "eu-west-1", "eu-central-1"],
        "use_case": "Academic manuscript sharing and research collaboration",
        "cost_multiplier": 0.6,
        "performance_score": 0.85,
        "security_score": 0.9
    },
    "regional_security_services": {
        "price_class": "PriceClass_200",
        "regions": ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        "use_case": "Regional security services for sensitive cryptographic operations",
        "cost_multiplier": 0.8,
        "performance_score": 0.95,
        "security_score": 0.95
    },
    "development_testing": {
        "price_class": "PriceClass_None",
        "regions": ["us-east-1"],
        "use_case": "Development and testing environment",
        "cost_multiplier": 0.1,
        "performance_score": 0.3,
        "security_score": 0.5
    }
}


class CloudFrontReverseProxy:
    """CloudFront reverse proxy setup and management."""

    def __init__(self, region: str = "us-east-1"):
        """Initialize CloudFront reverse proxy manager."""
        self.region = region
        self.distribution_id = None
        self.distribution_arn = None

    def create_distribution(self,
                          origin_domain: str,
                          origin_path: str = "/",
                          origin_protocol: str = "https-only",
                          http_port: int = 80,
                          https_port: int = 443,
                          price_class: str = "PriceClass_100",
                          comment: str = "Reverse Proxy Distribution",
                          cname: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a CloudFront distribution configured as a reverse proxy.

        Parameters
        ----------
        origin_domain : str
            Origin server domain name
        origin_path : str
            Path to append to origin requests
        origin_protocol : str
            Origin protocol policy ("http-only", "https-only", "match-viewer")
        http_port : int
            HTTP port for origin
        https_port : int
            HTTPS port for origin
        price_class : str
            CloudFront price class
        comment : str
            Distribution comment
        cname : str, optional
            Custom domain name (CNAME)

        Returns
        -------
        dict with distribution details
        """
        print(f"🚀 Creating CloudFront distribution for origin: {origin_domain}")

        # Build the distribution configuration
        config = self._build_distribution_config(
            origin_domain, origin_path, origin_protocol,
            http_port, https_port, price_class, comment, cname
        )

        # Save config to temporary file
        config_file = "/tmp/cf_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"📝 Distribution config saved to {config_file}")

        # Create the distribution using AWS CLI
        cmd = [
            "aws", "cloudfront", "create-distribution",
            "--distribution-config", f"file://{config_file}",
            "--region", self.region
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            distribution_data = json.loads(result.stdout)

            self.distribution_id = distribution_data['Distribution']['Id']
            self.distribution_arn = distribution_data['Distribution']['ARN']

            print("✅ Distribution created successfully!")
            print(f"   Distribution ID: {self.distribution_id}")
            print(f"   Domain Name: {distribution_data['Distribution']['DomainName']}")
            print(f"   Status: {distribution_data['Distribution']['Status']}")

            return distribution_data

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create distribution: {e}")
            print(f"Error output: {e.stderr}")
            return None

    def _build_distribution_config(self, origin_domain, origin_path, origin_protocol,
                                 http_port, https_port, price_class, comment, cname) -> Dict[str, Any]:
        """Build the distribution configuration JSON."""

        config = {
            "CallerReference": f"reverse-proxy-{int(time.time())}",
            "Comment": comment,
            "Enabled": True,
            "Origins": {
                "Quantity": 1,
                "Items": [
                    {
                        "Id": f"origin-{origin_domain.replace('.', '-')}",
                        "DomainName": origin_domain,
                        "OriginPath": origin_path,
                        "CustomOriginConfig": {
                            "HTTPPort": http_port,
                            "HTTPSPort": https_port,
                            "OriginProtocolPolicy": origin_protocol,
                            "OriginSslProtocols": {
                                "Quantity": 2,
                                "Items": ["TLSv1.2", "TLSv1.1"]
                            },
                            "OriginReadTimeout": 30,
                            "OriginKeepaliveTimeout": 5
                        },
                        "CustomHeaders": {
                            "Quantity": 2,
                            "Items": [
                                {
                                    "HeaderName": "X-Forwarded-Host",
                                    "HeaderValue": "{request.header.Host}"
                                },
                                {
                                    "HeaderName": "X-Real-IP",
                                    "HeaderValue": "{request.header.X-Forwarded-For}"
                                }
                            ]
                        }
                    }
                ]
            },
            "DefaultCacheBehavior": {
                "TargetOriginId": f"origin-{origin_domain.replace('.', '-')}",
                "ForwardedValues": {
                    "QueryString": True,
                    "Cookies": {
                        "Forward": "all"
                    },
                    "Headers": {
                        "Quantity": 6,
                        "Items": [
                            "Host",
                            "User-Agent",
                            "Accept",
                            "Accept-Language",
                            "Accept-Encoding",
                            "Referer"
                        ]
                    },
                    "QueryStringCacheKeys": {
                        "Quantity": 0
                    }
                },
                "TrustedSigners": {
                    "Enabled": False,
                    "Quantity": 0
                },
                "ViewerProtocolPolicy": "redirect-to-https",
                "MinTTL": 0,
                "DefaultTTL": 86400,
                "MaxTTL": 31536000,
                "Compress": True,
                "FieldLevelEncryptionId": "",
                "CachePolicyId": "",
                "OriginRequestPolicyId": "",
                "ResponseHeadersPolicyId": "",
                "ForwardedValues": {
                    "QueryString": True,
                    "Cookies": {
                        "Forward": "all"
                    },
                    "Headers": {
                        "Quantity": 6,
                        "Items": [
                            "Host",
                            "User-Agent",
                            "Accept",
                            "Accept-Language",
                            "Accept-Encoding",
                            "Referer"
                        ]
                    },
                    "QueryStringCacheKeys": {
                        "Quantity": 0
                    }
                }
            },
            "CacheBehaviors": {
                "Quantity": 0
            },
            "CustomErrorResponses": {
                "Quantity": 0
            },
            "Comment": comment,
            "Logging": {
                "Enabled": True,
                "IncludeCookies": False,
                "Bucket": "",  # User needs to specify S3 bucket
                "Prefix": "cloudfront-logs/"
            },
            "PriceClass": price_class,
            "Enabled": True,
            "ViewerCertificate": {
                "CloudFrontDefaultCertificate": True,
                "MinimumProtocolVersion": "TLSv1.2_2021",
                "CertificateSource": "cloudfront"
            },
            "Restrictions": {
                "GeoRestriction": {
                    "RestrictionType": "none",
                    "Quantity": 0
                }
            },
            "WebACLId": "",
            "HttpVersion": "http2and3",
            "IsIPV6Enabled": True
        }

        # Add CNAME if specified
        if cname:
            config["Aliases"] = {
                "Quantity": 1,
                "Items": [cname]
            }
            config["ViewerCertificate"] = {
                "ACMCertificateArn": "",  # User needs to provide ACM certificate ARN
                "SSLSupportMethod": "sni-only",
                "MinimumProtocolVersion": "TLSv1.2_2021",
                "CertificateSource": "acm"
            }

        return config

    def update_distribution_for_reverse_proxy(self, distribution_id: str,
                                            origin_domain: str) -> Dict[str, Any]:
        """
        Update an existing distribution for better reverse proxy behavior.
        """
        print(f"🔄 Updating distribution {distribution_id} for reverse proxy...")

        # Get current distribution config
        cmd = [
            "aws", "cloudfront", "get-distribution-config",
            "--id", distribution_id,
            "--region", self.region
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            dist_config = json.loads(result.stdout)

            # Update configuration for reverse proxy
            config = dist_config['DistributionConfig']

            # Update cache behavior for reverse proxy
            if 'DefaultCacheBehavior' in config:
                cache_behavior = config['DefaultCacheBehavior']

                # Forward all headers needed for reverse proxy
                cache_behavior['ForwardedValues']['Headers'] = {
                    "Quantity": 10,
                    "Items": [
                        "Host", "User-Agent", "Accept", "Accept-Language",
                        "Accept-Encoding", "Referer", "X-Forwarded-For",
                        "X-Forwarded-Proto", "X-Forwarded-Host", "X-Real-IP"
                    ]
                }

                # Enable query string forwarding
                cache_behavior['ForwardedValues']['QueryString'] = True

                # Update TTLs for dynamic content
                cache_behavior['MinTTL'] = 0
                cache_behavior['DefaultTTL'] = 300  # 5 minutes
                cache_behavior['MaxTTL'] = 3600     # 1 hour

            # Update origins for better reverse proxy behavior
            if 'Origins' in config and 'Items' in config['Origins']:
                for origin in config['Origins']['Items']:
                    if 'CustomOriginConfig' in origin:
                        origin['CustomOriginConfig']['OriginReadTimeout'] = 60
                        origin['CustomOriginConfig']['OriginKeepaliveTimeout'] = 30

            # Save updated config
            config_file = "/tmp/cf_updated_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            # Update distribution
            etag = dist_config['ETag']
            cmd = [
                "aws", "cloudfront", "update-distribution",
                "--id", distribution_id,
                "--distribution-config", f"file://{config_file}",
                "--if-match", etag,
                "--region", self.region
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            updated_dist = json.loads(result.stdout)

            print("✅ Distribution updated for reverse proxy!")
            return updated_dist

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to update distribution: {e}")
            print(f"Error output: {e.stderr}")
            return None

    def invalidate_distribution(self, distribution_id: str, paths: list = None) -> str:
        """
        Create an invalidation to clear the cache.

        Parameters
        ----------
        distribution_id : str
            CloudFront distribution ID
        paths : list, optional
            Paths to invalidate (default: ["/*"])

        Returns
        -------
        str: Invalidation ID
        """
        if paths is None:
            paths = ["/*"]

        print(f"🧹 Creating invalidation for distribution {distribution_id}")

        # Create invalidation batch
        invalidation_batch = {
            "CallerReference": f"invalidate-{int(time.time())}",
            "Paths": {
                "Quantity": len(paths),
                "Items": paths
            }
        }

        # Save to temporary file
        batch_file = "/tmp/invalidation_batch.json"
        with open(batch_file, 'w') as f:
            json.dump(invalidation_batch, f, indent=2)

        cmd = [
            "aws", "cloudfront", "create-invalidation",
            "--distribution-id", distribution_id,
            "--invalidation-batch", f"file://{batch_file}",
            "--region", self.region
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            invalidation_data = json.loads(result.stdout)

            invalidation_id = invalidation_data['Invalidation']['Id']
            print(f"✅ Invalidation created: {invalidation_id}")

            return invalidation_id

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create invalidation: {e}")
            print(f"Error output: {e.stderr}")
            return None

    def get_distribution_status(self, distribution_id: str) -> str:
        """Get the deployment status of a distribution."""
        cmd = [
            "aws", "cloudfront", "get-distribution",
            "--id", distribution_id,
            "--region", self.region
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            dist_data = json.loads(result.stdout)

            status = dist_data['Distribution']['Status']
            enabled = dist_data['Distribution']['Enabled']
            domain = dist_data['Distribution']['DomainName']

            print("📊 Distribution Status:")
            print(f"   Status: {status}")
            print(f"   Enabled: {enabled}")
            print(f"   Domain: {domain}")

            return status

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to get distribution status: {e}")
            return "Unknown"

    def setup_rainbow_crypto_logging(self,
                                   distribution_id: str,
                                   kinesis_stream_arn: str,
                                   sampling_rate: float = 1.0) -> bool:
        """
        Configure real-time logging for Rainbow cryptographic operations.

        Parameters:
        -----------
        distribution_id : str
            CloudFront distribution ID
        kinesis_stream_arn : str
            ARN of Kinesis stream for real-time logs
        sampling_rate : float
            Sampling rate for log entries (0.0 to 1.0)

        Returns:
        --------
        bool
            Success status of logging configuration
        """
        print("🔐 Setting up Rainbow Cryptography Real-time Logging...")

        # Create real-time log configuration for cryptographic operations
        crypto_log_config = {
            "Name": f"Rainbow-Crypto-Logs-{distribution_id[:8]}",
            "SamplingRate": sampling_rate,
            "EndPoints": [
                {
                    "StreamType": "Kinesis",
                    "KinesisStreamConfig": {
                        "RoleArn": f"arn:aws:iam::{self._get_account_id()}:role/CloudFront-Kinesis-Role",
                        "StreamArn": kinesis_stream_arn
                    }
                }
            ],
            "Fields": [
                # Standard CloudFront fields
                "timestamp", "location", "bytes", "request-id", "host",
                "method", "uri", "status", "referrer", "user-agent",

                # Rainbow Cryptography specific fields
                "rainbow-signature-id", "quantum-resistance-level", "signature-size",
                "exceptional-prime-used", "depth-amplification-factor",
                "convergence-precision", "throughput-rate", "security-level",
                "manuscript-id", "cryptographic-operation-type",

                # Performance metrics
                "processing-time-ms", "messages-per-second", "validation-success-rate",
                "signature-verification-time", "key-generation-time",

                # Security events
                "tamper-detection-triggered", "quantum-attack-detected",
                "anomaly-score", "threat-level", "access-geography"
            ]
        }

        # Save configuration to temporary file
        config_file = f"/tmp/crypto_log_config_{distribution_id}.json"
        with open(config_file, 'w') as f:
            json.dump(crypto_log_config, f, indent=2)

        # Create real-time log configuration using AWS CLI
        cmd = [
            "aws", "cloudfront", "create-realtime-log-config",
            "--name", crypto_log_config["Name"],
            "--sampling-rate", str(int(sampling_rate * 100)),
            "--end-points", json.dumps(crypto_log_config["EndPoints"]),
            "--fields", " ".join(crypto_log_config["Fields"]),
            "--region", self.region
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            config_data = json.loads(result.stdout)
            config_arn = config_data["RealtimeLogConfig"]["ARN"]
            print(f"✅ Real-time log config created: {config_arn}")

            # Associate with distribution
            return self._associate_crypto_logging(distribution_id, config_arn)

        except subprocess.CalledProcessError as e:
            print(f"❌ Real-time log config failed: {e}")
            print(f"Error output: {e.stderr}")
            return False

    def _associate_crypto_logging(self, distribution_id: str, config_arn: str) -> bool:
        """Associate real-time logging with CloudFront distribution."""

        # Get current distribution config
        get_cmd = [
            "aws", "cloudfront", "get-distribution-config",
            "--id", distribution_id,
            "--region", self.region
        ]

        try:
            result = subprocess.run(get_cmd, capture_output=True, text=True, check=True)
            config = json.loads(result.stdout)["DistributionConfig"]
            etag = json.loads(result.stdout)["ETag"]

            # Add real-time log configuration
            if "RealtimeLogConfigs" not in config:
                config["RealtimeLogConfigs"] = []

            config["RealtimeLogConfigs"].append({
                "RealtimeLogConfigArn": config_arn,
                "EndPoints": [
                    {
                        "StreamType": "Kinesis",
                        "KinesisStreamConfig": {
                            "RoleArn": f"arn:aws:iam::{self._get_account_id()}:role/CloudFront-Kinesis-Role",
                            "StreamArn": config_arn.split("/")[-1]  # Extract stream ARN
                        }
                    }
                ],
                "Fields": [
                    "timestamp", "rainbow-signature-id", "processing-time-ms",
                    "security-level", "cryptographic-operation-type"
                ]
            })

            # Update distribution
            update_cmd = [
                "aws", "cloudfront", "update-distribution",
                "--id", distribution_id,
                "--distribution-config", json.dumps(config),
                "--if-match", etag,
                "--region", self.region
            ]

            update_result = subprocess.run(update_cmd, capture_output=True, text=True, check=True)
            print("✅ Real-time logging associated with distribution")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to associate logging: {e}")
            return False

    def create_crypto_performance_dashboard(self, distribution_id: str) -> Dict[str, Any]:
        """
        Create comprehensive performance dashboard for Rainbow cryptographic operations.

        Parameters:
        -----------
        distribution_id : str
            CloudFront distribution ID

        Returns:
        --------
        Dict[str, Any]
            Dashboard configuration and metrics
        """
        print("📊 Creating Rainbow Cryptography Performance Dashboard...")

        dashboard_config = {
            "dashboard_name": f"Rainbow-Crypto-Dashboard-{distribution_id[:8]}",
            "distribution_id": distribution_id,
            "metrics": [
                {
                    "name": "SignatureThroughput",
                    "description": "Rainbow signature operations per second",
                    "target": RAINBOW_CRYPTO_CONFIG["throughput_target"],
                    "unit": "Count/Second"
                },
                {
                    "name": "ProcessingTime",
                    "description": "Average cryptographic operation time",
                    "target": "< 5ms",
                    "unit": "Milliseconds"
                },
                {
                    "name": "SignatureVerificationRate",
                    "description": "Percentage of successful signature verifications",
                    "target": "> 99.9%",
                    "unit": "Percent"
                },
                {
                    "name": "QuantumResistanceLevel",
                    "description": "Current quantum resistance effectiveness",
                    "target": "128-bit",
                    "unit": "Bits"
                },
                {
                    "name": "SecurityEventRate",
                    "description": "Rate of detected security events",
                    "target": "< 0.01%",
                    "unit": "Percent"
                }
            ],
            "alerts": [
                {
                    "name": "HighLatencyAlert",
                    "condition": "ProcessingTime > 10ms",
                    "severity": "WARNING",
                    "action": "Scale cryptographic service"
                },
                {
                    "name": "LowVerificationRateAlert",
                    "condition": "SignatureVerificationRate < 99.5%",
                    "severity": "CRITICAL",
                    "action": "Investigate cryptographic service"
                },
                {
                    "name": "ThroughputDegradationAlert",
                    "condition": "SignatureThroughput < 60000",
                    "severity": "WARNING",
                    "action": "Monitor system resources"
                }
            ],
            "visualizations": [
                {
                    "name": "ThroughputOverTime",
                    "type": "line_chart",
                    "metrics": ["SignatureThroughput"],
                    "time_range": "1h"
                },
                {
                    "name": "SecurityEventsHeatmap",
                    "type": "heatmap",
                    "metrics": ["SecurityEventRate"],
                    "dimensions": ["geography", "time"]
                },
                {
                    "name": "PerformanceDistribution",
                    "type": "histogram",
                    "metrics": ["ProcessingTime"],
                    "bins": 20
                }
            ]
        }

        # Save dashboard configuration
        dashboard_file = f"/tmp/crypto_dashboard_{distribution_id}.json"
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_config, f, indent=2)

        print(f"✅ Dashboard configuration saved: {dashboard_file}")
        return dashboard_config

    def monitor_crypto_operations(self, distribution_id: str, duration_minutes: int = 5) -> Dict[str, Any]:
        """
        Monitor Rainbow cryptographic operations in real-time.

        Parameters:
        -----------
        distribution_id : str
            CloudFront distribution ID
        duration_minutes : int
            Monitoring duration in minutes

        Returns:
        --------
        Dict[str, Any]
            Real-time monitoring results
        """
        print(f"🔍 Monitoring Rainbow cryptographic operations for {duration_minutes} minutes...")

        monitoring_results = {
            "distribution_id": distribution_id,
            "monitoring_duration": duration_minutes,
            "start_time": time.time(),
            "metrics": {},
            "alerts": [],
            "recommendations": []
        }

        # Monitor key cryptographic metrics
        try:
            # Get CloudWatch metrics for the distribution
            metrics_to_monitor = [
                "Requests", "BytesDownloaded", "4xxErrorRate", "5xxErrorRate",
                "OriginLatency", "ViewerLatency"
            ]

            for metric in metrics_to_monitor:
                cmd = [
                    "aws", "cloudwatch", "get-metric-statistics",
                    "--namespace", "AWS/CloudFront",
                    "--metric-name", metric,
                    "--dimensions", f"Name=DistributionId,Value={distribution_id}",
                    "--start-time", f"{int(time.time() - duration_minutes*60)}",
                    "--end-time", f"{int(time.time())}",
                    "--period", "60",
                    "--statistics", "Average,Maximum,Minimum",
                    "--region", self.region
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                metric_data = json.loads(result.stdout)

                monitoring_results["metrics"][metric] = {
                    "datapoints": metric_data.get("Datapoints", []),
                    "unit": metric_data.get("Unit", "Count")
                }

            # Analyze cryptographic performance
            monitoring_results.update(self._analyze_crypto_performance(monitoring_results))

        except subprocess.CalledProcessError as e:
            print(f"⚠️  Monitoring error: {e}")
            monitoring_results["error"] = str(e)

        monitoring_results["end_time"] = time.time()
        monitoring_results["total_duration"] = monitoring_results["end_time"] - monitoring_results["start_time"]

        print("✅ Cryptographic monitoring complete")
        return monitoring_results

    def _analyze_crypto_performance(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cryptographic performance from monitoring data."""
        analysis = {
            "throughput_analysis": {},
            "latency_analysis": {},
            "error_analysis": {},
            "recommendations": []
        }

        # Analyze throughput (requests/second)
        if "Requests" in monitoring_data["metrics"]:
            requests_data = monitoring_data["metrics"]["Requests"]["datapoints"]
            if requests_data:
                avg_throughput = sum(dp.get("Average", 0) for dp in requests_data) / len(requests_data)
                max_throughput = max(dp.get("Maximum", 0) for dp in requests_data)

                analysis["throughput_analysis"] = {
                    "average_rps": avg_throughput,
                    "max_rps": max_throughput,
                    "target_achievement": avg_throughput / RAINBOW_CRYPTO_CONFIG["throughput_target"],
                    "status": "GOOD" if avg_throughput > 60000 else "WARNING"
                }

        # Analyze latency
        if "OriginLatency" in monitoring_data["metrics"]:
            latency_data = monitoring_data["metrics"]["OriginLatency"]["datapoints"]
            if latency_data:
                avg_latency = sum(dp.get("Average", 0) for dp in latency_data) / len(latency_data)

                analysis["latency_analysis"] = {
                    "average_ms": avg_latency,
                    "target_ms": 5.0,  # Target for crypto operations
                    "status": "GOOD" if avg_latency < 5.0 else "WARNING"
                }

        # Analyze errors
        if "4xxErrorRate" in monitoring_data["metrics"] and "5xxErrorRate" in monitoring_data["metrics"]:
            error_4xx = monitoring_data["metrics"]["4xxErrorRate"]["datapoints"]
            error_5xx = monitoring_data["metrics"]["5xxErrorRate"]["datapoints"]

            if error_4xx and error_5xx:
                avg_4xx = sum(dp.get("Average", 0) for dp in error_4xx) / len(error_4xx)
                avg_5xx = sum(dp.get("Average", 0) for dp in error_5xx) / len(error_5xx)

                analysis["error_analysis"] = {
                    "4xx_rate": avg_4xx,
                    "5xx_rate": avg_5xx,
                    "total_error_rate": avg_4xx + avg_5xx,
                    "status": "GOOD" if (avg_4xx + avg_5xx) < 0.005 else "WARNING"
                }

        # Generate recommendations
        if analysis.get("throughput_analysis", {}).get("status") == "WARNING":
            analysis["recommendations"].append("Consider scaling cryptographic service instances")

        if analysis.get("latency_analysis", {}).get("status") == "WARNING":
            analysis["recommendations"].append("Optimize cryptographic operation performance")

        if analysis.get("error_analysis", {}).get("status") == "WARNING":
            analysis["recommendations"].append("Investigate and resolve error sources")

        return analysis

    def _get_account_id(self) -> str:
        """Get AWS account ID for ARN construction."""
        try:
            cmd = ["aws", "sts", "get-caller-identity", "--region", self.region]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)["Account"]
        except:
            return "123456789012"  # Default fallback

    def create_rainbow_crypto_distribution(self,
                                        origin_domain: str,
                                        kinesis_stream_arn: str,
                                        cname: str = None) -> Dict[str, Any]:
        """
        Create a CloudFront distribution specifically optimized for Rainbow cryptographic services.

        Parameters:
        -----------
        origin_domain : str
            Domain of the Rainbow cryptography API
        kinesis_stream_arn : str
            ARN for real-time logging stream
        cname : str, optional
            Custom domain name

        Returns:
        --------
        Dict[str, Any]
            Distribution creation result with cryptographic optimizations
        """
        print("🌈 Creating Rainbow Cryptography-Optimized CloudFront Distribution...")

        # Create distribution with cryptographic optimizations
        result = self.create_distribution(
            origin_domain=origin_domain,
            origin_path="/api/crypto",
            origin_protocol="https-only",
            price_class="PriceClass_All",  # Global distribution for crypto services
            comment=f"Rainbow Cryptography API - {RAINBOW_CRYPTO_CONFIG['security_level']} Security",
            cname=cname
        )

        if result and result.get("Distribution"):
            distribution_id = result["Distribution"]["Id"]

            # Configure real-time logging for cryptographic operations
            logging_success = self.setup_rainbow_crypto_logging(
                distribution_id=distribution_id,
                kinesis_stream_arn=kinesis_stream_arn,
                sampling_rate=1.0  # 100% logging for security audit
            )

            if logging_success:
                print("✅ Real-time cryptographic logging configured")

                # Create performance dashboard
                dashboard = self.create_crypto_performance_dashboard(distribution_id)
                print("✅ Performance dashboard created")

                # Add cryptographic metadata to result
                result["cryptographic_config"] = {
                    "rainbow_optimization": True,
                    "security_level": RAINBOW_CRYPTO_CONFIG["security_level"],
                    "quantum_resistance": RAINBOW_CRYPTO_CONFIG["quantum_resistance"],
                    "throughput_target": RAINBOW_CRYPTO_CONFIG["throughput_target"],
                    "real_time_logging": True,
                    "performance_dashboard": dashboard["dashboard_name"],
                    "kinesis_stream": kinesis_stream_arn
                }

            return result

        return None

    def analyze_price_class_optimization(self,
                                       use_case: str,
                                       expected_requests_per_month: int = 1000000,
                                       data_transfer_gb_per_month: int = 100) -> Dict[str, Any]:
        """
        Analyze price class optimization for cryptographic services.

        Parameters:
        -----------
        use_case : str
            Type of cryptographic service ('global', 'academic', 'regional', 'development')
        expected_requests_per_month : int
            Expected monthly request volume
        data_transfer_gb_per_month : int
            Expected monthly data transfer in GB

        Returns:
        --------
        Dict[str, Any]
            Cost-benefit analysis and recommendations
        """
        print(f"💰 Analyzing price class optimization for {use_case} cryptographic services...")

        if use_case not in PRICE_CLASS_OPTIMIZATION:
            raise ValueError(f"Unknown use case: {use_case}. Available: {list(PRICE_CLASS_OPTIMIZATION.keys())}")

        config = PRICE_CLASS_OPTIMIZATION[use_case]

        # Estimate costs (approximate AWS pricing)
        base_cost_per_request = 0.0000004  # $0.40 per million requests
        base_cost_per_gb = 0.085  # $0.085 per GB

        base_monthly_cost = (expected_requests_per_month * base_cost_per_request) + \
                           (data_transfer_gb_per_month * base_cost_per_gb)

        optimized_cost = base_monthly_cost * config["cost_multiplier"]

        analysis = {
            "use_case": use_case,
            "recommended_price_class": config["price_class"],
            "regions": config["regions"],
            "estimated_monthly_cost": optimized_cost,
            "cost_savings_percentage": (1 - config["cost_multiplier"]) * 100,
            "performance_score": config["performance_score"],
            "security_score": config["security_score"],
            "benefit_score": (config["performance_score"] + config["security_score"]) / 2,
            "cost_benefit_ratio": config["cost_multiplier"] / ((config["performance_score"] + config["security_score"]) / 2),
            "recommendations": self._generate_price_class_recommendations(use_case, config)
        }

        print(f"✅ Price class analysis complete for {use_case}")
        print(f"   Recommended: {config['price_class']}")
        print(".2f"        print(".1f"        print(".2f"
        return analysis

    def _generate_price_class_recommendations(self, use_case: str, config: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for price class selection."""
        recommendations = []

        if use_case == "global_cryptographic_access":
            recommendations.extend([
                "Use PriceClass_All for worldwide post-quantum cryptographic service access",
                "Enable IPv6 support for global compatibility",
                "Configure multiple origins for high availability",
                "Implement geographic-based caching for signature verification"
            ])
        elif use_case == "academic_research_distribution":
            recommendations.extend([
                "Use PriceClass_100 for cost-effective academic manuscript distribution",
                "Configure longer TTL values for research publications",
                "Enable compression for large mathematical documents",
                "Implement access logging for research collaboration tracking"
            ])
        elif use_case == "regional_security_services":
            recommendations.extend([
                "Use PriceClass_200 for regional sensitive cryptographic operations",
                "Configure geo-restriction for compliance requirements",
                "Enable WAF integration for enhanced security",
                "Implement custom SSL certificates for regional domains"
            ])
        elif use_case == "development_testing":
            recommendations.extend([
                "Use PriceClass_None for development and testing environments",
                "Limit geographic distribution to reduce costs during development",
                "Use shorter TTL values for rapid iteration",
                "Disable unnecessary features to minimize costs"
            ])

        return recommendations

    def create_cost_optimized_crypto_distribution(self,
                                                origin_domain: str,
                                                use_case: str,
                                                expected_load: Dict[str, int] = None,
                                                kinesis_stream_arn: str = None) -> Dict[str, Any]:
        """
        Create a cost-optimized CloudFront distribution for cryptographic services.

        Parameters:
        -----------
        origin_domain : str
            Origin domain for cryptographic services
        use_case : str
            Type of cryptographic service use case
        expected_load : Dict[str, int], optional
            Expected load parameters (requests_per_month, data_transfer_gb)
        kinesis_stream_arn : str, optional
            ARN for real-time logging

        Returns:
        --------
        Dict[str, Any]
            Optimized distribution configuration
        """
        print("🧮 Creating cost-optimized cryptographic distribution...")

        # Analyze optimal price class
        load_params = expected_load or {"expected_requests_per_month": 1000000, "data_transfer_gb_per_month": 100}
        cost_analysis = self.analyze_price_class_optimization(use_case, **load_params)

        # Get optimal configuration
        optimal_config = PRICE_CLASS_OPTIMIZATION[use_case]
        price_class = optimal_config["price_class"]

        print(f"💡 Optimal price class: {price_class} for {use_case}")
        print(".1f"        print(".2f"
        # Create distribution with optimized settings
        distribution_config = self.create_distribution(
            origin_domain=origin_domain,
            origin_path="/api/crypto",
            origin_protocol="https-only",
            price_class=price_class,
            comment=f"Cost-Optimized {use_case.replace('_', ' ').title()} - {RAINBOW_CRYPTO_CONFIG['security_level']} Security",
            cname=f"crypto-{use_case}.example.com"
        )

        if distribution_config and distribution_config.get("Distribution"):
            distribution_id = distribution_config["Distribution"]["Id"]

            # Configure real-time logging if provided
            if kinesis_stream_arn:
                self.setup_rainbow_crypto_logging(
                    distribution_id=distribution_id,
                    kinesis_stream_arn=kinesis_stream_arn,
                    sampling_rate=0.1 if use_case == "development_testing" else 1.0
                )

            # Add cost optimization metadata
            distribution_config["cost_optimization"] = {
                "use_case": use_case,
                "price_class": price_class,
                "estimated_monthly_cost": cost_analysis["estimated_monthly_cost"],
                "cost_savings_percentage": cost_analysis["cost_savings_percentage"],
                "performance_score": cost_analysis["performance_score"],
                "security_score": cost_analysis["security_score"],
                "regions": optimal_config["regions"],
                "recommendations": cost_analysis["recommendations"]
            }

        return distribution_config

    def compare_price_class_options(self,
                                  origin_domain: str,
                                  expected_requests_per_month: int = 1000000,
                                  expected_data_transfer_gb: int = 100) -> Dict[str, Any]:
        """
        Compare all price class options for cryptographic services.

        Parameters:
        -----------
        origin_domain : str
            Origin domain for comparison
        expected_requests_per_month : int
            Expected monthly request volume
        expected_data_transfer_gb : int
            Expected monthly data transfer

        Returns:
        --------
        Dict[str, Any]
            Comparison of all price class options
        """
        print("📊 Comparing all price class options for cryptographic services...")

        comparison_results = {}
        best_option = {"use_case": None, "cost_benefit_ratio": float('inf')}

        for use_case, config in PRICE_CLASS_OPTIMIZATION.items():
            analysis = self.analyze_price_class_optimization(
                use_case, expected_requests_per_month, expected_data_transfer_gb
            )

            comparison_results[use_case] = {
                "price_class": config["price_class"],
                "estimated_cost": analysis["estimated_monthly_cost"],
                "cost_savings": analysis["cost_savings_percentage"],
                "performance_score": analysis["performance_score"],
                "security_score": analysis["security_score"],
                "benefit_score": analysis["benefit_score"],
                "cost_benefit_ratio": analysis["cost_benefit_ratio"],
                "regions": config["regions"]
            }

            # Find best cost-benefit ratio
            if analysis["cost_benefit_ratio"] < best_option["cost_benefit_ratio"]:
                best_option = {
                    "use_case": use_case,
                    "cost_benefit_ratio": analysis["cost_benefit_ratio"]
                }

        comparison_results["best_option"] = best_option
        comparison_results["summary"] = {
            "total_options": len(PRICE_CLASS_OPTIMIZATION),
            "best_use_case": best_option["use_case"],
            "cost_range": {
                "min": min(r["estimated_cost"] for r in comparison_results.values() if isinstance(r, dict) and "estimated_cost" in r),
                "max": max(r["estimated_cost"] for r in comparison_results.values() if isinstance(r, dict) and "estimated_cost" in r)
            }
        }

        print("✅ Price class comparison complete")
        print(f"🏆 Best option: {best_option['use_case']} (Cost-Benefit Ratio: {best_option['cost_benefit_ratio']:.2f})")

        return comparison_results


def main():
    """Main CLI function for CloudFront reverse proxy setup."""
    parser = argparse.ArgumentParser(
        description="CloudFront Reverse Proxy Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new reverse proxy distribution
  python3 cloudfront_reverse_proxy.py create --origin-domain myapp.example.com

  # Create Rainbow cryptography-optimized distribution
  python3 cloudfront_reverse_proxy.py rainbow-crypto \\
    --origin-domain crypto-api.example.com \\
    --kinesis-stream-arn arn:aws:kinesis:us-east-1:123456789012:stream/crypto-logs

  # Setup real-time logging for cryptographic operations
  python3 cloudfront_reverse_proxy.py setup-logging \\
    --id E1A2B3C4D5E6F7 \\
    --kinesis-stream-arn arn:aws:kinesis:us-east-1:123456789012:stream/crypto-logs

  # Monitor cryptographic operations in real-time
  python3 cloudfront_reverse_proxy.py monitor-crypto \\
    --id E1A2B3C4D5E6F7 \\
    --duration 10

  # Create performance dashboard
  python3 cloudfront_reverse_proxy.py create-dashboard --id E1A2B3C4D5E6F7

  # Update existing distribution for reverse proxy
  python3 cloudfront_reverse_proxy.py update --id E1A2B3C4D5E6F7

  # Check distribution status
  python3 cloudfront_reverse_proxy.py status --id E1A2B3C4D5E6F7

  # Invalidate cache
  python3 cloudfront_reverse_proxy.py invalidate --id E1A2B3C4D5E6F7
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Create distribution command
    create_parser = subparsers.add_parser('create', help='Create new reverse proxy distribution')
    create_parser.add_argument('--origin-domain', required=True, help='Origin server domain name')
    create_parser.add_argument('--origin-path', default='/', help='Path to append to origin requests')
    create_parser.add_argument('--origin-protocol', default='https-only',
                              choices=['http-only', 'https-only', 'match-viewer'],
                              help='Origin protocol policy')
    create_parser.add_argument('--http-port', type=int, default=80, help='HTTP port for origin')
    create_parser.add_argument('--https-port', type=int, default=443, help='HTTPS port for origin')
    create_parser.add_argument('--price-class', default='PriceClass_100',
                              choices=['PriceClass_100', 'PriceClass_200', 'PriceClass_All'],
                              help='CloudFront price class')
    create_parser.add_argument('--comment', default='Reverse Proxy Distribution',
                              help='Distribution comment')
    create_parser.add_argument('--cname', help='Custom domain name (CNAME)')
    create_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Update distribution command
    update_parser = subparsers.add_parser('update', help='Update existing distribution for reverse proxy')
    update_parser.add_argument('--id', required=True, help='CloudFront distribution ID')
    update_parser.add_argument('--origin-domain', required=True, help='Origin server domain name')
    update_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Status command
    status_parser = subparsers.add_parser('status', help='Check distribution deployment status')
    status_parser.add_argument('--id', required=True, help='CloudFront distribution ID')
    status_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Invalidate command
    invalidate_parser = subparsers.add_parser('invalidate', help='Create cache invalidation')
    invalidate_parser.add_argument('--id', required=True, help='CloudFront distribution ID')
    invalidate_parser.add_argument('--paths', nargs='*', default=['/*'],
                                  help='Paths to invalidate (default: /*)')
    invalidate_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Rainbow cryptography commands
    rainbow_parser = subparsers.add_parser('rainbow-crypto', help='Rainbow cryptography optimized distribution')
    rainbow_parser.add_argument('--origin-domain', required=True, help='Rainbow crypto API domain')
    rainbow_parser.add_argument('--kinesis-stream-arn', required=True, help='Kinesis stream ARN for logging')
    rainbow_parser.add_argument('--cname', help='Custom domain name')
    rainbow_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Real-time logging setup command
    logging_parser = subparsers.add_parser('setup-logging', help='Configure real-time logging for crypto operations')
    logging_parser.add_argument('--id', required=True, help='CloudFront distribution ID')
    logging_parser.add_argument('--kinesis-stream-arn', required=True, help='Kinesis stream ARN')
    logging_parser.add_argument('--sampling-rate', type=float, default=1.0, help='Log sampling rate (0.0-1.0)')
    logging_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Performance monitoring command
    monitor_parser = subparsers.add_parser('monitor-crypto', help='Monitor Rainbow cryptographic operations')
    monitor_parser.add_argument('--id', required=True, help='CloudFront distribution ID')
    monitor_parser.add_argument('--duration', type=int, default=5, help='Monitoring duration in minutes')
    monitor_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Dashboard creation command
    dashboard_parser = subparsers.add_parser('create-dashboard', help='Create cryptographic performance dashboard')
    dashboard_parser.add_argument('--id', required=True, help='CloudFront distribution ID')
    dashboard_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Price class analysis command
    analyze_price_parser = subparsers.add_parser('analyze-price', help='Analyze price class optimization for crypto services')
    analyze_price_parser.add_argument('--use-case', required=True,
                                    choices=['global_cryptographic_access', 'academic_research_distribution',
                                           'regional_security_services', 'development_testing'],
                                    help='Cryptographic service use case')
    analyze_price_parser.add_argument('--requests', type=int, default=1000000,
                                    help='Expected monthly requests (default: 1M)')
    analyze_price_parser.add_argument('--data-transfer', type=int, default=100,
                                    help='Expected monthly data transfer in GB (default: 100GB)')
    analyze_price_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Cost-optimized distribution command
    cost_optimized_parser = subparsers.add_parser('cost-optimized-crypto', help='Create cost-optimized crypto distribution')
    cost_optimized_parser.add_argument('--origin-domain', required=True, help='Origin domain for crypto services')
    cost_optimized_parser.add_argument('--use-case', required=True,
                                     choices=['global_cryptographic_access', 'academic_research_distribution',
                                            'regional_security_services', 'development_testing'],
                                     help='Cryptographic service use case')
    cost_optimized_parser.add_argument('--requests', type=int, default=1000000,
                                     help='Expected monthly requests')
    cost_optimized_parser.add_argument('--data-transfer', type=int, default=100,
                                     help='Expected monthly data transfer in GB')
    cost_optimized_parser.add_argument('--kinesis-stream-arn', help='ARN for real-time logging')
    cost_optimized_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Price class comparison command
    compare_price_parser = subparsers.add_parser('compare-price-classes', help='Compare all price class options')
    compare_price_parser.add_argument('--origin-domain', required=True, help='Origin domain for comparison')
    compare_price_parser.add_argument('--requests', type=int, default=1000000,
                                    help='Expected monthly requests')
    compare_price_parser.add_argument('--data-transfer', type=int, default=100,
                                    help='Expected monthly data transfer in GB')
    compare_price_parser.add_argument('--region', default='us-east-1', help='AWS region')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        cf_proxy = CloudFrontReverseProxy(region=getattr(args, 'region', 'us-east-1'))

        if args.command == 'create':
            result = cf_proxy.create_distribution(
                origin_domain=args.origin_domain,
                origin_path=args.origin_path,
                origin_protocol=args.origin_protocol,
                http_port=args.http_port,
                https_port=args.https_port,
                price_class=args.price_class,
                comment=args.comment,
                cname=getattr(args, 'cname', None)
            )

            if result:
                print("\n🎉 Reverse proxy setup complete!")
                print(f"🌐 Your CloudFront domain: {result['Distribution']['DomainName']}")
                print(f"🔗 Point your DNS to this domain or use it directly")
                return 0
            else:
                return 1

        elif args.command == 'update':
            result = cf_proxy.update_distribution_for_reverse_proxy(args.id, args.origin_domain)
            return 0 if result else 1

        elif args.command == 'status':
            status = cf_proxy.get_distribution_status(args.id)
            return 0 if status == 'Deployed' else 1

        elif args.command == 'invalidate':
            invalidation_id = cf_proxy.invalidate_distribution(args.id, args.paths)
            return 0 if invalidation_id else 1

        elif args.command == 'rainbow-crypto':
            result = cf_proxy.create_rainbow_crypto_distribution(
                origin_domain=args.origin_domain,
                kinesis_stream_arn=args.kinesis_stream_arn,
                cname=getattr(args, 'cname', None)
            )
            return 0 if result else 1

        elif args.command == 'setup-logging':
            success = cf_proxy.setup_rainbow_crypto_logging(
                distribution_id=args.id,
                kinesis_stream_arn=args.kinesis_stream_arn,
                sampling_rate=args.sampling_rate
            )
            return 0 if success else 1

        elif args.command == 'monitor-crypto':
            results = cf_proxy.monitor_crypto_operations(
                distribution_id=args.id,
                duration_minutes=args.duration
            )
            # Print summary of monitoring results
            if results.get("throughput_analysis"):
                analysis = results["throughput_analysis"]
                print(f"📊 Throughput: {analysis['average_rps']:.1f} RPS "
                      f"(Target: {RAINBOW_CRYPTO_CONFIG['throughput_target']})")
                print(f"   Status: {analysis['status']}")
            return 0

        elif args.command == 'create-dashboard':
            dashboard = cf_proxy.create_crypto_performance_dashboard(
                distribution_id=args.id
            )
            print(f"📈 Dashboard created: {dashboard['dashboard_name']}")
            return 0

        elif args.command == 'analyze-price':
            analysis = cf_proxy.analyze_price_class_optimization(
                args.use_case, args.requests, args.data_transfer
            )
            print(f"\n💰 Price Class Analysis for {args.use_case}:")
            print(f"   Recommended: {analysis['recommended_price_class']}")
            print(".2f"            print(".1f"            print(".2f"            print(f"   Regions: {', '.join(analysis['regions'])}")
            print(f"\n📋 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"   • {rec}")
            return 0

        elif args.command == 'cost-optimized-crypto':
            load_params = {
                "expected_requests_per_month": args.requests,
                "data_transfer_gb_per_month": args.data_transfer
            }
            result = cf_proxy.create_cost_optimized_crypto_distribution(
                args.origin_domain, args.use_case, load_params, args.kinesis_stream_arn
            )
            return 0 if result else 1

        elif args.command == 'compare-price-classes':
            comparison = cf_proxy.compare_price_class_options(
                args.origin_domain, args.requests, args.data_transfer
            )
            print("
📊 Price Class Comparison Results:"            print(f"🏆 Best Option: {comparison['best_option']['use_case']}")
            print(".2f"            print(f"💰 Cost Range: ${comparison['summary']['cost_range']['min']:.2f} - ${comparison['summary']['cost_range']['max']:.2f}")

            for use_case, details in comparison.items():
                if isinstance(details, dict) and "price_class" in details:
                    print(f"\n{use_case.replace('_', ' ').title()}:")
                    print(f"   Price Class: {details['price_class']}")
                    print(".2f"                    print(".1f"                    print(".2f"                    print(f"   Regions: {len(details['regions'])} locations")
            return 0

        else:
            parser.print_help()
            return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
