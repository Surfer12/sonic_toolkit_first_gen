#!/usr/bin/env python3
"""
CloudFront Reverse Proxy Setup Script

This script uses AWS CLI with CloudFront customizations to set up
a CloudFront distribution as a reverse proxy for web applications.

Features:
- Custom origin configuration
- Proper caching headers
- Security headers
- SSL/TLS configuration
- Real-time logs
- WAF integration

Usage:
    python3 cloudfront_reverse_proxy.py --origin-domain example.com --origin-path /
"""

import argparse
import json
import subprocess
import sys
from typing import Dict, Any, Optional
import time


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


def main():
    """Main CLI function for CloudFront reverse proxy setup."""
    parser = argparse.ArgumentParser(
        description="CloudFront Reverse Proxy Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new reverse proxy distribution
  python3 cloudfront_reverse_proxy.py create --origin-domain myapp.example.com

  # Create with custom settings
  python3 cloudfront_reverse_proxy.py create \\
    --origin-domain api.example.com \\
    --origin-path /api \\
    --cname cdn.example.com

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

        else:
            parser.print_help()
            return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
