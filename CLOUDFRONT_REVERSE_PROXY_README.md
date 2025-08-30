# CloudFront Reverse Proxy Setup

A comprehensive solution for setting up CloudFront distributions as reverse proxies for web applications, APIs, and microservices.

## 🚀 Overview

This toolkit provides automated setup and management of CloudFront distributions configured as reverse proxies. It includes:

- **Automated Distribution Creation**: Pre-configured distributions for reverse proxy use cases
- **Security Headers**: Proper security headers and SSL/TLS configuration
- **Caching Optimization**: Smart caching rules for dynamic content
- **Monitoring & Management**: Tools for monitoring, updating, and maintaining distributions
- **Demo & Testing**: Complete demonstration with real-world examples

## 📋 Features

### Core Functionality
- ✅ **One-command setup** for reverse proxy distributions
- ✅ **Custom origin configuration** with protocol policies
- ✅ **Header forwarding** for proper reverse proxy behavior
- ✅ **SSL/TLS configuration** with modern security standards
- ✅ **Caching rules** optimized for dynamic content
- ✅ **IPv6 support** and HTTP/2 enabled by default

### Management Tools
- ✅ **Distribution status monitoring**
- ✅ **Cache invalidation** with custom path support
- ✅ **Configuration updates** for existing distributions
- ✅ **Performance metrics** and logging integration
- ✅ **Custom domain (CNAME)** support with SSL certificates

### Security & Performance
- ✅ **Security headers** (X-Forwarded-Host, X-Real-IP, etc.)
- ✅ **Geo-restriction** and WAF integration ready
- ✅ **Compression enabled** for faster content delivery
- ✅ **Origin timeouts** optimized for reverse proxy
- ✅ **Keep-alive connections** for better performance

## 🛠️ Installation

### Prerequisites
```bash
# AWS CLI v2.x
aws --version  # Should show 2.x.x

# Python 3.7+
python3 --version

# Required packages
pip install requests boto3
```

### Setup
```bash
# Clone or download the scripts
# cloudfront_reverse_proxy.py
# demo_cloudfront_proxy.py
```

### AWS Configuration
```bash
# Configure AWS credentials
aws configure

# Set your preferred region
aws configure set region us-east-1
```

## 🚀 Quick Start

### Basic Reverse Proxy Setup
```bash
# Create a reverse proxy for your application
python3 cloudfront_reverse_proxy.py create --origin-domain myapp.example.com

# Output:
# 🚀 Creating CloudFront distribution for origin: myapp.example.com
# 📝 Distribution config saved to /tmp/cf_config.json
# ✅ Distribution created successfully!
#    Distribution ID: E1A2B3C4D5E6F7
#    Domain Name: d1234567890abc.cloudfront.net
#    Status: InProgress
```

### Advanced Setup with Custom Domain
```bash
# Create with custom domain and specific settings
python3 cloudfront_reverse_proxy.py create \
  --origin-domain api.example.com \
  --origin-path /api/v1 \
  --cname api.cdn.example.com \
  --price-class PriceClass_All \
  --comment "API Gateway Reverse Proxy"
```

### Run the Demonstration
```bash
# Dry run (no AWS resources created)
python3 demo_cloudfront_proxy.py

# Real deployment (requires AWS credentials)
python3 demo_cloudfront_proxy.py --real-aws --origin-domain your-app.com
```

## 📖 Usage Guide

### Command Reference

#### Create Distribution
```bash
python3 cloudfront_reverse_proxy.py create [OPTIONS]

Options:
  --origin-domain TEXT      Origin server domain name [required]
  --origin-path TEXT        Path to append to origin requests [default: /]
  --origin-protocol TEXT    Origin protocol policy [default: https-only]
  --http-port INTEGER      HTTP port for origin [default: 80]
  --https-port INTEGER     HTTPS port for origin [default: 443]
  --price-class TEXT        CloudFront price class [default: PriceClass_100]
  --comment TEXT           Distribution comment [default: Reverse Proxy Distribution]
  --cname TEXT             Custom domain name (CNAME)
  --region TEXT            AWS region [default: us-east-1]
```

#### Update Distribution
```bash
python3 cloudfront_reverse_proxy.py update [OPTIONS]

Options:
  --id TEXT                CloudFront distribution ID [required]
  --origin-domain TEXT     Origin server domain name [required]
  --region TEXT            AWS region [default: us-east-1]
```

#### Check Status
```bash
python3 cloudfront_reverse_proxy.py status [OPTIONS]

Options:
  --id TEXT                CloudFront distribution ID [required]
  --region TEXT            AWS region [default: us-east-1]
```

#### Invalidate Cache
```bash
python3 cloudfront_reverse_proxy.py invalidate [OPTIONS]

Options:
  --id TEXT                CloudFront distribution ID [required]
  --paths TEXT             Paths to invalidate [default: /*]
  --region TEXT            AWS region [default: us-east-1]
```

## 🏗️ Architecture

### Reverse Proxy Configuration
```
Internet → CloudFront Distribution → Custom Origin → Your Application
```

### Key Configuration Details

#### Origin Configuration
```json
{
  "DomainName": "your-app.example.com",
  "OriginPath": "/",
  "CustomOriginConfig": {
    "HTTPPort": 80,
    "HTTPSPort": 443,
    "OriginProtocolPolicy": "https-only",
    "OriginReadTimeout": 30,
    "OriginKeepaliveTimeout": 5
  }
}
```

#### Cache Behavior
```json
{
  "ForwardedValues": {
    "QueryString": true,
    "Cookies": {"Forward": "all"},
    "Headers": [
      "Host", "User-Agent", "Accept",
      "X-Forwarded-For", "X-Forwarded-Host"
    ]
  },
  "ViewerProtocolPolicy": "redirect-to-https",
  "MinTTL": 0,
  "DefaultTTL": 86400,
  "MaxTTL": 31536000
}
```

### Security Features
- **TLS 1.2+** minimum protocol version
- **HTTPS redirection** for all viewers
- **Custom headers** for origin identification
- **SSL certificate** support (CloudFront managed or ACM)
- **Security headers** forwarded to origin

## 🔧 Advanced Configuration

### Custom Cache Behaviors
```python
# For API endpoints - different caching rules
"CacheBehaviors": {
  "Quantity": 1,
  "Items": [
    {
      "PathPattern": "/api/*",
      "TargetOriginId": "api-origin",
      "ForwardedValues": {
        "QueryString": true,
        "Cookies": {"Forward": "all"},
        "Headers": {"Quantity": 15, "Items": [...]}
      },
      "MinTTL": 0,
      "DefaultTTL": 300,
      "MaxTTL": 3600
    }
  ]
}
```

### Multiple Origins
```python
# Support multiple backend services
"Origins": {
  "Quantity": 2,
  "Items": [
    {
      "Id": "web-origin",
      "DomainName": "web.example.com"
    },
    {
      "Id": "api-origin",
      "DomainName": "api.example.com"
    }
  ]
}
```

### WAF Integration
```python
# Add WAF for security
"WebACLId": "arn:aws:wafv2:us-east-1:123456789012:regional/webacl/reverse-proxy-waf/12345678-1234-1234-1234-123456789012"
```

## 📊 Monitoring & Analytics

### CloudWatch Metrics
```bash
# View key metrics
aws cloudwatch get-metric-data \
  --metric-data-queries '[
    {
      "Id": "requests",
      "MetricStat": {
        "Metric": {
          "Namespace": "AWS/CloudFront",
          "MetricName": "Requests",
          "Dimensions": [{"Name": "DistributionId", "Value": "YOUR_DIST_ID"}]
        },
        "Period": 300,
        "Stat": "Sum"
      }
    }
  ]' \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ)
```

### Real-time Logs
```python
# Enable real-time logs to Kinesis
{
  "Logging": {
    "Enabled": true,
    "Bucket": "your-logs-bucket",
    "Prefix": "cloudfront-realtime/",
    "IncludeCookies": false
  },
  "RealtimeLogConfigs": [
    {
      "ARN": "arn:aws:cloudfront::123456789012:realtime-log-config/logs-config",
      "Fields": ["timestamp", "location", "bytes", "status"]
    }
  ]
}
```

## 🚀 Deployment Examples

### API Gateway Reverse Proxy
```bash
python3 cloudfront_reverse_proxy.py create \
  --origin-domain api.example.com \
  --origin-path /prod \
  --cname api.cdn.example.com \
  --comment "API Gateway Reverse Proxy"
```

### Microservices Architecture
```bash
# Web service
python3 cloudfront_reverse_proxy.py create \
  --origin-domain web-service.com \
  --cname web.example.com

# API service
python3 cloudfront_reverse_proxy.py create \
  --origin-domain api-service.com \
  --cname api.example.com

# Static assets
python3 cloudfront_reverse_proxy.py create \
  --origin-domain static.example.com \
  --origin-path /assets
```

### Global CDN Setup
```bash
python3 cloudfront_reverse_proxy.py create \
  --origin-domain global-app.com \
  --price-class PriceClass_All \
  --comment "Global CDN Reverse Proxy"
```

## 🔍 Troubleshooting

### Common Issues

#### Distribution Not Deploying
```bash
# Check status
python3 cloudfront_reverse_proxy.py status --id YOUR_DIST_ID

# Check AWS console for detailed error messages
# Common issues: SSL certificate, permissions, origin unreachable
```

#### Origin Connection Issues
```bash
# Test origin connectivity
curl -I https://your-origin.com

# Check CloudFront origin settings
aws cloudfront get-distribution-config --id YOUR_DIST_ID
```

#### Cache Not Invalidating
```bash
# Force complete cache invalidation
python3 cloudfront_reverse_proxy.py invalidate --id YOUR_DIST_ID --paths "/*"

# Check invalidation status
aws cloudfront get-invalidation --distribution-id YOUR_DIST_ID --id INVALIDATION_ID
```

### Performance Optimization

#### Cache Hit Ratio
```bash
# Monitor cache hit ratio
aws cloudwatch get-metric-data \
  --metric-name CacheHitRate \
  --namespace AWS/CloudFront \
  --dimensions Name=DistributionId,Value=YOUR_DIST_ID \
  --start-time $(date -u -d '24 hours ago') \
  --end-time $(date -u)
```

#### Response Times
```bash
# Monitor origin response times
aws cloudwatch get-metric-data \
  --metric-name OriginResponseTime \
  --namespace AWS/CloudFront \
  --dimensions Name=DistributionId,Value=YOUR_DIST_ID \
  --start-time $(date -u -d '1 hour ago') \
  --end-time $(date -u)
```

## 📚 Best Practices

### Security
- ✅ **Always use HTTPS** with redirect-to-https
- ✅ **Enable security headers** (CSP, HSTS, etc.)
- ✅ **Use custom SSL certificates** for branded domains
- ✅ **Implement WAF rules** for API protection
- ✅ **Enable logging** for security monitoring

### Performance
- ✅ **Enable compression** for text-based content
- ✅ **Use appropriate TTL values** for different content types
- ✅ **Forward necessary headers** for dynamic content
- ✅ **Implement cache behaviors** for different paths
- ✅ **Monitor cache hit ratios** and adjust policies

### Cost Optimization
- ✅ **Choose appropriate price class** for your audience
- ✅ **Set reasonable TTL values** to maximize cache hits
- ✅ **Use compression** to reduce data transfer
- ✅ **Monitor usage patterns** and adjust configurations
- ✅ **Implement cost allocation tags** for tracking

## 🤝 Contributing

### Development Setup
```bash
# Clone the repository
git clone <repository-url>
cd cloudfront-reverse-proxy

# Install dependencies
pip install -r requirements.txt

# Run tests
python3 -m pytest tests/ -v

# Run demo
python3 demo_cloudfront_proxy.py
```

### Code Style
```bash
# Format code
black *.py

# Lint code
flake8 *.py

# Type checking
mypy *.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

This reverse proxy setup is based on AWS CloudFront best practices and real-world implementations. Special thanks to the AWS community and documentation contributors.

---

**Quick Deploy**: `python3 cloudfront_reverse_proxy.py create --origin-domain your-app.com`

**Monitor**: `python3 cloudfront_reverse_proxy.py status --id YOUR_DIST_ID`

**Invalidate**: `python3 cloudfront_reverse_proxy.py invalidate --id YOUR_DIST_ID`
