# Scientific Computing Toolkit - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Scientific Computing Toolkit across various environments, from local development to production enterprise systems.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Containerization Strategy](#containerization-strategy)
4. [Cloud Deployment](#cloud-deployment)
5. [Enterprise Deployment](#enterprise-deployment)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Linux, macOS, or Windows 10/11
- **CPU**: 4-core processor (8+ cores recommended)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 10GB free space
- **Python**: 3.8+ (3.10+ recommended)

#### Recommended Production Requirements
- **OS**: Ubuntu 20.04 LTS or Red Hat Enterprise Linux 8+
- **CPU**: 8-core processor with AVX2 support
- **RAM**: 32GB+ (64GB+ for large-scale processing)
- **Storage**: 100GB+ SSD storage
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional, for accelerated computing)

### Software Dependencies

#### Core Dependencies
```bash
# Python packages
pip install numpy>=1.21.0 scipy>=1.7.0 torch>=1.12.0 matplotlib>=3.5.0
pip install aiohttp websockets fastapi uvicorn
pip install pytest pytest-cov pytest-asyncio

# System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3-dev build-essential git curl wget

# Java dependencies (for Corpus framework)
sudo apt-get install openjdk-17-jdk maven

# Swift dependencies (for iOS framework)
# Install Xcode 14+ on macOS or swift-lang on Linux
```

#### Optional Dependencies
```bash
# GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Database support
pip install sqlalchemy psycopg2-binary

# Monitoring and logging
pip install prometheus-client elasticsearch

# Security enhancements
pip install cryptography pyjwt bcrypt
```

---

## Local Development Setup

### Quick Start Setup

1. **Clone Repository**
```bash
git clone https://github.com/your-org/scientific-computing-toolkit.git
cd scientific-computing-toolkit
```

2. **Environment Setup**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

3. **Configuration Setup**
```bash
# Copy configuration template
cp config.template.json config.json

# Edit configuration for your environment
nano config.json  # or your preferred editor
```

4. **Database Setup** (Optional)
```bash
# Initialize local database
python scripts/init_database.py

# Run migrations
python scripts/migrate_database.py
```

5. **Run Initial Tests**
```bash
# Run basic functionality tests
python complete_integration_test.py

# Run unit tests
python -m pytest tests/ -v
```

### Development Environment Configuration

#### VS Code Configuration
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"]
}
```

#### PyCharm Configuration
```xml
<!-- .idea/misc.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<project version="4">
  <component name="ProjectRootManager" version="2" project-jdk-name="Python 3.10" project-jdk-type="Python SDK" />
</project>
```

---

## Containerization Strategy

### Docker Setup

#### Base Dockerfile
```dockerfile
# Dockerfile for Scientific Computing Toolkit
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose ports
EXPOSE 8000 8080 9001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "main.py"]
```

#### Multi-Stage Build Dockerfile
```dockerfile
# Multi-stage build for optimized production image
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim as production

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application directory
WORKDIR /app

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main.py"]
```

### Docker Compose Configuration

#### Development Environment
```yaml
# docker-compose.yml
version: '3.8'

services:
  scientific-computing:
    build: .
    ports:
      - "8000:8000"
      - "8080:8080"
      - "9001:9001"
    volumes:
      - .:/app
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: scientific_computing
      POSTGRES_USER: sci_user
      POSTGRES_PASSWORD: secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

#### Production Environment
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  scientific-computing:
    image: scientific-computing-toolkit:latest
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
      - DATABASE_URL=postgresql://user:password@postgres:5432/db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: scientific_computing
      POSTGRES_USER: sci_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - scientific-computing
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

### Kubernetes Deployment

#### Deployment Manifests
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: scientific-computing
  labels:
    app: scientific-computing
spec:
  replicas: 3
  selector:
    matchLabels:
      app: scientific-computing
  template:
    metadata:
      labels:
        app: scientific-computing
    spec:
      containers:
      - name: scientific-computing
        image: scientific-computing-toolkit:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service Manifest
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: scientific-computing-service
spec:
  selector:
    app: scientific-computing
  ports:
    - port: 8000
      targetPort: 8000
  type: LoadBalancer
```

#### Ingress Configuration
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: scientific-computing-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: scientific-computing.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: scientific-computing-service
            port:
              number: 8000
```

---

## Cloud Deployment

### AWS Deployment

#### EC2 Setup
```bash
# Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --count 1 \
  --instance-type t3.large \
  --key-name your-key-pair \
  --security-groups scientific-computing-sg

# Configure instance
sudo yum update -y
sudo yum install python3 git -y

# Clone repository
git clone https://github.com/your-org/scientific-computing-toolkit.git
cd scientific-computing-toolkit

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### ECS Configuration
```json
{
  "family": "scientific-computing-task",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "scientific-computing",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/scientific-computing-toolkit:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "DATABASE_URL", "value": "postgresql://..."}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/scientific-computing",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Azure Deployment

#### Web App Configuration
```json
{
  "name": "scientific-computing-app",
  "location": "East US",
  "kind": "app,linux",
  "resourceGroup": "scientific-computing-rg",
  "properties": {
    "serverFarmId": "/subscriptions/.../resourceGroups/.../providers/Microsoft.Web/serverfarms/...",
    "siteConfig": {
      "linuxFxVersion": "PYTHON|3.10",
      "appSettings": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "DATABASE_URL", "value": "..."}
      ],
      "alwaysOn": true
    }
  }
}
```

### GCP Deployment

#### Cloud Run Configuration
```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: scientific-computing
spec:
  template:
    spec:
      containers:
      - image: gcr.io/your-project/scientific-computing-toolkit:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          limits:
            cpu: 1000m
            memory: 1Gi
```

---

## Enterprise Deployment

### Multi-Tier Architecture

#### Application Tier
```yaml
# Enterprise application deployment
version: '3.8'
services:
  web:
    image: scientific-computing-toolkit:latest
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    depends_on:
      - redis
      - database

  worker:
    image: scientific-computing-toolkit:latest
    command: python worker.py
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    depends_on:
      - redis
      - database
    deploy:
      replicas: 3

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  database:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: scientific_computing
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  redis_data:
  db_data:
```

#### Load Balancing Tier
```nginx
# nginx.conf
upstream scientific_computing_backend {
    server web1:8000;
    server web2:8000;
    server web3:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://scientific_computing_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

### High Availability Setup

#### Database Clustering
```yaml
# PostgreSQL cluster configuration
version: '3.8'
services:
  postgres-master:
    image: bitnami/postgresql:15
    environment:
      POSTGRESQL_REPLICATION_MODE: master
      POSTGRESQL_REPLICATION_USER: repl_user
      POSTGRESQL_REPLICATION_PASSWORD: repl_password
      POSTGRESQL_PASSWORD: admin_password

  postgres-slave:
    image: bitnami/postgresql:15
    environment:
      POSTGRESQL_REPLICATION_MODE: slave
      POSTGRESQL_MASTER_HOST: postgres-master
      POSTGRESQL_MASTER_PORT_NUMBER: 5432
      POSTGRESQL_REPLICATION_USER: repl_user
      POSTGRESQL_REPLICATION_PASSWORD: repl_password
      POSTGRESQL_PASSWORD: admin_password
    depends_on:
      - postgres-master
```

#### Redis Sentinel
```yaml
# Redis Sentinel configuration
version: '3.8'
services:
  redis-master:
    image: redis:7-alpine
    command: redis-server /etc/redis/redis.conf
    volumes:
      - ./redis-master.conf:/etc/redis/redis.conf

  redis-slave:
    image: redis:7-alpine
    command: redis-server /etc/redis/redis.conf
    volumes:
      - ./redis-slave.conf:/etc/redis/redis.conf
    depends_on:
      - redis-master

  redis-sentinel:
    image: redis:7-alpine
    command: redis-sentinel /etc/redis/sentinel.conf
    volumes:
      - ./redis-sentinel.conf:/etc/redis/sentinel.conf
    depends_on:
      - redis-master
      - redis-slave
```

---

## Monitoring and Maintenance

### Application Monitoring

#### Health Check Endpoints
```python
# main.py
from fastapi import FastAPI
from scientific_computing_toolkit import framework

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": framework.__version__,
        "uptime": framework.get_uptime(),
        "memory_usage": framework.get_memory_usage()
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    # Check database connectivity
    db_status = await check_database_connection()

    # Check external services
    external_status = await check_external_services()

    if db_status and external_status:
        return {"status": "ready"}
    else:
        return {"status": "not ready"}, 503

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_prometheus_metrics()
```

#### Logging Configuration
```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Configure comprehensive logging."""

    # Create logger
    logger = logging.getLogger('scientific_computing')
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/scientific_computing.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d'
    )
    file_handler.setFormatter(file_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
```

### Automated Maintenance

#### Backup Scripts
```bash
#!/bin/bash
# backup.sh

# Database backup
docker exec scientific_computing_postgres_1 pg_dump -U sci_user scientific_computing > backup_$(date +%Y%m%d_%H%M%S).sql

# Application data backup
tar -czf application_backup_$(date +%Y%m%d_%H%M%S).tar.gz /app/data /app/logs /app/config

# Upload to cloud storage
aws s3 cp backup_*.sql s3://scientific-computing-backups/database/
aws s3 cp application_backup_*.tar.gz s3://scientific-computing-backups/application/
```

#### Update Scripts
```bash
#!/bin/bash
# update.sh

# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run database migrations
python scripts/migrate_database.py

# Restart services
docker-compose down
docker-compose up -d

# Run tests
python -m pytest tests/ -v

# Health check
curl -f http://localhost:8000/health
```

---

## Troubleshooting

### Common Issues

#### Docker Issues
```bash
# Check Docker status
docker ps -a

# View container logs
docker logs scientific-computing

# Restart containers
docker-compose restart

# Rebuild containers
docker-compose build --no-cache
```

#### Database Issues
```bash
# Check database connectivity
docker exec -it scientific_computing_postgres_1 psql -U sci_user -d scientific_computing

# View database logs
docker logs scientific_computing_postgres_1

# Reset database
docker-compose down -v
docker-compose up -d
```

#### Performance Issues
```bash
# Check system resources
top
df -h
free -h

# Profile application
python -m cProfile -s time main.py

# Check GPU usage (if applicable)
nvidia-smi
```

#### Network Issues
```bash
# Check port availability
netstat -tlnp | grep :8000

# Test connectivity
curl http://localhost:8000/health

# Check firewall settings
sudo ufw status
```

### Debug Mode Configuration

#### Development Debug Setup
```python
# debug_config.py
import logging

DEBUG_CONFIG = {
    'debug': True,
    'log_level': 'DEBUG',
    'database_echo': True,
    'profile_requests': True,
    'cors_origins': ['http://localhost:3000', 'http://localhost:8080']
}

def setup_debug_mode(app):
    """Configure application for debug mode."""

    # Enable debug logging
    logging.basicConfig(level=logging.DEBUG)

    # Enable CORS for frontend development
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEBUG_CONFIG['cors_origins'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Enable request profiling
    if DEBUG_CONFIG['profile_requests']:
        from fastapi.middleware.base import BaseHTTPMiddleware

        class ProfilingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                import time
                start_time = time.time()
                response = await call_next(request)
                process_time = time.time() - start_time
                logger.info(f"Request processing time: {process_time:.3f}s")
                return response

        app.add_middleware(ProfilingMiddleware)

    return app
```

---

## Security Considerations

### Production Security

#### SSL/TLS Configuration
```nginx
# nginx.conf for SSL
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;

    location / {
        proxy_pass http://scientific-computing:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Secrets Management
```python
# secrets.py
import os
from dotenv import load_dotenv

load_dotenv()

class SecretsManager:
    """Manage application secrets securely."""

    @staticmethod
    def get_database_url():
        """Get database URL from environment or secrets manager."""
        return os.getenv('DATABASE_URL')

    @staticmethod
    def get_jwt_secret():
        """Get JWT secret key."""
        return os.getenv('JWT_SECRET')

    @staticmethod
    def get_api_keys():
        """Get API keys for external services."""
        return {
            'openai': os.getenv('OPENAI_API_KEY'),
            'aws': os.getenv('AWS_ACCESS_KEY_ID'),
            'stripe': os.getenv('STRIPE_SECRET_KEY')
        }
```

---

## Performance Optimization

### Production Optimizations

#### Gunicorn Configuration
```python
# gunicorn.conf.py
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Logging
loglevel = "info"
accesslog = "/var/log/scientific-computing/access.log"
errorlog = "/var/log/scientific-computing/error.log"

# Process naming
proc_name = "scientific-computing"

# Server mechanics
daemon = False
pidfile = "/var/run/scientific-computing.pid"
user = "www-data"
group = "www-data"
tmp_upload_dir = None
```

#### Database Optimization
```python
# database_config.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

def create_optimized_engine(database_url: str):
    """Create optimized database engine."""

    return create_engine(
        database_url,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=3600,
        echo=False,  # Disable in production
        connect_args={
            "connect_timeout": 10,
            "read_timeout": 30,
            "write_timeout": 30
        }
    )
```

---

## Backup and Recovery

### Automated Backup Strategy
```bash
#!/bin/bash
# automated_backup.sh

# Configuration
BACKUP_DIR="/opt/backups/scientific-computing"
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
echo "Creating database backup..."
docker exec scientific_computing_postgres_1 pg_dump -U sci_user scientific_computing | gzip > $BACKUP_DIR/db_backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Application data backup
echo "Creating application backup..."
tar -czf $BACKUP_DIR/app_backup_$(date +%Y%m%d_%H%M%S).tar.gz /app/data /app/logs /app/config

# Configuration backup
echo "Creating configuration backup..."
cp /app/config.json $BACKUP_DIR/config_backup_$(date +%Y%m%d_%H%M%S).json

# Clean up old backups
echo "Cleaning up old backups..."
find $BACKUP_DIR -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
find $BACKUP_DIR -name "*.json" -mtime +$RETENTION_DAYS -delete

echo "Backup completed successfully"
```

### Disaster Recovery Plan
```yaml
# disaster_recovery.yaml
recovery_plan:
  rto: 4  # Recovery Time Objective (hours)
  rpo: 1  # Recovery Point Objective (hours)

  backup_strategy:
    frequency: "daily"
    type: "incremental"
    retention: 30  # days
    location: "s3://scientific-computing-backups"

  recovery_procedures:
    - step: "Assess damage and isolate affected systems"
    - step: "Restore from latest backup"
    - step: "Verify data integrity"
    - step: "Bring systems online gradually"
    - step: "Validate functionality with test suite"

  contact_information:
    - name: "Primary DBA"
      role: "Database Administrator"
      phone: "+1-555-0101"
      email: "dba@company.com"
    - name: "DevOps Lead"
      role: "Infrastructure Manager"
      phone: "+1-555-0102"
      email: "devops@company.com"
```

---

## Support and Resources

### Documentation Resources
- [API Documentation](./api_reference.md)
- [User Tutorials](./tutorials/)
- [Troubleshooting Guide](./troubleshooting.md)
- [Performance Optimization](./performance_guide.md)

### Community Support
- GitHub Issues: https://github.com/your-org/scientific-computing-toolkit/issues
- Documentation Wiki: https://github.com/your-org/scientific-computing-toolkit/wiki
- Community Forum: https://community.scientific-computing-toolkit.org

### Professional Services
- Enterprise Support: enterprise@scientific-computing-toolkit.com
- Custom Development: development@scientific-computing-toolkit.com
- Training and Consulting: training@scientific-computing-toolkit.com

---

**This deployment guide provides comprehensive instructions for deploying the Scientific Computing Toolkit across various environments. For additional support or custom deployment requirements, please contact our enterprise support team.**
