# Scientific Computing Toolkit - Maintenance and Update Procedures

## Comprehensive Maintenance Guide

This document provides detailed procedures for maintaining, updating, and ensuring the long-term health of the Scientific Computing Toolkit deployment.

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Regular Maintenance Tasks](#regular-maintenance-tasks)
3. [Update Procedures](#update-procedures)
4. [Backup and Recovery](#backup-and-recovery)
5. [Performance Monitoring](#performance-monitoring)
6. [Security Maintenance](#security-maintenance)
7. [Troubleshooting Procedures](#troubleshooting-procedures)
8. [Emergency Response](#emergency-response)
9. [Documentation Updates](#documentation-updates)
10. [Compliance and Auditing](#compliance-and-auditing)

---

## System Architecture Overview

### Component Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web/API       │    │   Processing    │    │   Storage       │
│   Services      │◄──►│   Framework     │◄──►│   Systems       │
│                 │    │                 │    │                 │
│ • REST APIs     │    │ • Hybrid UQ     │    │ • Databases     │
│ • WebSockets    │    │ • Data Pipeline │    │ • File Systems  │
│ • Load Balancer │    │ • Security      │    │ • Cache         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Monitoring    │
                    │   & Alerting    │
                    │                 │
                    │ • Health Checks │
                    │ • Metrics       │
                    │ • Logging       │
                    └─────────────────┘
```

### Key Components

1. **Hybrid UQ Framework**: Core uncertainty quantification engine
2. **Cross-Framework Communication**: Inter-component messaging system
3. **Data Processing Pipeline**: ETL and analysis workflows
4. **Security Framework**: Authentication, authorization, encryption
5. **Monitoring System**: Health checks, alerting, performance metrics
6. **Web Services**: REST APIs, WebSocket connections, user interfaces

---

## Regular Maintenance Tasks

### Daily Maintenance Tasks

#### 1. System Health Checks

```bash
#!/bin/bash
# daily_health_check.sh

echo "=== Daily Health Check - $(date) ==="

# Check service status
echo "Checking service status..."
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check disk usage
echo -e "\nChecking disk usage..."
df -h | grep -E "(Filesystem|/$|/data|/logs)"

# Check memory usage
echo -e "\nChecking memory usage..."
free -h

# Check CPU usage
echo -e "\nChecking CPU usage..."
top -bn1 | head -10

# Check network connectivity
echo -e "\nChecking network connectivity..."
ping -c 3 google.com

# Check application logs for errors
echo -e "\nChecking application logs..."
tail -n 20 /var/log/scientific-computing/*.log | grep -i error || echo "No recent errors found"

echo "=== Health Check Complete ==="
```

#### 2. Database Maintenance

```bash
#!/bin/bash
# daily_db_maintenance.sh

echo "=== Daily Database Maintenance - $(date) ==="

# Backup database
echo "Creating database backup..."
docker exec scientific_computing_postgres_1 pg_dump -U sci_user scientific_computing > /backups/daily_$(date +%Y%m%d_%H%M%S).sql

# Analyze database performance
echo -e "\nAnalyzing database performance..."
docker exec scientific_computing_postgres_1 psql -U sci_user -d scientific_computing -c "SELECT * FROM pg_stat_user_tables ORDER BY n_tup_ins DESC LIMIT 10;"

# Check for long-running queries
echo -e "\nChecking for long-running queries..."
docker exec scientific_computing_postgres_1 psql -U sci_user -d scientific_computing -c "
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE state = 'active' AND now() - pg_stat_activity.query_start > interval '5 minutes'
ORDER BY duration DESC;
"

# Vacuum analyze (if needed)
echo -e "\nRunning vacuum analyze..."
docker exec scientific_computing_postgres_1 psql -U sci_user -d scientific_computing -c "VACUUM ANALYZE;"

echo "=== Database Maintenance Complete ==="
```

#### 3. Log Rotation and Cleanup

```bash
#!/bin/bash
# daily_log_maintenance.sh

echo "=== Daily Log Maintenance - $(date) ==="

# Rotate application logs
echo "Rotating application logs..."
find /var/log/scientific-computing -name "*.log" -size +100M -exec gzip {} \;
find /var/log/scientific-computing -name "*.log.gz" -mtime +30 -exec rm {} \;

# Rotate container logs
echo -e "\nRotating container logs..."
docker run --rm -v /var/lib/docker/containers:/var/lib/docker/containers \
  alpine:latest sh -c "find /var/lib/docker/containers -name *-json.log -size +50M -exec truncate -s 0 {} \;"

# Clean up old temporary files
echo -e "\nCleaning temporary files..."
find /tmp -name "scientific_computing_*" -mtime +1 -exec rm {} \;
find /app/tmp -type f -mtime +7 -exec rm {} \;

# Archive old results
echo -e "\nArchiving old results..."
find /app/results -type f -mtime +90 -exec mv {} /app/archive/ \;

echo "=== Log Maintenance Complete ==="
```

### Weekly Maintenance Tasks

#### 1. Performance Optimization

```bash
#!/bin/bash
# weekly_performance_optimization.sh

echo "=== Weekly Performance Optimization - $(date) ==="

# Update system packages
echo "Updating system packages..."
apt-get update && apt-get upgrade -y

# Optimize database
echo -e "\nOptimizing database..."
docker exec scientific_computing_postgres_1 psql -U sci_user -d scientific_computing -c "REINDEX DATABASE scientific_computing;"

# Clear system cache
echo -e "\nClearing system cache..."
sync; echo 3 > /proc/sys/vm/drop_caches

# Restart services for fresh state
echo -e "\nRestarting services..."
docker-compose restart

# Run performance benchmarks
echo -e "\nRunning performance benchmarks..."
python3 performance_benchmarks.py --quick

echo "=== Performance Optimization Complete ==="
```

#### 2. Security Updates

```bash
#!/bin/bash
# weekly_security_updates.sh

echo "=== Weekly Security Updates - $(date) ==="

# Update container images
echo "Updating container images..."
docker-compose pull

# Update Python packages
echo -e "\nUpdating Python packages..."
pip list --outdated
pip install --upgrade pip
pip install --upgrade -r requirements.txt

# Update system security packages
echo -e "\nUpdating system security packages..."
apt-get update && apt-get install -y unattended-upgrades
unattended-upgrades --dry-run

# Rotate encryption keys (if applicable)
echo -e "\nRotating encryption keys..."
python3 -c "
from security_framework import security_framework
security_framework.encryption_manager.rotate_key()
print('Encryption keys rotated successfully')
"

# Update SSL certificates
echo -e "\nChecking SSL certificates..."
certbot certificates

echo "=== Security Updates Complete ==="
```

#### 3. Data Integrity Checks

```bash
#!/bin/bash
# weekly_data_integrity.sh

echo "=== Weekly Data Integrity Check - $(date) ==="

# Check database integrity
echo "Checking database integrity..."
docker exec scientific_computing_postgres_1 psql -U sci_user -d scientific_computing -c "
SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del
FROM pg_stat_user_tables
ORDER BY n_tup_ins DESC;
"

# Validate data files
echo -e "\nValidating data files..."
find /app/data -name "*.json" -exec python3 -c "
import json
import sys
try:
    with open(sys.argv[1], 'r') as f:
        json.load(f)
    print(f'✓ {sys.argv[1]}')
except Exception as e:
    print(f'✗ {sys.argv[1]}: {e}')
" {} \;

# Check file system integrity
echo -e "\nChecking file system integrity..."
fsck -n /dev/sda1

# Validate backup integrity
echo -e "\nValidating recent backups..."
ls -la /backups/*.sql | tail -5 | while read line; do
    backup_file=$(echo $line | awk '{print $9}')
    if [ -f "$backup_file" ]; then
        echo "✓ $backup_file"
    else
        echo "✗ $backup_file missing"
    fi
done

echo "=== Data Integrity Check Complete ==="
```

### Monthly Maintenance Tasks

#### 1. Comprehensive System Audit

```bash
#!/bin/bash
# monthly_system_audit.sh

echo "=== Monthly System Audit - $(date) ==="

# System resource audit
echo "System Resource Audit:"
echo "======================"
echo "CPU Cores: $(nproc)"
echo "Total Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "Total Disk: $(df -h / | tail -1 | awk '{print $2}')"
echo "Uptime: $(uptime -p)"

# Service audit
echo -e "\nService Audit:"
echo "=============="
docker ps --format "table {{.Names}}\t{{.Status}}"

# User access audit
echo -e "\nUser Access Audit:"
echo "=================="
last | head -10

# Security audit
echo -e "\nSecurity Audit:"
echo "==============="
echo "Open ports:"
netstat -tlnp | grep LISTEN

echo "Failed login attempts:"
grep "Failed password" /var/log/auth.log | wc -l

# Performance audit
echo -e "\nPerformance Audit:"
echo "=================="
echo "Average CPU usage (last 30 days):"
sar -u | tail -30 | awk '{sum+=$3} END {print sum/NR "%"}'

echo "Average memory usage (last 30 days):"
sar -r | tail -30 | awk '{sum+=$4} END {print sum/NR "%"}'

echo "=== Monthly System Audit Complete ==="
```

#### 2. Backup Verification

```bash
#!/bin/bash
# monthly_backup_verification.sh

echo "=== Monthly Backup Verification - $(date) ==="

# List recent backups
echo "Recent Backups:"
echo "==============="
ls -la /backups/ | grep -E "\.(sql|tar\.gz)$" | tail -10

# Test database backup restoration
echo -e "\nTesting Database Backup Restoration..."
LATEST_BACKUP=$(ls -t /backups/*.sql | head -1)
if [ -f "$LATEST_BACKUP" ]; then
    echo "Testing $LATEST_BACKUP..."
    # Create test database
    docker exec scientific_computing_postgres_1 createdb -U sci_user test_restore

    # Restore backup to test database
    docker exec -i scientific_computing_postgres_1 psql -U sci_user test_restore < $LATEST_BACKUP

    # Verify restoration
    TABLES_COUNT=$(docker exec scientific_computing_postgres_1 psql -U sci_user -d test_restore -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" -t)
    echo "Restored $TABLES_COUNT tables"

    # Clean up test database
    docker exec scientific_computing_postgres_1 dropdb -U sci_user test_restore

    echo "✓ Database backup verification successful"
else
    echo "✗ No database backup found"
fi

# Test file system backup
echo -e "\nTesting File System Backup..."
LATEST_FS_BACKUP=$(ls -t /backups/*.tar.gz | head -1)
if [ -f "$LATEST_FS_BACKUP" ]; then
    echo "Testing $LATEST_FS_BACKUP..."
    mkdir -p /tmp/backup_test
    tar -tzf $LATEST_FS_BACKUP | head -10
    rm -rf /tmp/backup_test
    echo "✓ File system backup verification successful"
else
    echo "✗ No file system backup found"
fi

echo "=== Backup Verification Complete ==="
```

---

## Update Procedures

### Framework Updates

#### 1. Minor Version Updates

```bash
#!/bin/bash
# minor_update_procedure.sh

echo "=== Minor Version Update Procedure ==="

# Backup current state
echo "Creating pre-update backup..."
./backup.sh pre_update_$(date +%Y%m%d_%H%M%S)

# Update application code
echo -e "\nUpdating application code..."
git pull origin main

# Update dependencies
echo -e "\nUpdating dependencies..."
pip install -r requirements.txt --upgrade

# Run database migrations (if any)
echo -e "\nRunning database migrations..."
python3 scripts/migrate_database.py

# Update configuration
echo -e "\nUpdating configuration..."
python3 scripts/update_config.py

# Restart services
echo -e "\nRestarting services..."
docker-compose restart

# Run tests
echo -e "\nRunning tests..."
python3 -m pytest tests/ -v

# Health check
echo -e "\nPerforming health check..."
curl -f http://localhost:8000/health || exit 1

echo "=== Minor Update Complete ==="
```

#### 2. Major Version Updates

```bash
#!/bin/bash
# major_update_procedure.sh

echo "=== Major Version Update Procedure ==="

# Pre-update checklist
echo "Pre-update Checklist:"
echo "===================="
echo "☐ Backup all data and configurations"
echo "☐ Notify users of planned downtime"
echo "☐ Schedule maintenance window"
echo "☐ Prepare rollback plan"
echo "☐ Test update in staging environment"
read -p "Have you completed the pre-update checklist? (y/N): " -n 1 -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please complete the checklist before proceeding."
    exit 1
fi

# Create comprehensive backup
echo -e "\nCreating comprehensive backup..."
./comprehensive_backup.sh major_update_$(date +%Y%m%d_%H%M%S)

# Stop services
echo -e "\nStopping services..."
docker-compose down

# Update application code
echo -e "\nUpdating application code..."
git pull origin main
git checkout v2.0.0  # Replace with actual version

# Update system dependencies
echo -e "\nUpdating system dependencies..."
apt-get update && apt-get upgrade -y

# Update container images
echo -e "\nUpdating container images..."
docker-compose pull

# Run database migrations
echo -e "\nRunning database migrations..."
python3 scripts/migrate_database_v2.py

# Update configuration
echo -e "\nUpdating configuration..."
python3 scripts/migrate_config_v2.py

# Start services
echo -e "\nStarting services..."
docker-compose up -d

# Run comprehensive tests
echo -e "\nRunning comprehensive tests..."
python3 complete_integration_test.py

# Post-update validation
echo -e "\nPerforming post-update validation..."
python3 scripts/post_update_validation.py

# Notify completion
echo -e "\n=== Major Update Complete ==="
echo "Please verify the system is working correctly and notify users."
```

#### 3. Emergency Rollback Procedure

```bash
#!/bin/bash
# emergency_rollback.sh

echo "=== Emergency Rollback Procedure ==="

# Stop services immediately
echo "Stopping all services..."
docker-compose down --timeout 30

# Restore from backup
echo -e "\nRestoring from backup..."
LATEST_BACKUP=$(ls -t /backups/comprehensive_*.tar.gz | head -1)
if [ -f "$LATEST_BACKUP" ]; then
    echo "Restoring from $LATEST_BACKUP..."
    tar -xzf $LATEST_BACKUP -C /
else
    echo "No comprehensive backup found!"
    exit 1
fi

# Restore database
echo -e "\nRestoring database..."
LATEST_DB_BACKUP=$(ls -t /backups/db_*.sql | head -1)
if [ -f "$LATEST_DB_BACKUP" ]; then
    docker-compose up -d postgres
    sleep 10
    docker exec -i scientific_computing_postgres_1 psql -U sci_user scientific_computing < $LATEST_DB_BACKUP
else
    echo "No database backup found!"
fi

# Start services
echo -e "\nStarting services..."
docker-compose up -d

# Verify system health
echo -e "\nVerifying system health..."
curl -f http://localhost:8000/health || echo "Health check failed!"

echo "=== Rollback Complete ==="
echo "Please investigate the cause of the emergency and plan next steps."
```

### Dependency Updates

#### 1. Python Package Updates

```bash
#!/bin/bash
# update_python_packages.sh

echo "=== Python Package Update Procedure ==="

# Check for outdated packages
echo "Checking for outdated packages..."
pip list --outdated

# Create requirements backup
cp requirements.txt requirements.txt.backup

# Update packages with compatibility checking
echo -e "\nUpdating packages..."
pip install --upgrade pip

# Update packages in groups to avoid conflicts
echo "Updating core scientific packages..."
pip install --upgrade numpy scipy matplotlib pandas

echo "Updating web framework packages..."
pip install --upgrade fastapi uvicorn aiohttp

echo "Updating machine learning packages..."
pip install --upgrade torch torchvision

# Test compatibility
echo -e "\nTesting compatibility..."
python3 -c "
import numpy as np
import scipy
import torch
import fastapi
print('✓ All packages imported successfully')
"

# Run tests
echo -e "\nRunning tests..."
python3 -m pytest tests/ -x || {
    echo "Tests failed! Rolling back..."
    pip install -r requirements.txt.backup
    exit 1
}

# Update requirements.txt
echo -e "\nUpdating requirements.txt..."
pip freeze > requirements.txt

echo "=== Python Package Update Complete ==="
```

#### 2. System Package Updates

```bash
#!/bin/bash
# update_system_packages.sh

echo "=== System Package Update Procedure ==="

# Check for available updates
echo "Checking for available updates..."
apt-get update
apt-get --dry-run upgrade | grep "^Inst" | wc -l | xargs echo " packages to update"

# Create system snapshot
echo -e "\nCreating system snapshot..."
docker commit scientific_computing_app scientific_computing_backup:$(date +%Y%m%d_%H%M%S)

# Update packages
echo -e "\nUpdating packages..."
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# Check for kernel updates
echo -e "\nChecking for kernel updates..."
apt-get --dry-run install linux-image-generic | grep "^Inst" || echo "No kernel updates available"

# Update kernel if needed
read -p "Update kernel? (y/N): " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\nUpdating kernel..."
    apt-get install -y linux-image-generic linux-headers-generic
    echo "Kernel updated. Reboot may be required."
fi

# Restart services
echo -e "\nRestarting services..."
docker-compose restart

# Verify system health
echo -e "\nVerifying system health..."
uptime
free -h
df -h /

echo "=== System Package Update Complete ==="
```

---

## Backup and Recovery

### Automated Backup System

```bash
#!/bin/bash
# automated_backup.sh

# Configuration
BACKUP_ROOT="/backups"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_ROOT/$TIMESTAMP

echo "=== Automated Backup - $TIMESTAMP ==="

# Database backup
echo "Creating database backup..."
docker exec scientific_computing_postgres_1 pg_dump -U sci_user -d scientific_computing | gzip > $BACKUP_ROOT/$TIMESTAMP/database.sql.gz

# Application data backup
echo "Creating application data backup..."
tar -czf $BACKUP_ROOT/$TIMESTAMP/app_data.tar.gz -C /app data results logs config

# Configuration backup
echo "Creating configuration backup..."
tar -czf $BACKUP_ROOT/$TIMESTAMP/config.tar.gz /etc/scientific-computing /app/docker-compose.yml /app/.env

# System configuration backup
echo "Creating system configuration backup..."
tar -czf $BACKUP_ROOT/$TIMESTAMP/system.tar.gz /etc/nginx /etc/systemd/system/scientific-computing*

# Encrypt backups
echo "Encrypting backups..."
gpg --encrypt --recipient backup@scientific-computing.local $BACKUP_ROOT/$TIMESTAMP/*.tar.gz
gpg --encrypt --recipient backup@scientific-computing.local $BACKUP_ROOT/$TIMESTAMP/*.sql.gz

# Upload to cloud storage
echo "Uploading to cloud storage..."
aws s3 cp $BACKUP_ROOT/$TIMESTAMP/ s3://scientific-computing-backups/$TIMESTAMP/ --recursive

# Verify backup integrity
echo "Verifying backup integrity..."
for file in $BACKUP_ROOT/$TIMESTAMP/*.gz.gpg; do
    if gpg --decrypt $file | head -1 > /dev/null; then
        echo "✓ $file verified"
    else
        echo "✗ $file verification failed"
    fi
done

# Clean up old backups
echo "Cleaning up old backups..."
find $BACKUP_ROOT -name "*.gz.gpg" -mtime +$RETENTION_DAYS -exec rm {} \;

# Send notification
echo "Sending backup notification..."
curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"Backup completed successfully: $TIMESTAMP\"}" \
    https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

echo "=== Backup Complete ==="
```

### Recovery Procedures

#### 1. Complete System Recovery

```bash
#!/bin/bash
# complete_system_recovery.sh

echo "=== Complete System Recovery ==="

# Stop all services
echo "Stopping all services..."
docker-compose down

# Select backup to restore
echo "Available backups:"
ls -la /backups/ | grep "^d" | tail -10
read -p "Enter backup timestamp to restore: " BACKUP_TIMESTAMP

BACKUP_DIR="/backups/$BACKUP_TIMESTAMP"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Backup directory not found!"
    exit 1
fi

# Decrypt and restore database
echo "Restoring database..."
gpg --decrypt $BACKUP_DIR/database.sql.gz.gpg | docker exec -i scientific_computing_postgres_1 psql -U sci_user scientific_computing

# Restore application data
echo "Restoring application data..."
gpg --decrypt $BACKUP_DIR/app_data.tar.gz.gpg | tar -xzf - -C /app

# Restore configuration
echo "Restoring configuration..."
gpg --decrypt $BACKUP_DIR/config.tar.gz.gpg | tar -xzf - -C /

# Restore system configuration
echo "Restoring system configuration..."
gpg --decrypt $BACKUP_DIR/system.tar.gz.gpg | tar -xzf - -C /

# Start services
echo "Starting services..."
docker-compose up -d

# Verify recovery
echo "Verifying recovery..."
sleep 30
curl -f http://localhost:8000/health || echo "Health check failed!"

echo "=== Recovery Complete ==="
```

#### 2. Partial Data Recovery

```bash
#!/bin/bash
# partial_data_recovery.sh

echo "=== Partial Data Recovery ==="

# Select data to recover
echo "Select recovery option:"
echo "1. Database only"
echo "2. Application data only"
echo "3. Configuration only"
echo "4. Specific files"
read -p "Enter choice (1-4): " RECOVERY_CHOICE

# Select backup
echo "Available backups:"
ls -la /backups/ | grep "^d" | tail -5
read -p "Enter backup timestamp: " BACKUP_TIMESTAMP

BACKUP_DIR="/backups/$BACKUP_TIMESTAMP"

case $RECOVERY_CHOICE in
    1)
        echo "Recovering database..."
        gpg --decrypt $BACKUP_DIR/database.sql.gz.gpg | docker exec -i scientific_computing_postgres_1 psql -U sci_user scientific_computing
        ;;
    2)
        echo "Recovering application data..."
        gpg --decrypt $BACKUP_DIR/app_data.tar.gz.gpg | tar -xzf - -C /app
        ;;
    3)
        echo "Recovering configuration..."
        gpg --decrypt $BACKUP_DIR/config.tar.gz.gpg | tar -xzf - -C /
        ;;
    4)
        echo "Listing available files..."
        gpg --decrypt $BACKUP_DIR/app_data.tar.gz.gpg | tar -tzf - | less
        read -p "Enter file path to recover: " FILE_PATH
        gpg --decrypt $BACKUP_DIR/app_data.tar.gz.gpg | tar -xzf - -C /tmp $FILE_PATH
        cp /tmp/$FILE_PATH $FILE_PATH
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo "=== Partial Recovery Complete ==="
```

---

## Performance Monitoring

### Real-time Performance Monitoring

```python
# performance_monitor.py
import time
import psutil
import threading
from collections import deque
import json
from datetime import datetime, timezone

class PerformanceMonitor:
    """Real-time performance monitoring system."""

    def __init__(self, window_size=3600):  # 1 hour window
        self.window_size = window_size
        self.metrics = {
            'cpu': deque(maxlen=window_size),
            'memory': deque(maxlen=window_size),
            'disk': deque(maxlen=window_size),
            'network': deque(maxlen=window_size)
        }
        self.is_monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start performance monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            self.collect_metrics()
            time.sleep(1)  # Collect every second

    def collect_metrics(self):
        """Collect current system metrics."""
        timestamp = datetime.now(timezone.utc).isoformat() + "Z"

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.metrics['cpu'].append({
            'timestamp': timestamp,
            'value': cpu_percent
        })

        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics['memory'].append({
            'timestamp': timestamp,
            'value': memory.percent,
            'used': memory.used,
            'available': memory.available
        })

        # Disk usage
        disk = psutil.disk_usage('/')
        self.metrics['disk'].append({
            'timestamp': timestamp,
            'value': disk.percent,
            'used': disk.used,
            'free': disk.free
        })

        # Network I/O
        network = psutil.net_io_counters()
        self.metrics['network'].append({
            'timestamp': timestamp,
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        })

    def get_current_metrics(self):
        """Get current system metrics."""
        return {
            'cpu': psutil.cpu_percent(interval=None),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent,
            'network': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv
            }
        }

    def get_historical_metrics(self, metric_type, hours=1):
        """Get historical metrics for specified time period."""
        if metric_type not in self.metrics:
            return []

        cutoff_seconds = hours * 3600
        current_time = time.time()

        historical_data = []
        for metric in list(self.metrics[metric_type]):
            metric_time = datetime.fromisoformat(metric['timestamp'][:-1]).timestamp()
            if current_time - metric_time <= cutoff_seconds:
                historical_data.append(metric)

        return historical_data

    def get_performance_summary(self, hours=1):
        """Generate performance summary."""
        summary = {}

        for metric_type in self.metrics.keys():
            historical = self.get_historical_metrics(metric_type, hours)

            if historical:
                values = [item['value'] for item in historical]
                summary[metric_type] = {
                    'current': values[-1],
                    'average': sum(values) / len(values),
                    'maximum': max(values),
                    'minimum': min(values),
                    'data_points': len(values)
                }

        return summary

    def export_metrics(self, filepath):
        """Export metrics to file."""
        export_data = {
            'export_timestamp': datetime.now(timezone.utc).isoformat() + "Z",
            'metrics': dict(self.metrics)
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

    def detect_anomalies(self, metric_type, threshold_sigma=3):
        """Detect performance anomalies."""
        historical = self.get_historical_metrics(metric_type, hours=1)

        if len(historical) < 10:
            return []

        values = [item['value'] for item in historical]
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

        anomalies = []
        for i, value in enumerate(values):
            if abs(value - mean) > threshold_sigma * std:
                anomalies.append({
                    'index': i,
                    'timestamp': historical[i]['timestamp'],
                    'value': value,
                    'deviation': abs(value - mean) / std,
                    'direction': 'high' if value > mean else 'low'
                })

        return anomalies
```

### Performance Alerting

```python
# performance_alerts.py
from performance_monitor import PerformanceMonitor
from alert_manager import AlertManager
import time

class PerformanceAlertSystem:
    """Performance-based alerting system."""

    def __init__(self, monitor: PerformanceMonitor, alert_manager: AlertManager):
        self.monitor = monitor
        self.alert_manager = alert_manager
        self.thresholds = {
            'cpu': {'warning': 70, 'critical': 90},
            'memory': {'warning': 80, 'critical': 95},
            'disk': {'warning': 85, 'critical': 95}
        }
        self.alert_cooldown = 300  # 5 minutes cooldown between similar alerts

    def check_performance_alerts(self):
        """Check for performance-related alerts."""
        current_metrics = self.monitor.get_current_metrics()

        for metric_type, value in current_metrics.items():
            if metric_type in self.thresholds:
                thresholds = self.thresholds[metric_type]

                if value >= thresholds['critical']:
                    self.alert_manager.create_alert(
                        severity="critical",
                        component="system",
                        message=f"Critical {metric_type} usage: {value:.1f}% (threshold: {thresholds['critical']}%)",
                        metric_value=value,
                        threshold=thresholds['critical']
                    )

                elif value >= thresholds['warning']:
                    self.alert_manager.create_alert(
                        severity="warning",
                        component="system",
                        message=f"High {metric_type} usage: {value:.1f}% (threshold: {thresholds['warning']}%)",
                        metric_value=value,
                        threshold=thresholds['warning']
                    )

    def monitor_performance_trends(self):
        """Monitor performance trends and predict issues."""
        summary = self.monitor.get_performance_summary(hours=1)

        for metric_type, data in summary.items():
            # Check for concerning trends
            if data['maximum'] - data['minimum'] > 20:  # High variability
                self.alert_manager.create_alert(
                    severity="warning",
                    component="system",
                    message=f"High {metric_type} variability detected: {data['maximum']:.1f}% - {data['minimum']:.1f}%",
                    metric_value=data['maximum'] - data['minimum']
                )

            # Check for sustained high usage
            if data['average'] > self.thresholds.get(metric_type, {}).get('warning', 70):
                self.alert_manager.create_alert(
                    severity="warning",
                    component="system",
                    message=f"Sustained high {metric_type} usage: {data['average']:.1f}% average",
                    metric_value=data['average']
                )

    def start_monitoring(self):
        """Start performance monitoring and alerting."""
        while True:
            self.check_performance_alerts()
            self.monitor_performance_trends()
            time.sleep(60)  # Check every minute
```

---

## Security Maintenance

### Security Update Procedures

```bash
#!/bin/bash
# security_updates.sh

echo "=== Security Maintenance - $(date) ==="

# Update system packages
echo "Updating system security packages..."
apt-get update
apt-get upgrade -y --security

# Update container images
echo "Updating container security patches..."
docker-compose pull

# Update Python packages with security fixes
echo "Checking for Python security updates..."
pip list --outdated | grep -i security || echo "No security updates needed"

# Update SSL/TLS certificates
echo "Checking SSL certificate expiry..."
certbot certificates | grep -A 5 "Certificate Name" | while read line; do
    if echo "$line" | grep -q "Expiry Date"; then
        expiry_date=$(echo "$line" | sed 's/.*Expiry Date: //')
        echo "Certificate expires: $expiry_date"
    fi
done

# Renew certificates if needed
echo "Renewing SSL certificates..."
certbot renew

# Update firewall rules
echo "Updating firewall rules..."
ufw --force enable
ufw status verbose

# Security audit
echo "Running security audit..."
echo "Open ports:"
netstat -tlnp | grep LISTEN

echo "Failed login attempts (last 24h):"
grep "Failed password" /var/log/auth.log | wc -l

echo "Suspicious activities:"
grep -i "attack\|exploit\|hack" /var/log/auth.log /var/log/syslog | wc -l

# Rotate logs
echo "Rotating security logs..."
logrotate -f /etc/logrotate.d/scientific-computing

echo "=== Security Maintenance Complete ==="
```

### Access Control Maintenance

```bash
#!/bin/bash
# access_control_maintenance.sh

echo "=== Access Control Maintenance ==="

# Review user accounts
echo "Reviewing user accounts..."
cat /etc/passwd | grep -E "home|app" | while read line; do
    username=$(echo $line | cut -d: -f1)
    uid=$(echo $line | cut -d: -f3)

    # Check for accounts with UID 0 (root)
    if [ "$uid" = "0" ] && [ "$username" != "root" ]; then
        echo "WARNING: Non-root user with UID 0: $username"
    fi

    # Check password expiry
    passwd -S $username | while read user status; do
        if echo "$status" | grep -q "Password expires"; then
            echo "Password expiry for $user: $status"
        fi
    done
done

# Check sudo access
echo -e "\nChecking sudo access..."
grep -v "^#" /etc/sudoers | grep -v "^$" | while read line; do
    echo "Sudo rule: $line"
done

# Review SSH access
echo -e "\nReviewing SSH access..."
if [ -f /etc/ssh/sshd_config ]; then
    echo "SSH PermitRootLogin: $(grep 'PermitRootLogin' /etc/ssh/sshd_config | tail -1)"
    echo "SSH PasswordAuthentication: $(grep 'PasswordAuthentication' /etc/ssh/sshd_config | tail -1)"
    echo "SSH X11Forwarding: $(grep 'X11Forwarding' /etc/ssh/sshd_config | tail -1)"
fi

# Check file permissions
echo -e "\nChecking critical file permissions..."
critical_files=(
    "/etc/passwd"
    "/etc/shadow"
    "/etc/sudoers"
    "/etc/ssh/sshd_config"
    "/app/config"
)

for file in "${critical_files[@]}"; do
    if [ -f "$file" ]; then
        permissions=$(stat -c "%a" "$file")
        owner=$(stat -c "%U" "$file")
        echo "File: $file, Permissions: $permissions, Owner: $owner"
    fi
done

# Review recent login activity
echo -e "\nReviewing recent login activity..."
last | head -20

echo "=== Access Control Maintenance Complete ==="
```

---

## Troubleshooting Procedures

### Common Issues and Solutions

#### 1. Service Startup Issues

```bash
#!/bin/bash
# troubleshoot_service_startup.sh

echo "=== Service Startup Troubleshooting ==="

# Check Docker status
echo "Checking Docker status..."
docker ps -a

# Check container logs
echo -e "\nChecking container logs..."
docker-compose logs --tail=50

# Check resource availability
echo -e "\nChecking resource availability..."
echo "Memory:"
free -h
echo -e "\nDisk space:"
df -h
echo -e "\nCPU load:"
uptime

# Check network connectivity
echo -e "\nChecking network connectivity..."
ping -c 3 8.8.8.8
curl -I http://localhost:8000/health

# Restart services
echo -e "\nAttempting service restart..."
docker-compose restart

# Verify restart
sleep 10
docker ps

echo "=== Service Startup Troubleshooting Complete ==="
```

#### 2. Performance Issues

```bash
#!/bin/bash
# troubleshoot_performance.sh

echo "=== Performance Troubleshooting ==="

# Check system load
echo "System load:"
uptime
cat /proc/loadavg

# Check memory usage
echo -e "\nMemory usage:"
free -h
ps aux --sort=-%mem | head -10

# Check CPU usage
echo -e "\nCPU usage:"
top -bn1 | head -20

# Check disk I/O
echo -e "\nDisk I/O:"
iostat -x 1 5

# Check network I/O
echo -e "\nNetwork I/O:"
iftop -t -s 5 2>/dev/null || echo "iftop not available"

# Check application performance
echo -e "\nApplication performance:"
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Database performance
echo -e "\nDatabase performance:"
docker exec scientific_computing_postgres_1 psql -U sci_user -d scientific_computing -c "
SELECT * FROM pg_stat_activity WHERE state = 'active';
"

echo "=== Performance Troubleshooting Complete ==="
```

#### 3. Data Corruption Issues

```bash
#!/bin/bash
# troubleshoot_data_corruption.sh

echo "=== Data Corruption Troubleshooting ==="

# Check file system integrity
echo "Checking file system integrity..."
fsck -n /

# Check database integrity
echo "Checking database integrity..."
docker exec scientific_computing_postgres_1 psql -U sci_user -d scientific_computing -c "
SELECT schemaname, tablename, n_dead_tup, last_vacuum, last_autovacuum
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;
"

# Validate data files
echo "Validating data files..."
find /app/data -name "*.json" -exec python3 -c "
import json
import sys
try:
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    print(f'✓ {sys.argv[1]}: {len(data)} records')
except Exception as e:
    print(f'✗ {sys.argv[1]}: {e}')
" {} \;

# Check backup integrity
echo "Checking backup integrity..."
LATEST_BACKUP=$(ls -t /backups/*.sql.gz | head -1)
if [ -f "$LATEST_BACKUP" ]; then
    echo "Testing backup: $LATEST_BACKUP"
    gunzip -c $LATEST_BACKUP | head -10
fi

echo "=== Data Corruption Troubleshooting Complete ==="
```

---

## Emergency Response

### Critical Incident Response Plan

```bash
#!/bin/bash
# emergency_response.sh

echo "=== EMERGENCY RESPONSE ACTIVATED ==="
echo "Timestamp: $(date)"
echo "==========================================="

# Step 1: Assess the situation
echo "Step 1: Situation Assessment"
echo "============================"

# Check system status
echo "System status:"
uptime
echo -e "\nService status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check critical metrics
echo -e "\nCritical metrics:"
echo "Memory usage: $(free | grep Mem | awk '{printf \"%.1f%%\", $3/$2 * 100.0}')"
echo "Disk usage: $(df / | tail -1 | awk '{print $5}')"
echo "Load average: $(uptime | sed 's/.*load average: //' | awk '{print $1}')"

# Step 2: Isolate the problem
echo -e "\nStep 2: Problem Isolation"
echo "========================="

# Stop non-critical services
echo "Stopping non-critical services..."
# docker-compose stop background_worker
# docker-compose stop monitoring

# Check recent logs for errors
echo "Checking recent error logs..."
grep -i error /var/log/scientific-computing/*.log | tail -20

# Step 3: Implement immediate fixes
echo -e "\nStep 3: Immediate Fixes"
echo "======================="

# Restart critical services
echo "Restarting critical services..."
docker-compose restart web postgres

# Clear temporary issues
echo "Clearing temporary files..."
rm -rf /tmp/scientific_computing_*

# Step 4: Restore from backup if needed
echo -e "\nStep 4: Backup Restoration Check"
echo "================================="

read -p "Does the system need restoration from backup? (y/N): " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Initiating backup restoration..."
    ./emergency_rollback.sh
fi

# Step 5: Communication
echo -e "\nStep 5: Communication"
echo "====================="

# Notify team
echo "Sending emergency notification..."
curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"🚨 EMERGENCY: System incident detected and response initiated\"}" \
    https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# Step 6: Recovery verification
echo -e "\nStep 6: Recovery Verification"
echo "============================="

# Wait for services to stabilize
sleep 30

# Verify system health
echo "Verifying system recovery..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ System health check passed"
else
    echo "❌ System health check failed"
fi

# Step 7: Post-incident review
echo -e "\nStep 7: Post-Incident Review"
echo "============================"

# Log incident details
echo "Logging incident details..."
cat >> /var/log/scientific-computing/incidents.log << EOF
$(date): Emergency response activated
System status: $(docker ps | wc -l) containers running
Memory usage: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')
Resolution: $(if curl -f http://localhost:8000/health > /dev/null 2>&1; then echo "Successful"; else echo "Failed"; fi)
EOF

echo "=== EMERGENCY RESPONSE COMPLETE ==="
echo "Please conduct a thorough post-mortem analysis."
```

### Incident Response Checklist

- [ ] **Immediate Actions**
  - [ ] Stop affected services
  - [ ] Isolate compromised systems
  - [ ] Notify security team
  - [ ] Preserve evidence

- [ ] **Assessment**
  - [ ] Determine incident scope
  - [ ] Identify root cause
  - [ ] Assess data exposure
  - [ ] Evaluate business impact

- [ ] **Containment**
  - [ ] Block malicious activity
  - [ ] Remove malware/backdoors
  - [ ] Change compromised credentials
  - [ ] Implement temporary fixes

- [ ] **Recovery**
  - [ ] Restore from clean backups
  - [ ] Verify system integrity
  - [ ] Test critical functions
  - [ ] Monitor for anomalies

- [ ] **Lessons Learned**
  - [ ] Document incident timeline
  - [ ] Identify prevention measures
  - [ ] Update response procedures
  - [ ] Conduct team debriefing

---

## Documentation Updates

### Documentation Maintenance Procedure

```bash
#!/bin/bash
# update_documentation.sh

echo "=== Documentation Update Procedure ==="

# Step 1: Check for outdated documentation
echo "Checking for outdated documentation..."

# Check file modification dates
find docs/ -name "*.md" -mtime +90 -exec ls -la {} \; | while read line; do
    echo "Potentially outdated: $line"
done

# Step 2: Update API documentation
echo -e "\nUpdating API documentation..."
python3 scripts/generate_api_docs.py

# Step 3: Update configuration documentation
echo -e "\nUpdating configuration documentation..."
python3 scripts/generate_config_docs.py

# Step 4: Update deployment documentation
echo -e "\nUpdating deployment documentation..."
python3 scripts/generate_deployment_docs.py

# Step 5: Review and update procedures
echo -e "\nReviewing maintenance procedures..."
echo "Please review the following procedures:"
echo "- Backup procedures (last updated: $(stat -c %y docs/backup_procedures.md))"
echo "- Security procedures (last updated: $(stat -c %y docs/security_procedures.md))"
echo "- Recovery procedures (last updated: $(stat -c %y docs/recovery_procedures.md))"

# Step 6: Update version information
echo -e "\nUpdating version information..."
CURRENT_VERSION=$(git describe --tags --abbrev=0)
sed -i "s/Version: [0-9]\+\.[0-9]\+\.[0-9]\+/Version: $CURRENT_VERSION/" docs/*.md

# Step 7: Validate documentation
echo -e "\nValidating documentation..."
find docs/ -name "*.md" -exec markdown-link-check {} \; 2>/dev/null || echo "Link checker not available"

# Step 8: Commit documentation updates
echo -e "\nCommitting documentation updates..."
git add docs/
git commit -m "docs: Update documentation for version $CURRENT_VERSION

- Updated API documentation
- Refreshed configuration guides
- Reviewed maintenance procedures
- Updated version information"

echo "=== Documentation Update Complete ==="
```

---

## Compliance and Auditing

### Regulatory Compliance Maintenance

```bash
#!/bin/bash
# compliance_audit.sh

echo "=== Compliance Audit Procedure ==="

# Audit configuration
AUDIT_DATE=$(date +%Y%m%d)
AUDIT_LOG="/var/log/scientific-computing/compliance_audit_$AUDIT_DATE.log"

echo "Compliance Audit - $AUDIT_DATE" > $AUDIT_LOG
echo "=================================" >> $AUDIT_LOG

# 1. Data Security Compliance
echo -e "\n1. Data Security Compliance" >> $AUDIT_LOG
echo "===========================" >> $AUDIT_LOG

# Check encryption
echo "Encryption status:" >> $AUDIT_LOG
ls -la /app/config/ | grep -E "\.(key|pem|cert)" || echo "No encryption keys found" >> $AUDIT_LOG

# Check access controls
echo -e "\nAccess controls:" >> $AUDIT_LOG
ls -la /app/data/ | head -5 >> $AUDIT_LOG

# 2. Backup Compliance
echo -e "\n2. Backup Compliance" >> $AUDIT_LOG
echo "===================" >> $AUDIT_LOG

# Check backup frequency
echo "Recent backups:" >> $AUDIT_LOG
find /backups -name "*.sql.gz" -mtime -7 | wc -l | xargs echo "Database backups (last 7 days):" >> $AUDIT_LOG
find /backups -name "*.tar.gz" -mtime -7 | wc -l | xargs echo "File backups (last 7 days):" >> $AUDIT_LOG

# Check backup integrity
echo -e "\nBackup integrity:" >> $AUDIT_LOG
LATEST_BACKUP=$(find /backups -name "*.sql.gz" -mtime -1 | head -1)
if [ -f "$LATEST_BACKUP" ]; then
    gunzip -c $LATEST_BACKUP | wc -l | xargs echo "Latest backup lines:" >> $AUDIT_LOG
else
    echo "No recent backup found!" >> $AUDIT_LOG
fi

# 3. System Access Compliance
echo -e "\n3. System Access Compliance" >> $AUDIT_LOG
echo "===========================" >> $AUDIT_LOG

# Check user accounts
echo "User accounts:" >> $AUDIT_LOG
grep -c "/home" /etc/passwd | xargs echo "Total user accounts:" >> $AUDIT_LOG

# Check sudo access
echo "Sudo users:" >> $AUDIT_LOG
grep -c "^[^#]" /etc/sudoers | xargs echo "Sudo rules:" >> $AUDIT_LOG

# 4. Logging Compliance
echo -e "\n4. Logging Compliance" >> $AUDIT_LOG
echo "====================" >> $AUDIT_LOG

# Check log retention
echo "Log files:" >> $AUDIT_LOG
find /var/log/scientific-computing -name "*.log" -mtime -90 | wc -l | xargs echo "Logs (last 90 days):" >> $AUDIT_LOG

# Check log integrity
echo "Log integrity:" >> $AUDIT_LOG
for log_file in /var/log/scientific-computing/*.log; do
    if [ -f "$log_file" ]; then
        wc -l < "$log_file" | xargs echo "$log_file lines:" >> $AUDIT_LOG
    fi
done

# 5. Network Security Compliance
echo -e "\n5. Network Security Compliance" >> $AUDIT_LOG
echo "=============================" >> $AUDIT_LOG

# Check firewall rules
echo "Firewall status:" >> $AUDIT_LOG
ufw status | grep -E "(Status|ports)" >> $AUDIT_LOG

# Check SSL certificates
echo "SSL certificates:" >> $AUDIT_LOG
certbot certificates 2>/dev/null | grep -c "Certificate Name" | xargs echo "Valid certificates:" >> $AUDIT_LOG

# 6. Audit Summary
echo -e "\n6. Audit Summary" >> $AUDIT_LOG
echo "===============" >> $AUDIT_LOG

# Generate compliance score
COMPLIANCE_SCORE=100

# Deduct points for issues
if [ $(find /backups -name "*.sql.gz" -mtime -7 | wc -l) -lt 7 ]; then
    COMPLIANCE_SCORE=$((COMPLIANCE_SCORE - 20))
    echo "ISSUE: Insufficient database backups" >> $AUDIT_LOG
fi

if [ $(find /var/log/scientific-computing -name "*.log" -mtime -90 | wc -l) -lt 5 ]; then
    COMPLIANCE_SCORE=$((COMPLIANCE_SCORE - 15))
    echo "ISSUE: Insufficient log retention" >> $AUDIT_LOG
fi

echo "Overall Compliance Score: $COMPLIANCE_SCORE%" >> $AUDIT_LOG

# Send audit report
echo "Sending audit report..."
mail -s "Compliance Audit Report - $AUDIT_DATE" compliance@scientific-computing.local < $AUDIT_LOG

echo "=== Compliance Audit Complete ==="
echo "Audit log saved to: $AUDIT_LOG"
```

---

## Maintenance Schedule

### Daily Tasks
- [ ] System health checks
- [ ] Log rotation
- [ ] Backup verification
- [ ] Security scans

### Weekly Tasks
- [ ] Performance optimization
- [ ] Security updates
- [ ] Data integrity checks
- [ ] User access reviews

### Monthly Tasks
- [ ] Comprehensive system audit
- [ ] Backup restoration testing
- [ ] Documentation updates
- [ ] Compliance reviews

### Quarterly Tasks
- [ ] Major software updates
- [ ] Security assessments
- [ ] Disaster recovery testing
- [ ] Performance benchmarking

### Annual Tasks
- [ ] Complete system overhaul
- [ ] Compliance certifications
- [ ] Business continuity planning
- [ ] Vendor assessments

---

**This maintenance guide ensures the long-term reliability, security, and performance of the Scientific Computing Toolkit deployment. Regular adherence to these procedures will minimize downtime and maintain system integrity.**

*Scientific Computing Toolkit - Operations Team*
