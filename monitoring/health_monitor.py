#!/usr/bin/env python3
"""
Health Monitoring and Alerting System for Scientific Computing Toolkit

This module provides comprehensive monitoring and alerting capabilities for:
1. Framework performance metrics
2. System resource usage
3. Error detection and reporting
4. Predictive maintenance alerts
5. Real-time dashboard integration

Author: Scientific Computing Toolkit Team
Date: 2025
License: GPL-3.0-only
"""

import psutil
import time
import threading
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import asyncio
from dataclasses import dataclass, asdict
import numpy as np
from collections import deque
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HealthMetric:
    """Container for health monitoring metrics."""
    timestamp: str
    component: str
    metric_name: str
    value: float
    unit: str
    status: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    @property
    def is_warning(self) -> bool:
        """Check if metric is in warning state."""
        if self.threshold_warning is not None:
            return self.value >= self.threshold_warning
        return False

    @property
    def is_critical(self) -> bool:
        """Check if metric is in critical state."""
        if self.threshold_critical is not None:
            return self.value >= self.threshold_critical
        return False


@dataclass
class Alert:
    """Container for system alerts."""
    alert_id: str
    timestamp: str
    severity: str
    component: str
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    resolved: bool = False
    resolved_timestamp: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"
        if not self.alert_id:
            self.alert_id = f"alert_{int(time.time()*1000)}_{hash(self.message) % 10000}"


class MetricsCollector:
    """Collect system and application metrics."""

    def __init__(self, history_size: int = 1000):
        self.process = psutil.Process()
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.start_time = time.time()

    def collect_system_metrics(self) -> List[HealthMetric]:
        """Collect comprehensive system metrics."""
        metrics = []

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.append(HealthMetric(
            component="system",
            metric_name="cpu_usage",
            value=cpu_percent,
            unit="%",
            status="normal" if cpu_percent < 80 else "warning" if cpu_percent < 95 else "critical",
            threshold_warning=80.0,
            threshold_critical=95.0
        ))

        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(HealthMetric(
            component="system",
            metric_name="memory_usage",
            value=memory.percent,
            unit="%",
            status="normal" if memory.percent < 80 else "warning" if memory.percent < 95 else "critical",
            threshold_warning=80.0,
            threshold_critical=95.0,
            metadata={"total_gb": memory.total / (1024**3), "available_gb": memory.available / (1024**3)}
        ))

        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.append(HealthMetric(
            component="system",
            metric_name="disk_usage",
            value=disk.percent,
            unit="%",
            status="normal" if disk.percent < 85 else "warning" if disk.percent < 95 else "critical",
            threshold_warning=85.0,
            threshold_critical=95.0,
            metadata={"total_gb": disk.total / (1024**3), "free_gb": disk.free / (1024**3)}
        ))

        # Network metrics
        network = psutil.net_io_counters()
        metrics.append(HealthMetric(
            component="system",
            metric_name="network_bytes_sent",
            value=network.bytes_sent / (1024**2),  # MB
            unit="MB",
            status="normal",
            metadata={"bytes_recv": network.bytes_recv / (1024**2)}
        ))

        return metrics

    def collect_application_metrics(self) -> List[HealthMetric]:
        """Collect application-specific metrics."""
        metrics = []

        try:
            # Process memory
            process_memory = self.process.memory_info().rss / (1024**2)  # MB
            metrics.append(HealthMetric(
                component="application",
                metric_name="process_memory",
                value=process_memory,
                unit="MB",
                status="normal" if process_memory < 500 else "warning" if process_memory < 1000 else "critical",
                threshold_warning=500.0,
                threshold_critical=1000.0
            ))

            # Process CPU
            process_cpu = self.process.cpu_percent(interval=0.1)
            metrics.append(HealthMetric(
                component="application",
                metric_name="process_cpu",
                value=process_cpu,
                unit="%",
                status="normal" if process_cpu < 70 else "warning" if process_cpu < 90 else "critical",
                threshold_warning=70.0,
                threshold_critical=90.0
            ))

            # Uptime
            uptime = time.time() - self.start_time
            metrics.append(HealthMetric(
                component="application",
                metric_name="uptime",
                value=uptime / 3600,  # hours
                unit="hours",
                status="normal"
            ))

            # Thread count
            thread_count = self.process.num_threads()
            metrics.append(HealthMetric(
                component="application",
                metric_name="thread_count",
                value=thread_count,
                unit="count",
                status="normal" if thread_count < 50 else "warning" if thread_count < 100 else "critical",
                threshold_warning=50,
                threshold_critical=100
            ))

        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")

        return metrics

    def collect_framework_metrics(self) -> List[HealthMetric]:
        """Collect framework-specific metrics."""
        metrics = []

        # Mock framework metrics (would be replaced with actual framework monitoring)
        metrics.extend([
            HealthMetric(
                component="hybrid_uq",
                metric_name="prediction_latency",
                value=0.023,  # seconds
                unit="s",
                status="normal" if 0.023 < 0.1 else "warning",
                threshold_warning=0.1,
                threshold_critical=0.5
            ),
            HealthMetric(
                component="data_pipeline",
                metric_name="processing_throughput",
                value=1500,  # items/second
                unit="items/s",
                status="normal" if 1500 > 1000 else "warning",
                threshold_warning=1000,
                threshold_critical=500
            ),
            HealthMetric(
                component="cross_framework_comm",
                metric_name="message_queue_size",
                value=5,
                unit="messages",
                status="normal" if 5 < 100 else "warning" if 5 < 500 else "critical",
                threshold_warning=100,
                threshold_critical=500
            ),
            HealthMetric(
                component="security",
                metric_name="failed_auth_attempts",
                value=2,
                unit="attempts/hour",
                status="normal" if 2 < 10 else "warning" if 2 < 50 else "critical",
                threshold_warning=10,
                threshold_critical=50
            )
        ])

        return metrics

    def collect_all_metrics(self) -> List[HealthMetric]:
        """Collect all available metrics."""
        all_metrics = []
        all_metrics.extend(self.collect_system_metrics())
        all_metrics.extend(self.collect_application_metrics())
        all_metrics.extend(self.collect_framework_metrics())

        # Store in history
        for metric in all_metrics:
            self.metrics_history.append(asdict(metric))

        return all_metrics

    def get_metric_history(self, component: str = None, metric_name: str = None,
                          hours: int = 1) -> List[Dict[str, Any]]:
        """Get historical metrics data."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        history = []

        for metric_data in self.metrics_history:
            metric_time = datetime.fromisoformat(metric_data['timestamp'][:-1])

            if metric_time >= cutoff_time:
                if component and metric_data['component'] != component:
                    continue
                if metric_name and metric_data['metric_name'] != metric_name:
                    continue
                history.append(metric_data)

        return history

    def calculate_metric_statistics(self, component: str, metric_name: str,
                                  hours: int = 1) -> Dict[str, float]:
        """Calculate statistical summary of metrics."""
        history = self.get_metric_history(component, metric_name, hours)

        if not history:
            return {}

        values = [item['value'] for item in history]

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'count': len(values)
        }


class AlertManager:
    """Manage system alerts and notifications."""

    def __init__(self, config_path: str = "alert_config.json"):
        self.config_path = Path(config_path)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[str, Callable] = {}

        self.load_alert_configuration()

    def load_alert_configuration(self):
        """Load alert configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.load_notification_channels(config)
            except Exception as e:
                logger.error(f"Failed to load alert config: {e}")

        # Default email configuration
        default_config = {
            "email": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "alerts@scientific-toolkit.com",
                "recipient_emails": ["admin@scientific-toolkit.com"],
                "use_tls": True
            },
            "slack": {
                "webhook_url": "https://hooks.slack.com/services/...",
                "channel": "#alerts"
            },
            "pagerduty": {
                "integration_key": "your-integration-key"
            }
        }

        if not self.config_path.exists():
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)

    def load_notification_channels(self, config: Dict[str, Any]):
        """Load notification channel configurations."""
        if 'email' in config:
            email_config = config['email']
            self.notification_channels['email'] = lambda alert: self.send_email_alert(alert, email_config)

        if 'slack' in config:
            slack_config = config['slack']
            self.notification_channels['slack'] = lambda alert: self.send_slack_alert(alert, slack_config)

        if 'pagerduty' in config:
            pd_config = config['pagerduty']
            self.notification_channels['pagerduty'] = lambda alert: self.send_pagerduty_alert(alert, pd_config)

    def create_alert(self, severity: str, component: str, message: str,
                    metric_value: float = None, threshold: float = None) -> Alert:
        """Create a new alert."""
        alert = Alert(
            alert_id="",
            timestamp="",
            severity=severity,
            component=component,
            message=message,
            metric_value=metric_value,
            threshold=threshold
        )

        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        logger.warning(f"Alert created: {alert.alert_id} - {alert.message}")

        # Send notifications
        self.send_notifications(alert)

        return alert

    def resolve_alert(self, alert_id: str, resolution_message: str = "") -> bool:
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_timestamp = datetime.now(timezone.utc).isoformat() + "Z"

        # Update message with resolution
        if resolution_message:
            alert.message += f" [RESOLVED: {resolution_message}]"

        # Send resolution notifications
        self.send_notifications(alert, resolution=True)

        del self.active_alerts[alert_id]
        logger.info(f"Alert resolved: {alert_id}")

        return True

    def check_metric_alerts(self, metrics: List[HealthMetric]):
        """Check metrics for alert conditions."""
        for metric in metrics:
            alert_key = f"{metric.component}_{metric.metric_name}"

            # Check for critical alerts
            if metric.is_critical:
                if alert_key not in self.active_alerts:
                    self.create_alert(
                        severity="critical",
                        component=metric.component,
                        message=f"Critical: {metric.metric_name} = {metric.value}{metric.unit} (threshold: {metric.threshold_critical})",
                        metric_value=metric.value,
                        threshold=metric.threshold_critical
                    )

            # Check for warning alerts
            elif metric.is_warning:
                if alert_key not in self.active_alerts:
                    self.create_alert(
                        severity="warning",
                        component=metric.component,
                        message=f"Warning: {metric.metric_name} = {metric.value}{metric.unit} (threshold: {metric.threshold_warning})",
                        metric_value=metric.value,
                        threshold=metric.threshold_warning
                    )

            # Auto-resolve alerts when metrics return to normal
            elif alert_key in self.active_alerts and not metric.is_warning and not metric.is_critical:
                self.resolve_alert(alert_key, "Metric returned to normal range")

    def send_notifications(self, alert: Alert, resolution: bool = False):
        """Send alert notifications through all configured channels."""
        for channel_name, send_func in self.notification_channels.items():
            try:
                send_func(alert)
            except Exception as e:
                logger.error(f"Failed to send {channel_name} notification: {e}")

    def send_email_alert(self, alert: Alert, config: Dict[str, Any]):
        """Send email alert notification."""
        try:
            msg = MIMEMultipart()
            msg['From'] = config['sender_email']
            msg['To'] = ', '.join(config['recipient_emails'])
            msg['Subject'] = f"{'[RESOLVED] ' if alert.resolved else ''}[{alert.severity.upper()}] Scientific Toolkit Alert"

            body = f"""
Scientific Computing Toolkit Alert

Severity: {alert.severity.upper()}
Component: {alert.component}
Time: {alert.timestamp}
Message: {alert.message}

{f"Metric Value: {alert.metric_value}" if alert.metric_value else ""}
{f"Threshold: {alert.threshold}" if alert.threshold else ""}

This is an automated message from the Scientific Computing Toolkit monitoring system.
"""

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            if config.get('use_tls', True):
                server.starttls()
            # Note: In production, use proper authentication
            # server.login(username, password)
            server.sendmail(config['sender_email'], config['recipient_emails'], msg.as_string())
            server.quit()

        except Exception as e:
            logger.error(f"Email alert failed: {e}")

    def send_slack_alert(self, alert: Alert, config: Dict[str, Any]):
        """Send Slack alert notification."""
        try:
            payload = {
                "channel": config['channel'],
                "username": "Scientific Toolkit Monitor",
                "icon_emoji": ":warning:" if alert.severity == "warning" else ":alert:",
                "attachments": [{
                    "color": "danger" if alert.severity == "critical" else "warning",
                    "title": f"{'[RESOLVED] ' if alert.resolved else ''}Alert: {alert.component}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.upper(), "short": True},
                        {"title": "Time", "value": alert.timestamp, "short": True}
                    ]
                }]
            }

            response = requests.post(config['webhook_url'], json=payload)
            response.raise_for_status()

        except Exception as e:
            logger.error(f"Slack alert failed: {e}")

    def send_pagerduty_alert(self, alert: Alert, config: Dict[str, Any]):
        """Send PagerDuty alert notification."""
        try:
            payload = {
                "routing_key": config['integration_key'],
                "event_action": "resolve" if alert.resolved else "trigger",
                "payload": {
                    "summary": alert.message,
                    "severity": alert.severity,
                    "source": "scientific-computing-toolkit",
                    "component": alert.component,
                    "group": "monitoring",
                    "class": "health_check"
                }
            }

            if not alert.resolved:
                payload["dedup_key"] = alert.alert_id

            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload
            )
            response.raise_for_status()

        except Exception as e:
            logger.error(f"PagerDuty alert failed: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified time period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        return [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert.timestamp[:-1]) >= cutoff_time
        ]


class HealthDashboard:
    """Real-time health dashboard for monitoring."""

    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.dashboard_data = {}
        self.update_interval = 5  # seconds

    def update_dashboard(self):
        """Update dashboard with latest metrics and alerts."""
        # Collect latest metrics
        metrics = self.metrics_collector.collect_all_metrics()

        # Get active alerts
        active_alerts = self.alert_manager.get_active_alerts()

        # Update dashboard data
        self.dashboard_data = {
            'timestamp': datetime.now(timezone.utc).isoformat() + "Z",
            'system_health': self.calculate_system_health(metrics),
            'metrics': [asdict(metric) for metric in metrics],
            'alerts': [asdict(alert) for alert in active_alerts],
            'summary': self.generate_summary(metrics, active_alerts)
        }

    def calculate_system_health(self, metrics: List[HealthMetric]) -> str:
        """Calculate overall system health status."""
        critical_count = sum(1 for m in metrics if m.status == "critical")
        warning_count = sum(1 for m in metrics if m.status == "warning")

        if critical_count > 0:
            return "critical"
        elif warning_count > 0:
            return "warning"
        else:
            return "healthy"

    def generate_summary(self, metrics: List[HealthMetric], alerts: List[Alert]) -> Dict[str, Any]:
        """Generate dashboard summary."""
        return {
            'total_metrics': len(metrics),
            'healthy_metrics': sum(1 for m in metrics if m.status == "normal"),
            'warning_metrics': sum(1 for m in metrics if m.status == "warning"),
            'critical_metrics': sum(1 for m in metrics if m.status == "critical"),
            'active_alerts': len(alerts),
            'critical_alerts': sum(1 for a in alerts if a.severity == "critical"),
            'warning_alerts': sum(1 for a in alerts if a.severity == "warning")
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard_data

    def export_dashboard(self, filepath: str):
        """Export dashboard data to file."""
        with open(filepath, 'w') as f:
            json.dump(self.dashboard_data, f, indent=2, default=str)

    def generate_health_report(self) -> str:
        """Generate human-readable health report."""
        data = self.dashboard_data

        report = f"""
Scientific Computing Toolkit Health Report
Generated: {data.get('timestamp', 'Unknown')}

SYSTEM HEALTH: {data.get('system_health', 'Unknown').upper()}

METRICS SUMMARY:
- Total Metrics: {data.get('summary', {}).get('total_metrics', 0)}
- Healthy: {data.get('summary', {}).get('healthy_metrics', 0)}
- Warnings: {data.get('summary', {}).get('warning_metrics', 0)}
- Critical: {data.get('summary', {}).get('critical_metrics', 0)}

ACTIVE ALERTS: {data.get('summary', {}).get('active_alerts', 0)}
- Critical: {data.get('summary', {}).get('critical_alerts', 0)}
- Warnings: {data.get('summary', {}).get('warning_alerts', 0)}

RECENT METRICS:
"""

        for metric in data.get('metrics', [])[:10]:  # Show first 10 metrics
            report += f"- {metric['component']}.{metric['metric_name']}: {metric['value']:.2f}{metric['unit']} ({metric['status']})\n"

        if data.get('alerts'):
            report += "\nACTIVE ALERTS:\n"
            for alert in data.get('alerts', []):
                report += f"- [{alert['severity'].upper()}] {alert['component']}: {alert['message']}\n"

        return report


class MonitoringSystem:
    """Main monitoring system coordinating all components."""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.health_dashboard = HealthDashboard(self.metrics_collector, self.alert_manager)

        self.monitoring_active = False
        self.monitoring_thread = None
        self.dashboard_update_interval = 30  # seconds

    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            logger.warning("Monitoring system is already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("Monitoring system started")

    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        logger.info("Monitoring system stopped")

    def monitoring_loop(self):
        """Main monitoring loop."""
        last_dashboard_update = 0

        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self.metrics_collector.collect_all_metrics()

                # Check for alerts
                self.alert_manager.check_metric_alerts(metrics)

                # Update dashboard periodically
                current_time = time.time()
                if current_time - last_dashboard_update >= self.dashboard_update_interval:
                    self.health_dashboard.update_dashboard()
                    last_dashboard_update = current_time

                # Sleep before next iteration
                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)  # Wait longer on error

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        self.health_dashboard.update_dashboard()
        return self.health_dashboard.get_dashboard_data()

    def get_health_report(self) -> str:
        """Get human-readable health report."""
        self.health_dashboard.update_dashboard()
        return self.health_dashboard.generate_health_report()

    def export_health_data(self, filepath: str):
        """Export health monitoring data."""
        self.health_dashboard.export_dashboard(filepath)

    def simulate_metric_anomaly(self, component: str, metric_name: str, value: float):
        """Simulate a metric anomaly for testing (development only)."""
        # Create a fake critical metric
        metric = HealthMetric(
            component=component,
            metric_name=metric_name,
            value=value,
            unit="%",
            status="critical",
            threshold_critical=90.0
        )

        # Add to metrics for alert checking
        self.alert_manager.check_metric_alerts([metric])

        logger.info(f"Simulated anomaly: {component}.{metric_name} = {value}")

    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring system statistics."""
        return {
            'monitoring_active': self.monitoring_active,
            'metrics_collected': len(self.metrics_collector.metrics_history),
            'active_alerts': len(self.alert_manager.active_alerts),
            'alert_history_size': len(self.alert_manager.alert_history),
            'uptime': time.time() - self.metrics_collector.start_time
        }


# Global monitoring instance
monitoring_system = MonitoringSystem()


def start_monitoring():
    """Start the global monitoring system."""
    monitoring_system.start_monitoring()


def stop_monitoring():
    """Stop the global monitoring system."""
    monitoring_system.stop_monitoring()


def get_health_status():
    """Get current health status."""
    return monitoring_system.get_health_status()


def get_health_report():
    """Get human-readable health report."""
    return monitoring_system.get_health_report()


def simulate_anomaly(component: str, metric: str, value: float):
    """Simulate metric anomaly for testing."""
    monitoring_system.simulate_metric_anomaly(component, metric, value)


if __name__ == "__main__":
    # Start monitoring system
    start_monitoring()

    try:
        # Run for a few minutes to collect data
        print("Monitoring system active... Press Ctrl+C to stop")

        while True:
            time.sleep(10)
            report = get_health_report()
            print("\n" + "="*50)
            print(report)
            print("="*50)

    except KeyboardInterrupt:
        print("\nStopping monitoring system...")
        stop_monitoring()

        # Export final health data
        monitoring_system.export_health_data("final_health_status.json")
        print("Health data exported to final_health_status.json")
