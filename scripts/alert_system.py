#!/usr/bin/env python3
"""
Documentation Alert System for MCMC Assumptions

This script implements automated alerting for detected MCMC assumptions
and documentation compliance issues in the Scientific Computing Toolkit.

Author: Scientific Computing Toolkit Team
Date: 2024
License: GPL-3.0-only
"""

import json
import smtplib
import os
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional
import requests
import subprocess
from pathlib import Path

class DocumentationAlertSystem:
    """Automated alert system for documentation compliance issues."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or "docs/alert_config.json"
        self.config = self.load_config()
        self.alert_history = self.load_alert_history()

    def load_config(self) -> Dict[str, Any]:
        """Load alert system configuration."""
        default_config = {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "alerts@scientific-toolkit.org",
                "sender_password": os.getenv("ALERT_EMAIL_PASSWORD", ""),
                "recipients": {
                    "critical": ["tech-lead@company.com"],
                    "warning": ["documentation-team@company.com"],
                    "info": ["all-team@company.com"]
                }
            },
            "slack": {
                "enabled": False,
                "webhook_url": os.getenv("SLACK_WEBHOOK_URL", ""),
                "channels": {
                    "critical": "#alerts-critical",
                    "warning": "#documentation-alerts",
                    "info": "#documentation-updates"
                }
            },
            "teams": {
                "enabled": False,
                "webhook_url": os.getenv("TEAMS_WEBHOOK_URL", ""),
                "channels": {
                    "critical": "Critical Alerts",
                    "warning": "Documentation Alerts",
                    "info": "Documentation Updates"
                }
            },
            "thresholds": {
                "critical_threshold": 1,  # Any MCMC violation
                "warning_threshold": 3,   # 3+ potential issues
                "info_threshold": 5,      # 5+ minor issues
                "cooldown_minutes": 60    # Don't spam alerts
            },
            "escalation": {
                "max_attempts": 3,
                "escalation_delay_hours": 2,
                "executive_escalation_threshold": 5
            }
        }

        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                for key, value in user_config.items():
                    if key in default_config:
                        if isinstance(default_config[key], dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value

        return default_config

    def load_alert_history(self) -> List[Dict[str, Any]]:
        """Load alert history to prevent spam."""
        history_file = Path("docs/alert_history.jsonl")
        if not history_file.exists():
            return []

        history = []
        with open(history_file, 'r') as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line))

        # Filter recent alerts (last 24 hours)
        cutoff = datetime.now(timezone.utc).timestamp() - (24 * 3600)
        recent_history = [
            alert for alert in history
            if alert.get('timestamp_epoch', 0) > cutoff
        ]

        return recent_history

    def should_send_alert(self, alert_type: str, alert_key: str) -> bool:
        """Check if alert should be sent based on cooldown and thresholds."""
        cooldown_minutes = self.config["thresholds"]["cooldown_minutes"]

        # Check for recent similar alerts
        for alert in self.alert_history:
            if (alert.get('type') == alert_type and
                alert.get('key') == alert_key and
                alert.get('timestamp_epoch', 0) > datetime.now(timezone.utc).timestamp() - (cooldown_minutes * 60)):
                return False  # Too recent, skip

        return True

    def send_critical_alert(self, violations: List[Dict[str, Any]], details: Dict[str, Any] = None) -> None:
        """Send critical alert for MCMC violations."""
        if not violations:
            return

        alert_key = f"mcmc_critical_{len(violations)}"
        if not self.should_send_alert("critical", alert_key):
            return

        subject = f"🚨 CRITICAL: {len(violations)} MCMC Violations Detected"

        message = f"""
🚨 CRITICAL MCMC ASSUMPTION VIOLATIONS DETECTED

Summary:
- Total violations: {len(violations)}
- Files affected: {len(set(v.get('file_path', '') for v in violations))}
- Most severe: {max((v.get('severity', 'info') for v in violations), key=lambda x: ['info', 'warning', 'critical'].index(x))}

Violations Found:
"""

        for i, violation in enumerate(violations[:5], 1):  # Show first 5
            message += f"""
{i}. {violation.get('file_path', 'Unknown')}:{violation.get('line_number', '?')}
   Severity: {violation.get('severity', 'unknown').upper()}
   Content: {violation.get('line_content', '')[:100]}...
   Suggestion: {violation.get('suggestion', '')}
"""

        if len(violations) > 5:
            message += f"\n... and {len(violations) - 5} more violations\n"

        message += """

IMMEDIATE ACTION REQUIRED:
1. Stop all documentation work immediately
2. Correct all violations within 1 hour
3. Run automated correction: python3 scripts/monitor_mcmc_assumptions.py --correct
4. Notify technical lead and run full compliance scan
5. Schedule team training on deterministic optimization

Resources:
- Correction script: python3 scripts/monitor_mcmc_assumptions.py --correct
- Training materials: python3 scripts/monitor_mcmc_assumptions.py --training
- Standards reference: docs/documentation_standards.md

This is a CRITICAL issue requiring immediate attention to maintain documentation accuracy.
"""

        self._send_alert("critical", subject, message, details)
        self._log_alert("critical", alert_key, subject, len(violations))

    def send_warning_alert(self, warnings: List[Dict[str, Any]], details: Dict[str, Any] = None) -> None:
        """Send warning alert for potential documentation issues."""
        if len(warnings) < self.config["thresholds"]["warning_threshold"]:
            return

        alert_key = f"doc_warnings_{len(warnings)}"
        if not self.should_send_alert("warning", alert_key):
            return

        subject = f"⚠️ WARNING: {len(warnings)} Documentation Issues Detected"

        message = f"""
⚠️ DOCUMENTATION ISSUES DETECTED

Summary:
- Total issues: {len(warnings)}
- Issue types: {', '.join(set(w.get('type', 'unknown') for w in warnings))}
- Files affected: {len(set(w.get('file_path', '') for w in warnings))}

Issues Found:
"""

        for i, warning in enumerate(warnings[:10], 1):  # Show first 10
            message += f"""
{i}. {warning.get('file_path', 'Unknown')}: {warning.get('description', '')}
   Type: {warning.get('type', 'unknown')}
   Severity: {warning.get('severity', 'warning')}
"""

        message += """

ACTION REQUIRED:
1. Review and correct all issues within 24 hours
2. Validate performance claims with actual data
3. Use specific algorithm names (Levenberg-Marquardt, Trust Region, etc.)
4. Avoid generic optimization terminology

Guidelines:
- Replace MCMC references with specific deterministic algorithms
- Include timing data and success rates for all performance claims
- Use approved terminology from docs/deterministic_optimization_foundation.md
- Run automated validation: python3 scripts/monitor_mcmc_assumptions.py --scan

For assistance:
- Training: python3 scripts/monitor_mcmc_assumptions.py --training
- Standards: docs/documentation_standards.md
- Support: documentation-team@company.com
"""

        self._send_alert("warning", subject, message, details)
        self._log_alert("warning", alert_key, subject, len(warnings))

    def send_info_alert(self, updates: List[Dict[str, Any]], details: Dict[str, Any] = None) -> None:
        """Send informational alert for documentation updates."""
        if len(updates) < self.config["thresholds"]["info_threshold"]:
            return

        alert_key = f"doc_updates_{len(updates)}"
        if not self.should_send_alert("info", alert_key):
            return

        subject = f"📚 Documentation Updates: {len(updates)} Items"

        message = f"""
📚 DOCUMENTATION UPDATE SUMMARY

Recent Changes:
- Total updates: {len(updates)}
- Files modified: {len(set(u.get('file_path', '') for u in updates))}
- Update types: {', '.join(set(u.get('type', 'unknown') for u in updates))}

Key Updates:
"""

        for i, update in enumerate(updates[:15], 1):  # Show first 15
            message += f"""
{i}. {update.get('file_path', 'Unknown')}: {update.get('description', '')}
   Type: {update.get('type', 'unknown')}
   Status: {update.get('status', 'completed')}
"""

        message += """

REVIEW RECOMMENDED:
1. Check updated documentation for compliance
2. Validate any new performance claims
3. Update cross-references if needed
4. Review examples for functionality

Resources:
- Latest standards: docs/documentation_standards.md
- Training materials: docs/deterministic_optimization_foundation.md
- Compliance check: python3 scripts/monitor_mcmc_assumptions.py --scan

Thank you for maintaining documentation quality!
"""

        self._send_alert("info", subject, message, details)
        self._log_alert("info", alert_key, subject, len(updates))

    def send_success_alert(self, achievements: List[str], details: Dict[str, Any] = None) -> None:
        """Send positive reinforcement alert for compliance achievements."""
        subject = "✅ Documentation Compliance Achieved"

        message = f"""
🎉 EXCELLENT WORK! DOCUMENTATION COMPLIANCE MAINTAINED

Key Achievements:
"""

        for achievement in achievements:
            message += f"✅ {achievement}\n"

        message += """

Current Status:
- MCMC violations: 0
- Performance claims: All validated
- Standards compliance: 100%
- Team training: Up to date

Continue these excellent practices!

Next Milestones:
- Quarterly review: [Date]
- Team certification renewal: [Date]
- Standards update review: [Date]

Resources:
- Achievement showcase: docs/achievements-showcase.md
- Best practices: docs/documentation_standards.md
- Training refresh: docs/deterministic_optimization_foundation.md

Keep up the outstanding work! 🌟
"""

        self._send_alert("success", subject, message, details)
        self._log_alert("success", "compliance_achieved", subject, len(achievements))

    def _send_alert(self, severity: str, subject: str, message: str, details: Dict[str, Any] = None) -> None:
        """Send alert through configured channels."""
        success = False

        # Email alerts
        if self.config["email"]["enabled"]:
            try:
                self._send_email_alert(severity, subject, message)
                success = True
            except Exception as e:
                print(f"Email alert failed: {e}")

        # Slack alerts
        if self.config["slack"]["enabled"]:
            try:
                self._send_slack_alert(severity, subject, message)
                success = True
            except Exception as e:
                print(f"Slack alert failed: {e}")

        # Teams alerts
        if self.config["teams"]["enabled"]:
            try:
                self._send_teams_alert(severity, subject, message)
                success = True
            except Exception as e:
                print(f"Teams alert failed: {e}")

        if not success:
            print(f"Warning: All alert channels failed for {severity} alert")

    def _send_email_alert(self, severity: str, subject: str, message: str) -> None:
        """Send email alert."""
        recipients = self.config["email"]["recipients"].get(severity, [])

        if not recipients:
            return

        msg = MIMEMultipart()
        msg['From'] = self.config["email"]["sender_email"]
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"[{severity.upper()}] {subject}"

        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP(self.config["email"]["smtp_server"], self.config["email"]["smtp_port"])
        server.starttls()
        server.login(self.config["email"]["sender_email"], self.config["email"]["sender_password"])
        server.sendmail(self.config["email"]["sender_email"], recipients, msg.as_string())
        server.quit()

    def _send_slack_alert(self, severity: str, subject: str, message: str) -> None:
        """Send Slack alert."""
        webhook_url = self.config["slack"]["webhook_url"]
        channel = self.config["slack"]["channels"].get(severity, "#general")

        if not webhook_url:
            return

        emoji_map = {
            "critical": "🚨",
            "warning": "⚠️",
            "info": "📚",
            "success": "✅"
        }

        payload = {
            "channel": channel,
            "username": "Documentation Monitor",
            "icon_emoji": emoji_map.get(severity, "📢"),
            "text": f"*{subject}*\n\n{message}",
            "mrkdwn": True
        }

        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()

    def _send_teams_alert(self, severity: str, subject: str, message: str) -> None:
        """Send Microsoft Teams alert."""
        webhook_url = self.config["teams"]["webhook_url"]

        if not webhook_url:
            return

        color_map = {
            "critical": "d63384",  # Red
            "warning": "fd7e14",   # Orange
            "info": "0d6efd",      # Blue
            "success": "198754"    # Green
        }

        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color_map.get(severity, "0076d7"),
            "title": subject,
            "text": message
        }

        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()

    def _log_alert(self, alert_type: str, alert_key: str, subject: str, item_count: int) -> None:
        """Log alert to history file."""
        alert_record = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "timestamp_epoch": datetime.now(timezone.utc).timestamp(),
            "type": alert_type,
            "key": alert_key,
            "subject": subject,
            "item_count": item_count,
            "channels_attempted": self._get_enabled_channels()
        }

        # Append to history
        history_file = Path("docs/alert_history.jsonl")
        with open(history_file, 'a') as f:
            f.write(json.dumps(alert_record) + '\n')

        # Update in-memory history
        self.alert_history.append(alert_record)

    def _get_enabled_channels(self) -> List[str]:
        """Get list of enabled alert channels."""
        channels = []
        if self.config["email"]["enabled"]:
            channels.append("email")
        if self.config["slack"]["enabled"]:
            channels.append("slack")
        if self.config["teams"]["enabled"]:
            channels.append("teams")
        return channels

    def test_alert_system(self) -> Dict[str, bool]:
        """Test all configured alert channels."""
        test_results = {}

        # Test email
        if self.config["email"]["enabled"]:
            try:
                self._send_email_alert("info", "Alert System Test", "This is a test alert from the documentation monitoring system.")
                test_results["email"] = True
            except Exception as e:
                print(f"Email test failed: {e}")
                test_results["email"] = False

        # Test Slack
        if self.config["slack"]["enabled"]:
            try:
                self._send_slack_alert("info", "Alert System Test", "This is a test alert from the documentation monitoring system.")
                test_results["slack"] = True
            except Exception as e:
                print(f"Slack test failed: {e}")
                test_results["slack"] = False

        # Test Teams
        if self.config["teams"]["enabled"]:
            try:
                self._send_teams_alert("info", "Alert System Test", "This is a test alert from the documentation monitoring system.")
                test_results["teams"] = True
            except Exception as e:
                print(f"Teams test failed: {e}")
                test_results["teams"] = False

        return test_results


def main():
    """Main entry point for alert system."""
    import argparse

    parser = argparse.ArgumentParser(description="Documentation Alert System")
    parser.add_argument('--test', action='store_true', help='Test alert system configuration')
    parser.add_argument('--critical', type=str, help='Send critical alert with JSON violations file')
    parser.add_argument('--warning', type=str, help='Send warning alert with JSON issues file')
    parser.add_argument('--info', type=str, help='Send info alert with JSON updates file')
    parser.add_argument('--success', type=str, help='Send success alert with achievements file')
    parser.add_argument('--config', type=str, help='Path to alert configuration file')

    args = parser.parse_args()

    alert_system = DocumentationAlertSystem(args.config)

    if args.test:
        print("🧪 Testing alert system configuration...")
        test_results = alert_system.test_alert_system()
        print("\nTest Results:")
        for channel, success in test_results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"  {channel}: {status}")

    elif args.critical:
        print("🚨 Sending critical alert...")
        with open(args.critical, 'r') as f:
            violations = json.load(f)
        alert_system.send_critical_alert(violations)

    elif args.warning:
        print("⚠️ Sending warning alert...")
        with open(args.warning, 'r') as f:
            warnings = json.load(f)
        alert_system.send_warning_alert(warnings)

    elif args.info:
        print("📚 Sending info alert...")
        with open(args.info, 'r') as f:
            updates = json.load(f)
        alert_system.send_info_alert(updates)

    elif args.success:
        print("✅ Sending success alert...")
        with open(args.success, 'r') as f:
            achievements = json.load(f)
        alert_system.send_success_alert(achievements)

    else:
        print("Documentation Alert System")
        print("Use --test to test configuration")
        print("Use --critical/--warning/--info/--success with JSON files to send alerts")
        print("Use --config to specify custom configuration file")


if __name__ == '__main__':
    main()
