# ============================================
# Alert & Notification System
# ============================================
import json
import os
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib


class AlertSystem:

  def __init__(self):
    self.alert_log = []
    self.email_config = {
        'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('SMTP_PORT', 587)),
        'sender_email': os.getenv('SENDER_EMAIL', 'your_email@gmail.com'),
        'sender_password': os.getenv('SENDER_PASSWORD', 'your_app_password'),
        'receiver_emails': ['security@railway.gov.in'],
    }

  def send_alert(self, alert_data):
    """Dispatch alerts across console, log files, and email based on severity."""
    if 'timestamp' not in alert_data:
      alert_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    self.alert_log.append(alert_data)
    severity = alert_data.get('severity', 'LOW').upper()

    self._console_alert(alert_data)
    self._log_alert(alert_data)

    if severity in ['HIGH', 'CRITICAL']:
      self._email_alert(alert_data)

  def _console_alert(self, alert_data):
    """Display formatted notification in the terminal/console."""
    severity = alert_data.get('severity', 'UNKNOWN')
    icon = '🚨' if severity in ['HIGH', 'CRITICAL'] else '⚠️'

    print(f"\n{'='*50}")
    print(f"{icon} ALERT DETECTED: {alert_data.get('type', 'General Alert')}")
    print(f"📍 Location: {alert_data.get('location', 'Platform Stream 01')}")
    print(f"⏰ Timestamp: {alert_data.get('timestamp')}")
    print(f"⚡ Severity Level: {severity}")
    if 'name' in alert_data:
      print(f"👤 Target ID: {alert_data['name']}")
    if 'details' in alert_data:
      print(f"📝 Notes: {alert_data['details']}")
    print(f"{'='*50}\n")

  def _email_alert(self, alert_data):
    """Send formatted alert emails to security personnel."""
    if self.email_config['sender_email'] == 'your_email@gmail.com':
      print("📧 [Simulated Email] High-severity alert triggered!")
      return

    try:
      msg = MIMEMultipart()
      msg['From'] = self.email_config['sender_email']
      msg['To'] = ', '.join(self.email_config['receiver_emails'])
      msg['Subject'] = (
          f"🚨 Railway Security Alert: {alert_data.get('type', 'Incident')}"
      )

      body = f"""
            RAILWAY SURVEILLANCE AUTOMATED ALERT
            ------------------------------------
            Alert Type : {alert_data.get('type', 'Unknown')}
            Severity   : {alert_data.get('severity', 'HIGH')}
            Time       : {alert_data.get('timestamp')}
            Location   : {alert_data.get('location', 'Platform Stream')}
            
            Incident Details:
            {json.dumps(alert_data, indent=2, default=str)}
            """
      msg.attach(MIMEText(body, 'plain'))

      server = smtplib.SMTP(
          self.email_config['smtp_server'], self.email_config['smtp_port']
      )
      server.starttls()
      server.login(
          self.email_config['sender_email'],
          self.email_config['sender_password'],
      )
      server.sendmail(
          self.email_config['sender_email'],
          self.email_config['receiver_emails'],
          msg.as_string(),
      )
      server.quit()

      print("📧 Email alert dispatched successfully!")
    except Exception as e:
      print(f"❌ Failed to deliver email alert: {e}")

  def _log_alert(self, alert_data):
    """Append structured telemetry record to alerts_log.json."""
    try:
      with open('alerts_log.json', 'a') as f:
        f.write(json.dumps(alert_data, default=str) + '\n')
    except Exception as e:
      print(f"❌ Failed to log alert to disk: {e}")

  def get_alert_summary(self):
    """Summarize aggregated alerts by type and severity level."""
    summary = {
        'total_alerts': len(self.alert_log),
        'by_type': {},
        'by_severity': {},
    }

    for alert in self.alert_log:
      alert_type = alert.get('type', 'Unknown')
      severity = alert.get('severity', 'Unknown')

      summary['by_type'][alert_type] = (
          summary['by_type'].get(alert_type, 0) + 1
      )
      summary['by_severity'][severity] = (
          summary['by_severity'].get(severity, 0) + 1
      )

    return summary
