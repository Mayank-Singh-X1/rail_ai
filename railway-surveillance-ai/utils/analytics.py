# ============================================
# CELL 13: Analytics & Visualization
# ============================================
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')


class AnalyticsDashboard:

  @staticmethod
  def plot_crowd_trend(crowd_history):
    """Plot crowd count over time."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    counts = [
        h['count'] if isinstance(h, dict) else h for h in crowd_history
    ]
    if not counts:
      counts = [0]

    timestamps = list(range(len(counts)))

    ax.fill_between(timestamps, counts, alpha=0.3, color='blue')
    ax.plot(timestamps, counts, color='blue', linewidth=2)
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('People Count')
    ax.set_title('👥 Crowd Density Over Time')
    ax.grid(True, alpha=0.3)

    # Threshold lines
    ax.axhline(
        y=50, color='orange', linestyle='--', label='Medium Threshold', alpha=0.7
    )
    ax.axhline(
        y=100, color='red', linestyle='--', label='High Threshold', alpha=0.7
    )
    ax.legend()

    plt.tight_layout()
    output_filename = 'crowd_trend.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    return output_filename

  @staticmethod
  def plot_alert_distribution(alerts):
    """Plot alert type distribution and severity pie chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    type_counts = {}
    severity_counts = {}

    for alert in alerts:
      t = alert.get('type', 'Unknown')
      s = alert.get('severity', 'Unknown')
      type_counts[t] = type_counts.get(t, 0) + 1
      severity_counts[s] = severity_counts.get(s, 0) + 1

    if type_counts:
      ax1.barh(
          list(type_counts.keys()),
          list(type_counts.values()),
          color=['#e74c3c', '#e67e22', '#3498db', '#2ecc71'][: len(type_counts)],
      )
      ax1.set_title('🚨 Alerts by Type')
      ax1.set_xlabel('Count')
    else:
      ax1.text(
          0.5,
          0.5,
          'No Alerts Recorded',
          ha='center',
          va='center',
          fontsize=12,
          color='gray',
      )

    if severity_counts:
      colors = {
          'LOW': '#2ecc71',
          'MEDIUM': '#f1c40f',
          'HIGH': '#e67e22',
          'CRITICAL': '#e74c3c',
      }
      ax2.pie(
          list(severity_counts.values()),
          labels=list(severity_counts.keys()),
          colors=[colors.get(s, '#95a5a6') for s in severity_counts.keys()],
          autopct='%1.1f%%',
          startangle=90,
      )
      ax2.set_title('⚡ Alerts by Severity')
    else:
      ax2.text(
          0.5,
          0.5,
          'No Severity Data',
          ha='center',
          va='center',
          fontsize=12,
          color='gray',
      )

    plt.tight_layout()
    output_filename = 'alert_distribution.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    return output_filename

  @staticmethod
  def plot_zone_analytics(zone_counts_history):
    """Plot zone-wise people count over time."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if zone_counts_history:
      for zone_name, counts in zone_counts_history.items():
        ax.plot(counts, label=zone_name, linewidth=2)
      ax.legend()
    else:
      ax.text(
          0.5,
          0.5,
          'No Zone Data Available',
          ha='center',
          va='center',
          fontsize=12,
          color='gray',
      )

    ax.set_xlabel('Time')
    ax.set_ylabel('People Count')
    ax.set_title('📍 Zone-wise Crowd Distribution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_filename = 'zone_analytics.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    return output_filename

  @staticmethod
  def generate_report(results):
    """Generate comprehensive Markdown analysis report."""
    if not results:
      return "⚠️ No analysis results available to generate report."

    total_frames = len(results)
    avg_crowd = np.mean([r.get('crowd_count', 0) for r in results])
    max_crowd = max([r.get('crowd_count', 0) for r in results])
    avg_clean = np.mean([r.get('cleanliness_score', 100) for r in results])
    total_criminals = sum([r.get('criminals', 0) for r in results])
    total_anomalies = sum([r.get('anomalies', 0) for r in results])

    report = f"""
# 🚂 Indian Railways AI Surveillance Report
## Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Summary Statistics

| Metric | Value |
|--------|-------|
| Total Frames Analyzed | {total_frames} |
| Average Crowd Size | {avg_crowd:.1f} |
| Peak Crowd Size | {max_crowd} |
| Average Cleanliness | {avg_clean:.1f}% |
| Total Criminal Alerts | {total_criminals} |
| Total Anomaly Events | {total_anomalies} |

---

## 🎯 Recommendations

1. **Crowd Management**: {"⚠️ Deploy additional staff to control platform congestion" if max_crowd > 100 else "✅ Current staffing adequate"}
2. **Security**: {"🚨 Patrol frequency should be increased immediately" if total_criminals > 0 else "✅ No immediate high-priority security concerns"}
3. **Cleanliness**: {"🧹 Schedule immediate cleaning drive" if avg_clean < 70 else "✅ Cleanliness standards met"}

---
*Report generated by Indian Railways AI Surveillance System*
"""
    return report


# Instantiate Analytics Dashboard module
analytics = AnalyticsDashboard()