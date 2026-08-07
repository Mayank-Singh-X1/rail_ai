import glob
import re

for filepath in glob.glob('railway-surveillance-ai/modules/*.py'):
    if filepath.endswith('__init__.py'):
        continue
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove any notebook global instantiation lines at module level like:
    # crowd_analyzer = CrowdAnalyzer(system)
    # criminal_detector = CriminalDetector(system)
    # anomaly_detector = AnomalyDetector(system)
    # cleanliness_monitor = CleanlinessMonitor(system)
    # worker_monitor = WorkerMonitor(system)
    # person_tracker = PersonTracker(system)
    # alert_system = AlertSystem()

    patterns = [
        r'\n# Instantiate [^\n]+\n[a-z_]+ = [A-Za-z0-9_]+\([^\)]*\)\s*$',
        r'\n[a-z_]+ = [A-Za-z0-9_]+\([^\)]*\)\s*$'
    ]
    for p in patterns:
        content = re.sub(p, '\n', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content.strip() + '\n')
    print(f"Cleaned {filepath}")
