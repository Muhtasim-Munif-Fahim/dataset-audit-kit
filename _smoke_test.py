import pandas as pd
from dataset_audit_kit import DatasetAuditor

data = pd.DataFrame({"feat": [1.0, 2.0, None, 100.0], "label": ["A", "B", "A", "X"]})
auditor = DatasetAuditor()
report = auditor.audit_dataframe(data, label_column="label")
suggestions = report.fix_suggestions
print(f"Suggestions count: {len(suggestions)}")
for s in suggestions:
    print(f"  [{s['action']}] {s['description'][:60]}")

import tempfile, os
path = os.path.join(tempfile.gettempdir(), "test_report.json")
report.to_file(path)
with open(path) as f:
    content = f.read()
print(f"JSON saved, length: {len(content)} chars")
os.remove(path)

path = os.path.join(tempfile.gettempdir(), "test_report.md")
report.to_file(path)
with open(path) as f:
    content = f.read()
print(f"Markdown saved, length: {len(content)} chars")
os.remove(path)
print("All Push 3 smoke tests passed")
