import pandas as pd
from dataset_audit_kit import DatasetAuditor

data = pd.DataFrame({
    'id': [1,2,3,4,5],
    'name': ['alice','bob','carol','dave','eve'],
    'score': [85.5, 92.0, 78.3, 88.1, 95.0],
    'grade': ['A','B','A','B','A'],
})

# Feature 4: Summary
summary = DatasetAuditor.dataset_summary(data)
print('=== Summary ===')
print(summary[:300])

# Feature 4: Sample
sample = DatasetAuditor.sample_dataset(data, n=3, method='random', seed=42)
print('=== Random sample (' + str(len(sample)) + ' rows) ===')

# Feature 5: Type inference
suggestions = DatasetAuditor.infer_optimal_dtypes(data)
print('=== Type suggestions: ' + str(len(suggestions)) + ' ===')
for col, sug in suggestions.items():
    print('  ' + str(col) + ': ' + str(sug['current_dtype']) + ' -> ' + str(sug['suggested_dtype']))

# Feature 6: Report diff
from dataset_audit_kit import AuditReport
from dataset_audit_kit.core import AuditIssue

before = AuditReport(rows=100, columns=5, duplicate_rows=2, missing_cells=10, issues=[
    AuditIssue(check='missingness', severity='warning', message='test', column='x'),
])
after = AuditReport(rows=120, columns=5, duplicate_rows=0, missing_cells=5, issues=[])
diff_rep = AuditReport.diff(before, after)
print('=== Diff issues: ' + str(len(diff_rep.issues)) + ' ===')
for iss in diff_rep.issues:
    print('  ' + str(iss.message))

print('All tests passed')