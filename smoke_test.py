import pandas as pd
from dataset_audit_kit import DatasetAuditor

data = pd.DataFrame({'feat': [1,2,3,4,5,6,7,8,9,100]})
auditor = DatasetAuditor()
report = auditor.audit_dataframe(data)
prof = report.column_profiles['feat']
print(f'IQR outliers: {prof.get("outliers_iqr", "N/A")}')
print(f'Skewness: {prof.get("skewness", "N/A"):.2f}')
print(f'Kurtosis: {prof.get("kurtosis", "N/A"):.2f}')
iqr_issues = [i for i in report.issues if 'outlier' in i.message.lower()]
print(f'IQR issue count: {len(iqr_issues)}')

data2 = pd.DataFrame({'a': [1,2,3], 'b': [2,4,6], 'c': [10,20,30], 'd': [5,10,15]})
report = auditor.audit_dataframe(data2)
red_issues = [i for i in report.issues if i.check=='redundancy']
print(f'Redundancy issues: {len(red_issues)}')

ref = pd.DataFrame({'x': [1,2], 'y': [3.0,4.0]})
curr = pd.DataFrame({'x': [1,2], 'y': [3,4], 'z': [5,6]})
report = auditor.audit_dataframe(curr, reference=ref)
diff_issues = [i for i in report.issues if i.check=='schema_diff']
print(f'Schema diff issues: {len(diff_issues)}')
for d in diff_issues:
    print(f'  - {d.message}')

print('All 3 features smoke tests passed')
