"""Dataset audit demo using the public Iris dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_iris

from dataset_audit_kit import DatasetAuditor


def build_demo_frame():
    iris = load_iris(as_frame=True)
    frame = iris.frame.rename(columns={"target": "species"})
    reference = frame.sample(frac=0.70, random_state=7).reset_index(drop=True)

    demo = frame.copy().reset_index(drop=True)
    demo.loc[0, "sepal length (cm)"] = None
    demo = demo.iloc[[0, 1, 2, 3, 4]].copy()
    demo = pd.concat([demo, demo.iloc[[0]]], ignore_index=True)
    return demo, reference


def main() -> None:
    data, reference = build_demo_frame()
    auditor = DatasetAuditor(missing_threshold=0.10, drift_threshold=0.15)
    report = auditor.audit_dataframe(
        data,
        reference=reference,
        label_column="species",
        expected_columns=data.columns.tolist(),
    )
    print(report.to_markdown())


if __name__ == "__main__":
    main()
