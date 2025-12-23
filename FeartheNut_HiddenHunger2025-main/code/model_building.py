# code/model_building.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import joblib

from data_cleaning import clean_data

def main():
    # ensure outputs folder exists
    base = Path(__file__).resolve().parents[1]  # project root (FeartheNut_HiddenHunger2025-main)
    outputs_dir = base / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # load data and preprocess object
    df, preprocess = clean_data()

    target_col = "Hidden_Hunger_Flag"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # If preprocess is a ColumnTransformer that expects the original dataframe columns,
    # it's easier to build a pipeline that applies the transformer then the classifier.
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", clf)
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Fit
    pipeline.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = pipeline.predict(X_test)
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        # if predict_proba not available, fall back to zeros
        y_proba = np.zeros_like(y_pred, dtype=float)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else None

    print("Test Accuracy:", acc)
    print("Test F1:", f1)
    if auc is not None:
        print("Test ROC AUC:", auc)

    # Save model
    model_path = outputs_dir / "my_model.joblib"
    joblib.dump(pipeline, model_path)
    print("Saved trained model to:", model_path)

    # Save predictions to CSV: include index from original X_test
    pred_df = X_test.copy()
    pred_df["true_label"] = y_test.values
    pred_df["predicted_label"] = y_pred
    pred_df["predicted_proba"] = y_proba
    pred_df.to_csv(outputs_dir / "risk_predictions.csv", index=False)
    print("Saved predictions to outputs/risk_predictions.csv")

    # Save a simple metrics summary as key_findings.csv
    metrics = {
        "metric": ["accuracy", "f1", "roc_auc"],
        "value": [acc, f1, auc if auc is not None else ""]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(outputs_dir / "key_findings.csv", index=False)
    print("Saved metrics to outputs/key_findings.csv")

    # Save classification report text
    report = classification_report(y_test, y_pred, zero_division=0)
    with open(outputs_dir / "classification_report.txt", "w") as f:
        f.write(report)
    print("Saved classification report to outputs/classification_report.txt")

    # Optional: cross-validate for more robust metrics
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=["accuracy","f1"], n_jobs=-1)
    cv_summary = {
        "cv_metric": ["accuracy_mean", "accuracy_std", "f1_mean", "f1_std"],
        "value": [
            cv_results["test_accuracy"].mean(),
            cv_results["test_accuracy"].std(),
            cv_results["test_f1"].mean(),
            cv_results["test_f1"].std()
        ]
    }
    pd.DataFrame(cv_summary).to_csv(outputs_dir / "cv_summary.csv", index=False)
    print("Saved cross-validation summary to outputs/cv_summary.csv")

if __name__ == "__main__":
    main()
