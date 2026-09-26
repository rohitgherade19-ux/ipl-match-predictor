"""
Train Random Forest model for IPL match winner prediction.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from ml.feature_engineering import build_features


def train():
    print("=" * 60)
    print("  IPL Match Winner Prediction — Model Training")
    print("=" * 60)

    print("\n[1/4] Building features (this may take a moment)...")
    df = build_features()

    if len(df) == 0:
        print("ERROR: Feature dataframe is empty. Check db_setup.")
        return

    print(f"      >> {len(df)} samples, {len(df.columns) - 1} features")
    print(f"      >> Target distribution: {dict(df['team1_wins'].value_counts())}")

    X = df.drop('team1_wins', axis=1)
    y = df['team1_wins']

    # ---- Train / Test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"      >> Train: {len(X_train)}, Test: {len(X_test)}")

    # ---- GridSearchCV with Random Forest ----
    print("\n[2/4] Running GridSearchCV (Random Forest)...")
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [6, 10, 15, None],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [3, 5, 8],
        'max_features': ['sqrt', 'log2'],
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print(f"      >> Best params: {grid.best_params_}")
    print(f"      >> Best CV accuracy: {grid.best_score_:.4f}")

    # ---- Evaluate on test set ----
    print("\n[3/4] Evaluating on test set...")
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n      *** Test Accuracy: {acc:.4f} ({acc*100:.1f}%) ***\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Team 2 wins', 'Team 1 wins']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ---- Feature importances ----
    print("\nFeature Importances (top 15):")
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    for i, (feat, imp) in enumerate(feat_imp.head(15).items()):
        bar = '#' * int(imp * 100)
        print(f"  {i+1:2d}. {feat:30s} {imp:.4f}  {bar}")

    # ---- Cross-validation score ----
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
    print(f"\n5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ---- Save model ----
    print("\n[4/4] Saving model...")
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'rf_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"      >> Saved to {model_path}")
    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    train()
