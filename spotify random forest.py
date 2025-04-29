import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve
)
import matplotlib.pyplot as plt

def load_data(filepath, nrows=None):
    df = pd.read_csv(filepath, nrows=nrows)
    return df

def create_target(df, target_col='completed', target_func=None):
    if target_func is not None:
        df[target_col] = df.apply(target_func, axis=1)
    else:
        df[target_col] = (df['reason_end'] == 'trackdone').astype(int)
    return df

def prepare_features(df, features, categorical_features):
    X = df[features]
    X_encoded = pd.get_dummies(X, columns=categorical_features)
    return X_encoded

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_rf(X_train, y_train, param_grid=None):
    if param_grid is None:
        param_grid = {'max_depth': [5, 10, 20, None],'max_leaf_nodes': [10, 20, 50, 100, None]}
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validated ROC-AUC:", grid_search.best_score_)
    return grid_search.best_estimator_

def find_best_threshold_for_precision(y_test, y_proba, target_recall=0.7):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    # Find all thresholds where recall >= target_recall
    possible = [(p, r, t) for p, r, t in zip(precisions, recalls, np.append(thresholds, 1.0)) if r >= target_recall]
    if possible:
        best = max(possible, key=lambda x: x[0])  # maximize precision
        best_precision, best_recall, best_threshold = best
        print(f"\nBest threshold for precision with recall >= {target_recall}: {best_threshold:.2f}")
        print(f"Precision: {best_precision:.3f}, Recall: {best_recall:.3f}")
    else:
        best_threshold = 0.5
        print("\nNo threshold found with recall above target. Using 0.5.")
    return best_threshold, precisions, recalls, thresholds

def evaluate_model(model, X_test, y_test, best_threshold, precisions, recalls, thresholds):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= best_threshold).astype(int)

    print("\nTest Set Metrics at chosen threshold:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Completed', 'Completed'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # Feature Importance
    importances = model.feature_importances_
    feature_names = X_test.columns
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    print("\nTop 10 Feature Importances:")
    print(feat_imp.head(10))

    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp['feature'][:10][::-1], feat_imp['importance'][:10][::-1])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.show()

    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], label='Precision')
    plt.plot(thresholds, recalls[:-1], label='Recall')
    plt.axvline(best_threshold, color='red', linestyle='--', label='Chosen Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs. Threshold')
    plt.legend()
    plt.show()

    # Accuracy vs. Threshold Curve
    accuracies = [(y_test == (y_proba >= t).astype(int)).mean() for t in thresholds]
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.axvline(best_threshold, color='red', linestyle='--', label='Chosen Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Threshold')
    plt.legend()
    plt.show()

    return y_pred, y_proba




# Load cleaned data
df = load_data('/Users/mason/Documents/Data Science/spotify audio json files/cleaned_data_01012023_01082025.csv')

# Create target
df = create_target(df)

#features can be modified for different experiments. *note: more features create a better model
# Prepare features
features = ['artist', 'track title', 'reason_start', 'album title']
categorical_features = ['artist', 'track title', 'reason_start', 'album title']
X_encoded = prepare_features(df, features, categorical_features)
y = df['completed']

# Split data for training
X_train, X_test, y_train, y_test = split_data(X_encoded, y)

#  Train model using random forest classifier
best_rf = train_rf(X_train, y_train)

#  Predict probabilities
y_proba = best_rf.predict_proba(X_test)[:, 1]

#  Find best threshold for precision with recall >= target. Target recall can be adjusted to fine tune the model
target_recall = 0.7 
best_threshold, precisions, recalls, thresholds = find_best_threshold_for_precision(y_test, y_proba, target_recall=target_recall)

# Evaluate model at chosen threshold
y_pred, y_proba = evaluate_model(best_rf, X_test, y_test, best_threshold, precisions, recalls, thresholds)