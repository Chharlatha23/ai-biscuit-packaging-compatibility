import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from preprocess import get_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES

def evaluate():
    print("Loading data and model...")
    df = pd.read_csv('processed_data/packaging_pairing_registry.csv')
    df = df[df['recommendation_status'] != 'Insufficient Data'].copy()

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df['recommendation_status']
    groups = df['product_id']

    # Use exact same grouped split as training would have used
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    preprocessor = joblib.load('models/preprocessor.joblib')
    model = joblib.load('models/packaging_model.joblib')

    # Preprocess
    X_test_processed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_processed)

    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)

    print("Evaluation Metrics:")
    print(report_str)

    # Save metrics JSON
    metrics = {
        'accuracy': accuracy,
        'macro_avg_f1': report_dict['macro avg']['f1-score'],
        'weighted_avg_f1': report_dict['weighted avg']['f1-score']
    }
    os.makedirs('reports', exist_ok=True)
    with open('reports/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png')
    plt.close()

    # Load metadata to get model type dynamically
    with open('models/model_metadata.json', 'r') as meta_f:
        meta = json.load(meta_f)
        model_name = meta.get('model_type', 'Random Forest')
        cv_acc = meta.get('cv_accuracy')

    # Save Human-Readable Report
    with open('reports/model_evaluation.md', 'w') as f:
        f.write("# Model Evaluation Report\n\n")
        f.write(f"**Number of training samples:** {len(X_train)}\n")
        f.write(f"**Number of test samples:** {len(X_test)}\n\n")
        f.write("**Features Used:**\n")
        for feature in features:
            f.write(f"- {feature}\n")
        f.write(f"\n**Target Used:** recommendation_status\n")
        f.write(f"**Final Model Selected:** {model_name}\n\n")

        f.write("## Metrics\n")
        f.write("```text\n")
        f.write(report_str)
        f.write("```\n\n")

        f.write("## Model Performance & Limitations\n")
        if cv_acc is not None:
            f.write(f"**The {model_name} achieved {cv_acc*100:.2f}% mean accuracy under 5-fold product-grouped cross-validation. On the fixed grouped holdout test set, it achieved {accuracy*100:.0f}% accuracy.**\n\n")
        else:
            f.write(f"**The {model_name} achieved {accuracy*100:.0f}% accuracy on the fixed grouped holdout test set. (Cross-validation accuracy metadata unavailable).**\n\n")

        f.write("**Both metrics evaluate agreement with deterministic rule-generated labels, not experimentally measured packaging compatibility.**\n\n")
        f.write("The model achieved the reported evaluation performance on the available compatibility-labelled dataset. Because the labels were generated using deterministic rules rather than independent experimental ground truth, these metrics indicate agreement with the existing decision logic and should not be interpreted as independently validated real-world packaging compatibility accuracy.\n\n")

        f.write("1. **Rule-Based Ground Truth:** The model approximates deterministically generated labels from the original rule engine.\n")
        f.write("2. **Small Dataset Size:** The dataset has limited representation of the `Conditionally Recommended` class, and is evaluated using a grouped split to avoid product leakage.\n")
        f.write("3. **Missing Features:** Important features like `storage_temperature_c` and `relative_humidity_percent` were completely unavailable in the training dataset and thus are excluded from the model.\n\n")

        f.write("## Conclusion\n")
        f.write("The model effectively replicates the deterministic rules, making it suitable for a **prototype demonstration** of an AI-assisted decision-support system. It is not ready for production deployment without real-world empirical validation.\n")

    print("Evaluation complete. Artifacts saved in reports/")

if __name__ == '__main__':
    evaluate()
