import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from preprocess import get_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES

def train():
    print("Loading data...")
    df = pd.read_csv('processed_data/packaging_pairing_registry.csv')
    
    # Filter out Insufficient Data as per requirements
    df = df[df['recommendation_status'] != 'Insufficient Data'].copy()
    print(f"Data shape after filtering: {df.shape}")
    
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df['recommendation_status']
    groups = df['product_id']
    
    # Check for small classes
    class_counts = y.value_counts()
    print("Target class distribution:")
    print(class_counts)
    
    if len(class_counts) < 2:
        raise ValueError("Not enough classes for classification.")
        
    # Split data ensuring biscuit groups are kept together
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    
    # Verify class representation in train and test
    train_classes = set(y_train.unique())
    test_classes = set(y_test.unique())
    expected_classes = set(y.unique())
    
    if not expected_classes.issubset(train_classes):
        raise ValueError(f"Train split is missing classes! Expected {expected_classes}, got {train_classes}")
    if not expected_classes.issubset(test_classes):
        raise ValueError(f"Test split is missing classes! Expected {expected_classes}, got {test_classes}")
    
    print("Evaluating models with grouped cross-validation on training data...")
    preprocessor = get_preprocessor()
    
    # Build models using Pipeline to avoid data leakage
    models = {
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ]),
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
    }
    
    best_model_name = None
    best_score = 0
    
    gkf = GroupKFold(n_splits=5)
    
    for name, pipeline in models.items():
        # Using 5-fold GroupKFold CV
        scores = cross_val_score(pipeline, X_train, y_train, groups=groups_train, cv=gkf, scoring='accuracy')
        mean_score = scores.mean()
        print(f"{name} Grouped CV Accuracy: {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
            
    print(f"\nSelecting best model: {best_model_name}")
    
    # Train the best pipeline on the full training dataset
    final_pipeline = models[best_model_name]
    final_pipeline.fit(X_train, y_train)
    
    # Extract the fitted preprocessor and classifier to save separately (for API consistency)
    fitted_preprocessor = final_pipeline.named_steps['preprocessor']
    fitted_model = final_pipeline.named_steps['classifier']
    
    # Save artifacts
    print("Saving models and metadata...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(fitted_preprocessor, 'models/preprocessor.joblib')
    joblib.dump(fitted_model, 'models/packaging_model.joblib')
    
    metadata = {
        'model_type': best_model_name,
        'dataset_path': 'processed_data/packaging_pairing_registry.csv',
        'total_usable_rows': len(df),
        'training_sample_count': len(X_train),
        'testing_sample_count': len(X_test),
        'cv_accuracy': best_score,
        'feature_names': features,
        'target_name': 'recommendation_status',
        'target_classes': list(fitted_model.classes_),
        'random_state': 42,
        'evaluation_method': 'GroupKFold (n=5) by product_id',
        'training_limitations': 'The model is trained on deterministically generated rules. It approximates compatibility rules but lacks independent empirical shelf-life ground truth. The grouped cross-validation ensures no product leakage.'
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Training complete.")

if __name__ == '__main__':
    train()
