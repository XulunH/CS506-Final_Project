import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Load data
data = pd.read_csv('../datasets/preprocessed_dataset.csv')

# List of categorical features
categorical_features = [
    'Weather Conditions',
    'Light Conditions',
    'Road Surface Condition',
    'Age of Driver - Youngest Known',
    'Age of Driver - Oldest Known',
    'Driver Contributing Circumstances (All Drivers)',
    'Driver Distracted By (All Vehicles)',
    'First Harmful Event',
    'Manner of Collision',
    'Roadway Junction Type',
    'Traffic Control Device Type',
    'Trafficway Description',
    'Vehicle Actions Prior to Crash (All Vehicles)',
    'Vehicle Configuration (All Vehicles)',
    'Vehicle Emergency Use (All Vehicles)',
    'First Harmful Event Location',
    'City Town Name'
]

# Fill missing values in categorical features with 'Unknown'
for col in categorical_features:
    data[col] = data[col].fillna('Unknown')

# List of numerical features
numerical_features = [ 'Crash Hour Numeric', 'Number of Vehicles', 'Speed Limit']
# Prepare features and labels
X = data[categorical_features + numerical_features]
y = data['Crash Severity']

# Encode target variable
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)

# Preprocessing pipelines
# For categorical data, use OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# For numerical data, use StandardScaler
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine transformers
preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features),
    ('num', numerical_transformer, numerical_features)
])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2
)

# List of models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=10),
    'XGBoost': XGBClassifier(eval_metric='mlogloss')
}

for model_name, classifier in models.items():
    print(f"\nTraining the {model_name} model...")
    # Create a pipeline for each classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions on the test set
    print(f"Making predictions with the {model_name} model...")
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_enc.classes_)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"{model_name} Model Accuracy: {accuracy}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", conf_matrix)
    
    # Save evaluation metrics
    with open(f"../result_files/{model_name}_evaluation.txt", "w") as f:
        f.write(f"{model_name} Model Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(conf_matrix))
    
    # Feature importance visualization
    print(f"Visualizing and saving feature importance for {model_name}...")
    
    # Get feature names from preprocessor
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    if model_name == 'Logistic Regression':
        # For logistic regression, coefficients represent feature importance
        coefficients = pipeline.named_steps['classifier'].coef_[0]
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coefficients
        })
    else:
        # For tree-based models, use feature_importances_
        importances = pipeline.named_steps['classifier'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
    
    # Sort by absolute value of importance
    importance_df['Abs_Importance'] = importance_df['Importance'].abs()
    importance_df = importance_df.sort_values(by='Abs_Importance', ascending=False)
    
    # Optionally shorten feature names
    importance_df['Feature'] = importance_df['Feature'].str.replace('cat__', '')
    importance_df['Feature'] = importance_df['Feature'].str.replace('num__', '')
    
    # Visualize and save feature importance
    plt.figure(figsize=(15, 10))
    plt.barh(
        y=importance_df['Feature'][:20], 
        width=importance_df['Importance'][:20], 
        align='center'
    )
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title(f'Top 20 Feature Importances ({model_name})')
    plt.yticks(fontsize=10)
    plt.subplots_adjust(left=0.4)
    plt.tight_layout()
    plt.savefig(f"../result_pics/{model_name}_feature_importance.png", bbox_inches='tight')
    plt.show()
