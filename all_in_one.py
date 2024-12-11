import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# Load data
data = pd.read_csv('./revised_traffic_dataset.csv')

# Drop rows with missing 'Crash Severity'
data.dropna(subset=['Crash Severity'], inplace=True)

# Drop rows where 'Crash Severity' is 'Unknown'
data = data[data['Crash Severity'] != 'Unknown']

# Process 'Crash Hour' column to extract hour
def extract_hour(crash_hour_range):
    try:
        start_time = crash_hour_range.split(' to ')[0]
        hour = int(pd.to_datetime(start_time, format='%I:%M%p').hour)
        return hour
    except:
        return np.nan

data['Crash Hour Numeric'] = data['Crash Hour'].apply(extract_hour)

# Fill missing 'Crash Hour Numeric' with median
data['Crash Hour Numeric'] = data['Crash Hour Numeric'].fillna(data['Crash Hour Numeric'].median())

# Convert 'Number of Vehicles' and 'Speed Limit' to numeric
data['Number of Vehicles'] = pd.to_numeric(data['Number of Vehicles'], errors='coerce').fillna(data['Number of Vehicles'].median())
data['Speed Limit'] = pd.to_numeric(data['Speed Limit'], errors='coerce').fillna(data['Speed Limit'].median())

# Simplify 'Weather Conditions'
def simplify_weather(condition):
    if pd.isnull(condition) or condition.strip() == '':
        return 'Unknown'
    condition = condition.lower()
    if 'clear' in condition:
        return 'Clear'
    elif 'cloudy' in condition:
        return 'Cloudy'
    elif 'rain' in condition:
        return 'Rain'
    elif 'snow' in condition:
        return 'Snow'
    elif 'sleet' in condition or 'hail' in condition or 'freezing' in condition:
        return 'Sleet/Hail'
    elif 'fog' in condition or 'smog' in condition or 'smoke' in condition:
        return 'Fog/Smog/Smoke'
    elif 'wind' in condition or 'crosswind' in condition:
        return 'Severe Winds'
    elif 'blowing' in condition:
        return 'Blowing Sand/Snow'
    elif 'unknown' in condition or 'not reported' in condition:
        return 'Unknown'
    elif 'other' in condition or 'invalid' in condition:
        return 'Other'
    else:
        return 'Other'

data['Weather Conditions'] = data['Weather Conditions'].apply(simplify_weather)

# Simplify 'Light Conditions'
def simplify_light_conditions(condition):
    if pd.isnull(condition) or condition.strip() == '':
        return 'Unknown'
    condition = condition.lower()
    if 'daylight' in condition:
        return 'Daylight'
    elif 'dark' in condition:
        return 'Darkness'
    elif 'dawn' in condition or 'dusk' in condition:
        return 'Twilight'
    else:
        return 'Unknown'

data['Light Conditions'] = data['Light Conditions'].apply(simplify_light_conditions)

# Simplify 'Road Surface Condition'
def simplify_road_surface(condition):
    if pd.isnull(condition) or condition.strip() == '':
        return 'Unknown'
    condition = condition.lower()
    if 'dry' in condition:
        return 'Dry'
    elif 'wet' in condition or 'water' in condition:
        return 'Wet'
    elif 'snow' in condition:
        return 'Snow'
    elif 'ice' in condition:
        return 'Ice'
    elif 'slush' in condition:
        return 'Slush'
    elif 'sand' in condition or 'mud' in condition or 'dirt' in condition or 'oil' in condition or 'gravel' in condition:
        return 'Sand/Mud/Dirt/Oil/Gravel'
    elif 'unknown' in condition or 'not reported' in condition:
        return 'Unknown'
    else:
        return 'Other'

data['Road Surface Condition'] = data['Road Surface Condition'].apply(simplify_road_surface)

# Process driver age columns
def age_midpoint(age_range):
    if pd.isnull(age_range):
        return np.nan
    age_range = age_range.strip().replace('+', '').replace('>', '').replace('<', '').lower()
    if age_range == 'unknown':
        return np.nan
    parts = age_range.split('-')
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        else:
            return np.nan
    except ValueError:
        return np.nan

data['Youngest Driver Age'] = data['Age of Driver - Youngest Known'].apply(age_midpoint)
data['Oldest Driver Age'] = data['Age of Driver - Oldest Known'].apply(age_midpoint)

# Visualizations
sns.set_theme(style="whitegrid")

# Crash Severity Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=data, y='Crash Severity', color='lightcoral', order=data['Crash Severity'].value_counts().index)
plt.title('Distribution of Crash Severity')
plt.xlabel('Number of Crashes')
plt.ylabel('Crash Severity')
plt.tight_layout()
plt.savefig('./crash_severity_distribution.png')
plt.show()
plt.close()

# Speed Limit vs. Crash Severity
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Crash Severity', y='Speed Limit', palette='Set2')
plt.title('Speed Limit by Crash Severity')
plt.xlabel('Crash Severity')
plt.ylabel('Speed Limit (mph)')
plt.tight_layout()
plt.savefig('./speed_limit_by_crash_severity.png')
plt.show()
plt.close()

# Driver Age Distribution
young_age_df = data[['Youngest Driver Age', 'Crash Severity']].rename(columns={'Youngest Driver Age': 'Driver Age'})
old_age_df = data[['Oldest Driver Age', 'Crash Severity']].rename(columns={'Oldest Driver Age': 'Driver Age'})
combined_df = pd.concat([young_age_df, old_age_df]).dropna(subset=['Driver Age', 'Crash Severity'])
plt.figure(figsize=(10, 6))
sns.histplot(data=combined_df, x='Driver Age', hue='Crash Severity', bins=range(0, 100, 5), stat="density", alpha=0.6, multiple="dodge")
plt.title('Driver Age Distribution by Crash Severity')
plt.xlabel('Age')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('./driver_age_distribution_by_crash_severity.png')
plt.show()
plt.close()

# Proportion of Crash Severity by Weather Conditions
severity_counts = data.groupby(['Weather Conditions', 'Crash Severity']).size().reset_index(name='Count')
severity_counts['Proportion'] = severity_counts.groupby('Weather Conditions')['Count'].transform(lambda x: x / x.sum())
filtered_severity_counts = severity_counts[~severity_counts['Weather Conditions'].str.lower().isin(['unknown', 'other'])]

plt.figure(figsize=(12, 8))
sns.barplot(x='Weather Conditions', y='Proportion', hue='Crash Severity', data=filtered_severity_counts)
plt.title('Proportion of Crash Severity by Weather Conditions (Excluding Unknown and Other)')
plt.ylabel('Proportion')
plt.xlabel('Weather Conditions')
plt.xticks(rotation=45)
plt.legend(title='Crash Severity')
plt.tight_layout()
plt.savefig('./proportion_of_crash_severity_by_weather_conditions.png')
plt.show()
plt.close()

# Define categorical and numerical features
categorical_features = [
    'Weather Conditions', 'Light Conditions', 'Road Surface Condition',
    'Age of Driver - Youngest Known', 'Age of Driver - Oldest Known',
    'Driver Contributing Circumstances (All Drivers)',
    'Driver Distracted By (All Vehicles)', 'First Harmful Event',
    'Manner of Collision', 'Roadway Junction Type',
    'Traffic Control Device Type', 'Trafficway Description',
    'Vehicle Actions Prior to Crash (All Vehicles)',
    'Vehicle Configuration (All Vehicles)', 'Vehicle Emergency Use (All Vehicles)',
    'First Harmful Event Location', 'City Town Name'
]
numerical_features = ['Crash Hour Numeric', 'Number of Vehicles', 'Speed Limit']

# Fill missing values
for col in categorical_features:
    data[col] = data[col].fillna('Unknown')
for col in numerical_features:
    data[col] = data[col].fillna(data[col].median())

# Encode categorical features for MLP
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Scale numerical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine transformers
preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features),
    ('num', numerical_transformer, numerical_features)
])

# Encode target variable
y = data['Crash Severity']
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)

# Split data into train and test sets
X = data[categorical_features + numerical_features]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Original models evaluation
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=10),
    'XGBoost': XGBClassifier(eval_metric='mlogloss')
}

for model_name, classifier in models.items():
    print(f"\nTraining the {model_name} model...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_enc.classes_)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"{model_name} Model Accuracy: {accuracy}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", conf_matrix)

    # Visualizing feature importance
    print(f"Visualizing and saving feature importance for {model_name}...")
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    if model_name == 'Logistic Regression':
        coefficients = pipeline.named_steps['classifier'].coef_[0]
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coefficients
        })
    else:
        importances = pipeline.named_steps['classifier'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
    importance_df['Abs_Importance'] = importance_df['Importance'].abs()
    importance_df = importance_df.sort_values(by='Abs_Importance', ascending=False)
    plt.figure(figsize=(15, 10))
    plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title(f'Top 20 Feature Importances ({model_name})')
    plt.tight_layout()
    plt.savefig(f"{model_name}_feature_importance.png")
    plt.show()

# Transform data for MLP
X_train_transformed = preprocessor.fit_transform(X_train).toarray()
X_test_transformed = preprocessor.transform(X_test).toarray()

# Define MLP model
def build_mlp_model(input_dim, output_dim, dropout_rate=0.3, l2_reg=0.01):
    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(output_dim, activation='softmax')  # Softmax for multi-class classification
    ])
    model.compile(
        optimizer=SGD(learning_rate=0.001, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Build and train MLP model
input_dim = X_train_transformed.shape[1]
output_dim = len(np.unique(y_train))
mlp_model = build_mlp_model(input_dim, output_dim)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-5)

history = mlp_model.fit(
    X_train_transformed, y_train,
    validation_data=(X_test_transformed, y_test),
    epochs=30, batch_size=32,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Evaluate the MLP model
y_pred_mlp = np.argmax(mlp_model.predict(X_test_transformed), axis=1)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
report_mlp = classification_report(y_test, y_pred_mlp, target_names=label_enc.classes_)
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)

print(f"MLP Model Accuracy: {accuracy_mlp}")
print("\nClassification Report:\n", report_mlp)
print("\nConfusion Matrix:\n", conf_matrix_mlp)

# Confusion matrix heatmap for MLP
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_mlp, annot=True, fmt='d', cmap='Blues', xticklabels=label_enc.classes_, yticklabels=label_enc.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('MLP Confusion Matrix')
plt.tight_layout()
plt.savefig("MLP_confusion_matrix.png")
plt.show()

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('MLP Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('MLP Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# Encode categorical features directly for TabNet
label_encoders = {}
for col in categorical_features:
    label_enc = LabelEncoder()
    data[col] = label_enc.fit_transform(data[col])
    label_encoders[col] = label_enc

# Scale numerical features
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Encode target variable
y = data['Crash Severity']
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)

# Prepare features
X = data[categorical_features + numerical_features]
X_array = X.values
y_array = y_encoded

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, random_state=42)
# Initialize TabNet model
tabnet_model = TabNetClassifier(
    n_d=64, n_a=64, n_steps=5,
    gamma=1.5, n_independent=2, n_shared=2,
    lambda_sparse=1e-3, optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax',  # Use "sparsemax" or "entmax"
    scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9),
    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
    verbose=1  # Disable internal verbose to avoid clutter
)

# Train the model
max_epochs = 50
patience = 10

tabnet_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_name=['train', 'test'],
    eval_metric=['accuracy'],
    max_epochs=max_epochs,
    patience=patience,
    batch_size=256,
    virtual_batch_size=128,
    num_workers=0
)

# Evaluate the model
y_pred = tabnet_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_enc.classes_)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Test Accuracy: {accuracy}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)

# Feature importance visualization
tabnet_feature_importances = tabnet_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': tabnet_feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(15, 10))
plt.barh(
    y=importance_df['Feature'][:20],
    width=importance_df['Importance'][:20],
    align='center'
)
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances (TabNet)')
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("TabNet_feature_importance.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_enc.classes_, yticklabels=label_enc.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.savefig("Confusion_Matrix_Heatmap.png", bbox_inches='tight')
plt.show()

