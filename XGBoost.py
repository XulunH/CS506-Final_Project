import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
import matplotlib.pyplot as plt

# 加载数据
data_1 = pd.read_csv('export_11_2_2024_12_16_57.csv')
data_2 = pd.read_csv('export_11_2_2024_12_20_13.csv')
data = pd.concat([data_1, data_2], ignore_index=True)

# 数据清洗
# 填充缺失值或删除（视情况而定）
data.dropna(subset=['Crash Date', 'Crash Severity', 'Weather Conditions'], inplace=True)

# 提取时间特征
data['Crash Date'] = pd.to_datetime(data['Crash Date'])
data['Month'] = data['Crash Date'].dt.month
data['DayOfWeek'] = data['Crash Date'].dt.dayofweek

# 年龄特征组合
data['Age Range'] = data['Age of Driver - Youngest Known'] + '-' + data['Age of Driver - Oldest Known']

# 编码分类变量
label_enc = LabelEncoder()
data['Crash Severity'] = label_enc.fit_transform(data['Crash Severity'])
data['Weather Conditions'] = label_enc.fit_transform(data['Weather Conditions'])
data['Light Conditions'] = label_enc.fit_transform(data['Light Conditions'])
data['Road Surface Condition'] = label_enc.fit_transform(data['Road Surface Condition'])
data['Age Range'] = label_enc.fit_transform(data['Age Range'])

# 准备特征和标签
X = data[['Month', 'DayOfWeek', 'Weather Conditions', 'Light Conditions', 'Road Surface Condition', 'Age Range', 'Latitude', 'Longitude']]
y = data['Crash Severity']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(y.unique()),
    eval_metric='mlogloss',
)

# 超参数调优
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(xgb_model, param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

print("Fetching best model from Grid Search...")
best_model = grid_search.best_estimator_

print("Making predictions on the test set...")
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)

with open("model_evaluation.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(conf_matrix))

# 特征重要性可视化并保存图像
print("Visualizing and saving feature importance...")
plt.figure(figsize=(10, 8))
xgb.plot_importance(best_model, importance_type='weight')
plt.title('Feature Importance')
plt.savefig("feature_importance.png")
plt.show()