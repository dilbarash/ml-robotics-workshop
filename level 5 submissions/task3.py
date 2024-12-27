import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import LabelEncoder

classification_model = joblib.load('sensor_classification_model.pkl')
classification_data = pd.read_csv('robot_sensor_data.csv')

label_encoder = LabelEncoder()
classification_data['label'] = label_encoder.fit_transform(classification_data['label'])

X_class = classification_data.drop('label', axis=1)
y_class = classification_data['label']

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

y_pred_class = classification_model.predict(X_test_class)

accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class, average='weighted')
recall = recall_score(y_test_class, y_pred_class, average='weighted')

print(f"Classification Model - Accuracy: {accuracy}")
print(f"Classification Model - Precision: {precision}")
print(f"Classification Model - Recall: {recall}")

conf_matrix = confusion_matrix(y_test_class, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Classification)')
plt.show()
