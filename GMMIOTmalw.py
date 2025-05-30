import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
file_path = 'IOTmalw.csv'
data = pd.read_csv(file_path)

# Encode 'id.resp_h' using LabelEncoder
label_encoder = LabelEncoder()
data['id.resp_h'] = label_encoder.fit_transform(data['id.resp_h'])

# Select features and target
features = ['id.resp_h', 'proto_encoded']
X = data[features]
y = data['Target_encoded']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_scaled)

# Predict on the entire dataset
y_pred = gmm.predict(X_scaled)

# Calculate confusion matrix and accuracy
conf_matrix = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')

# Print results
print("GMM Confusion Matrix:")
print(conf_matrix)
print(f"GMM Accuracy: {accuracy * 100:.2f}%")
print(f"GMM Precision: {precision * 100:.2f}%")
print(f"GMM Recall: {recall * 100:.2f}%")
print(f"GMM F1-Score: {f1 * 100:.2f}%")
