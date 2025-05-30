import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
file_path = 'IOTmalw.csv'
data = pd.read_csv(file_path)

# Encode 'id.resp_h' and 'id.orig_h' using LabelEncoder
label_encoder = LabelEncoder()
data['id.resp_h'] = label_encoder.fit_transform(data['id.resp_h'])
data['id.orig_h'] = label_encoder.fit_transform(data['id.orig_h'])

# Select all features and target
features = ['id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'orig_pkts', 'resp_pkts', 
            'proto_encoded', 'conn_state_encoded', 'history_encoded']
X = data[features]
y = data['Target_encoded']

# Apply KMeans with 2 clusters directly on the unscaled features
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, random_state=42)
kmeans.fit(X)

# Predict on the entire dataset
y_pred = kmeans.predict(X)

# Calculate confusion matrix and accuracy
conf_matrix = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)

# Calculate precision, recall, and F1-score
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Print results
print("KMeans Confusion Matrix:")
print(conf_matrix)
print(f"KMeans Accuracy: {accuracy * 100:.2f}%")
print(f"KMeans Precision: {precision:.2f}")
print(f"KMeans Recall: {recall:.2f}")
print(f"KMeans F1-Score: {f1:.2f}")
