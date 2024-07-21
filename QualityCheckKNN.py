import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from KModesCluster import perform_clustering

# Load the dataset
data = pd.read_csv('customer_data.csv')

# Specify the number of clusters
num_clusters = 3  # Adjust this as needed

# Perform clustering
clustered_data = perform_clustering(data, num_clusters)

# Train KNN classifier using encoded data
X = clustered_data.drop(columns=['Cluster'])
y = clustered_data['Cluster']

# Encode categorical columns for KNN
label_encoders_knn = {}
categorical_cols_knn = X.select_dtypes(include=['object']).columns
for col in categorical_cols_knn:
    label_encoders_knn[col] = LabelEncoder()
    X[col] = label_encoders_knn[col].fit_transform(X[col])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Predict cluster labels on the testing set
y_pred = knn_classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy=", accuracy)