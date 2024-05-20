import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from KModesCluster import perform_clustering

# Load the dataset
data = pd.read_csv('customer_data.csv')

# Specify the number of clusters
num_clusters = 2  # Adjust this as needed

# Perform clustering
clustered_data = perform_clustering(data, num_clusters)

# Train Random Forest classifier using encoded data
X = clustered_data.drop(columns=['Cluster'])
y = clustered_data['Cluster']

# Encode categorical columns for Random Forest
label_encoders_rf = {}
categorical_cols_rf = X.select_dtypes(include=['object']).columns
for col in categorical_cols_rf:
    label_encoders_rf[col] = LabelEncoder()
    X[col] = label_encoders_rf[col].fit_transform(X[col])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict cluster labels on the testing set
y_pred = rf_classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)