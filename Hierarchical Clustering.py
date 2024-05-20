import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler, OneHotEncoder


data = pd.read_csv('customer_data.csv')
data = data.drop(columns=['id'])

# Define categorical features
categorical_features = ['gender', 'education', 'region', 'loyalty_status',
                        'purchase_frequency', 'product_category']

# Perform one-hot encoding for categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(data[categorical_features])

# Extract numerical features
numeric_features = ['age', 'income', 'purchase_amount', 'promotion_usage', 'satisfaction_score']

numeric_data = data[numeric_features]

# Preprocessing: Standardize numerical features
scaler = StandardScaler()
numeric_data = scaler.fit_transform(numeric_data)

# Concatenate encoded features with standardized numerical features
X = pd.concat([pd.DataFrame(encoded_features), pd.DataFrame(numeric_data)], axis=1)

# Perform hierarchical clustering
Z = linkage(X, method='ward')

# Predefined number of clusters
num_clusters = 3

# Perform hierarchical clustering with the predefined number of clusters
clusters = fcluster(Z, num_clusters, criterion='maxclust')

# Add cluster labels to the dataset
data['Cluster'] = clusters

# Create a subplot with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Visualize the dendrogram
dendrogram(Z, ax=ax1)
ax1.set_title('Dendrogram')
ax1.set_xlabel('Samples')
ax1.set_ylabel('Distance')

# Visualize the cluster distribution with bar graph
cluster_counts = data['Cluster'].value_counts()
ax2.bar(cluster_counts.index, cluster_counts.values, color='skyblue')
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Number of Data Points')
ax2.set_title('Cluster Distribution')

# Display the plot
plt.show()

# Display cluster distribution count
print("Cluster Distribution Count:")
print(cluster_counts)
