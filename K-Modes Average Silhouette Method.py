import pandas as pd
from sklearn.preprocessing import LabelEncoder
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("customer_data.csv")
data = data.drop(columns=['id'])

# Preprocessing of categorical features
label_encoder = LabelEncoder()
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Range of clusters to check
min_clusters = 2
max_clusters = 10
range_clusters = range(min_clusters, max_clusters + 1)

# Computing silhouette scores for number of clusters
silhouette_scores = []
for num_clusters in range_clusters:
    km = KModes(n_clusters=num_clusters, init='Cao', n_init=2, verbose=1)
    clusters = km.fit_predict(data)
    silhouette_scores.append(silhouette_score(data, clusters))

# Console output of optimal num of clusters
optimal_clusters = range_clusters[np.argmax(silhouette_scores)]
print("Optimal number of clusters:", optimal_clusters)

# Display graph
plt.plot(range_clusters, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Silhouette Score')
plt.title('Average Silhouette Method for Optimal Number of Clusters')
plt.show()



