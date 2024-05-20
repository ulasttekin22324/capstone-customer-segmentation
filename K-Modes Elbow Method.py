import pandas as pd
from sklearn.preprocessing import LabelEncoder
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("customer_data.csv")
data = data.drop(columns=['id'])

# Preprocessing of categorical features
label_encoder = LabelEncoder()
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Range of clusters to check
min_clusters = 2
max_clusters = 10
range_clusters = range(min_clusters, max_clusters + 1)

# Computing cost for number of clusters
costs = []
for num_clusters in range_clusters:
    km = KModes(n_clusters=num_clusters, init='Cao', n_init=2, verbose=1)
    km.fit(data)
    costs.append(km.cost_)

# Display graph
plt.plot(range_clusters, costs, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Cost (Inertia)')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()











