from sklearn.preprocessing import LabelEncoder
from kmodes.kmodes import KModes

# Perform K-modes clustering
def perform_clustering(data, num_clusters):
    # Exclude 'id' column
    data = data.drop(columns=['id'])

    # Encode categorical features
    label_encoders = {}
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

    # K-modes clustering
    km = KModes(n_clusters=num_clusters, init='Cao', n_init=2, verbose=1)
    clusters = km.fit_predict(data)

    # Add cluster labels to the dataset
    data['Cluster'] = clusters

    # Inverse transform cluster labels to original categorical labels
    for col in categorical_cols:
        data[col] = label_encoders[col].inverse_transform(data[col])

    return data