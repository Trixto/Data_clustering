import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Specify the paths to the data directories
Cesty = [
    'D:/Brno_clustering/Spolu_radiologovia/spolu_nativ/',
    'D:/Brno_clustering/Spolu_radiologovia/spolu_faza1/',
    'D:/Brno_clustering/Spolu_radiologovia/spolu_faza2/',
    'D:/Brno_clustering/Spolu_radiologovia/spolu_faza3/',
    'D:/Brno_clustering/Spolu_radiologovia/spolu_tMIP/',
    'D:/Brno_clustering/Spolu_vsetci/spolu_nativ/',
    'D:/Brno_clustering/Spolu_vsetci/spolu_faza1/',
    'D:/Brno_clustering/Spolu_vsetci/spolu_faza2/',
    'D:/Brno_clustering/Spolu_vsetci/spolu_faza3/',
    'D:/Brno_clustering/Spolu_vsetci/spolu_tMIP/'
]

# Specify the names of the data files
Data = [
    'Data_radiologovia_nativ_nativ',
    'Data_radiologovia_nativ_faza1',
    'Data_radiologovia_nativ_faza2',
    'Data_radiologovia_nativ_faza3',
    'Data_radiologovia_nativ_tMIP',
    'Data_vsetci_nativ',
    'Data_vsetci_faza1',
    'Data_vsetci_faza2',
    'Data_vsetci_faza3',
    'Data_vsetci_tMIP'
]

# Initialize empty lists to store the data
Data_radiologovia_nativ_nativ = []
Data_radiologovia_nativ_faza1 = []
Data_radiologovia_nativ_faza2 = []
Data_radiologovia_nativ_faza3 = []
Data_radiologovia_nativ_tMIP = []
Data_vsetci_nativ = []
Data_vsetci_faza1 = []
Data_vsetci_faza2 = []
Data_vsetci_faza3 = []
Data_vsetci_tMIP = []

def getLabel(cestaKdatam):
    # Read and transpose the label data from a CSV file
    dataframe1 = np.transpose(pd.read_csv(cestaKdatam))
    return dataframe1

def remove_low_variance_features(data, threshold):
    # Remove features with low variance from the data
    data = np.asarray(data)
    num_features = data.shape[1]
    selected_features = []
    
    for feature_idx in range(num_features):
        feature = data[:, feature_idx]
        
        # Calculate variance based on the data type
        if np.issubdtype(feature.dtype, np.number):
            variance = np.var(feature)
        elif np.issubdtype(feature.dtype, np.unicode_) or np.issubdtype(feature.dtype, np.object_):
            unique_values, counts = np.unique(feature, return_counts=True)
            variance = np.var(counts)
        else:
            variance = 0
        
        if variance >= threshold:
            selected_features.append(feature_idx)
    
    filtered_data = data[:, selected_features]
    return filtered_data

def perform_pca(data, num_components):
    # Perform PCA on the data
    pca = PCA(n_components=num_components)
    transformed_data = pca.fit_transform(data)
    return transformed_data

def compute_column_correlations(data):
    # Compute the column-wise correlations of the data
    correlations = np.corrcoef(data, rowvar=False)
    return correlations

def remove_non_numeric_values(data):
    # Replace non-numeric values in the data with appropriate numeric representations
    data = np.asarray(data)
    num_rows, num_columns = data.shape
    filtered_data = np.zeros_like(data)
    
    for column_idx in range(num_columns):
        column = data[:, column_idx]
        
        # Check if the column contains non-numeric values
        if not np.issubdtype(column.dtype, np.number):
            unique_values, counts = np.unique(column, return_counts=True)
            non_numeric_values = unique_values[np.logical_not(np.issubdtype(unique_values.dtype, np.number))]
            
            # Replace non-numeric values with appropriate numeric representations
            for non_numeric_value in non_numeric_values:
                numeric_value = np.argmax(unique_values == non_numeric_value)
                column[column == non_numeric_value] = numeric_value
        
        filtered_data[:, column_idx] = column
    
    return filtered_data

def Get_Only_Heterogeneity_Features(feature_map):
    # Remove shape features from the feature map
    features_to_drop = [i for i in range(feature_map.shape[1]) if 'shape' in str(i)]
    filtered_feature_map = np.delete(feature_map, features_to_drop, axis=1)
    return filtered_feature_map.astype(float)

def perform_tsne(data, num_components):
    # Perform t-SNE on the data
    perplexity = min(5, data.shape[0] - 1)  # Set perplexity to a value lower than the number of samples
    tsne = TSNE(n_components=num_components, perplexity=perplexity)
    transformed_data = tsne.fit_transform(data)
    return transformed_data

def perform_kmeans(data, num_clusters):
    # Perform k-means clustering on the data
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(data)
    return labels


# Iterate over each directory and file to read the data
for poradie in range(10):
    for subdir, dirs, files in os.walk(Cesty[poradie]):
        for file in files:
            typsuboru = os.path.splitext(subdir + os.sep + file)[1]
            if typsuboru == ".csv":
                base = os.path.splitext(subdir + os.sep + file)[0]
                cestaKlabelu = base + ".csv"
                if os.path.isfile(cestaKlabelu):
                    print(cestaKlabelu)
                    OneDatoLabel = getLabel(cestaKlabelu)
                    eval(Data[poradie]).append(OneDatoLabel)

    np.save(Cesty[poradie] + ".npy", eval(Data[poradie]))

# Perform analysis on Data_radiologovia_nativ_faza3
data = np.array(eval(Data[3]))
data = data[:, :, 0]

# Remove low variance features
kapa = Get_Only_Heterogeneity_Features(remove_low_variance_features(data, 0.01))

# Perform PCA
pca_data = perform_pca(kapa, num_components=2)

# Remove non-numeric values
preprocessed_data = remove_non_numeric_values(kapa)

# Compute column correlations
correlations = compute_column_correlations(preprocessed_data.astype(float))

# Perform t-SNE
tsne_data = perform_tsne(kapa, num_components=3)

# Perform k-means clustering
num_clusters = 3  # Set the desired number of clusters
kmeans_labels = perform_kmeans(kapa, num_clusters)

# Plot the original data
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1])
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Plot the PCA-transformed data
plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1])
plt.title('PCA-Transformed Data')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlations, cmap='coolwarm', annot=True, fmt=".2f", square=True)
plt.title('Correlation Matrix')
plt.show()

# Plot the t-SNE-transformed data
plt.figure(figsize=(8, 6))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1])
plt.title('t-SNE Transformed Data')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

# Plot the k-means clustering results
plt.figure(figsize=(8, 6))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=kmeans_labels)
plt.title('k-means Clustering')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
