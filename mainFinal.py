import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import umap.umap_ as umap

# Specify the paths to the data directories
Cesty = [
    'D:/Brno_clustering/Vox_based/spolu_nativ/',
    'D:/Brno_clustering/Vox_based/spolu_faza1/',
    'D:/Brno_clustering/Vox_based/spolu_faza2/',
    'D:/Brno_clustering/Vox_based/spolu_faza3/',
    'D:/Brno_clustering/Vox_based/spolu_tMIP/',
]

Data = [
    'Data_Vox_based_nativ',
    'Data_Vox_based_faza1',
    'Data_Vox_based_faza2',
    'Data_Vox_based_faza3',
    'Data_Vox_based_tMIP',
]

# Function to drop highly correlated features from the data
def drop_highly_correlated_features(data, threshold):
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # Drop highly correlated features
    data.drop(to_drop, axis=1, inplace=True)
    return data

# Function to perform PCA on the data
def perform_pca(data, num_components):
    pca = PCA(n_components=num_components)
    transformed_data = pca.fit_transform(data)
    # Calculate the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    # Calculate the cumulative explained variance ratio
    explained_variance_ratio_cumulative = np.cumsum(explained_variance_ratio)
    return transformed_data, explained_variance_ratio_cumulative

# Function to perform t-SNE on the data
def perform_tsne(data, num_components):
    perplexity = 50  # Set perplexity to a value lower than the number of samples
    tsne = TSNE(n_components=num_components, perplexity=perplexity)
    transformed_data = tsne.fit_transform(data)
    return transformed_data

# Function to read label data from a CSV file
def get_label(cestaKdatam):
    # Read and transpose the label data from a CSV file
    dataframe1 = np.transpose(pd.read_csv(cestaKdatam))
    return dataframe1

# Initialize empty lists for storing data
Data_Vox_based_nativ = []
Data_Vox_based_faza1 = []
Data_Vox_based_faza2 = []
Data_Vox_based_faza3 = []
Data_Vox_based_tMIP = []

IDs = []

# Iterate over each directory and file to read the data
for poradie in range(5):
    for subdir, dirs, files in os.walk(Cesty[poradie]):
        for file in files:
            typsuboru = os.path.splitext(subdir + os.sep + file)[1]
            if typsuboru == ".csv":
                base = os.path.splitext(subdir + os.sep + file)[0]
                cestaKlabelu = base + ".csv"
                if os.path.isfile(cestaKlabelu):
                    print(cestaKlabelu)
                    IDs.append(base)
                    OneDatoLabel = get_label(cestaKlabelu)
                    eval(Data[poradie]).append(OneDatoLabel)

data = eval(Data[3])
df = pd.DataFrame()
for i in range(len(data)):
    obj_df = pd.DataFrame(data[i])
    obj_df.columns = [f'Attribute_{i+1}_{j+1}' for j in range(obj_df.shape[1])]
    df = pd.concat([df, obj_df], axis=1)

# Remove NaN values from the DataFrame
df = pd.DataFrame(df).T
df = df.dropna()

# Perform PCA
PCA_comp, explained_variance_ratio_cumulative = perform_pca(df, 5)

# Perform t-SNE on the data
tsne_data = perform_tsne(df, num_components=2)

# Perform UMAP on the data
umap_data = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric='euclidean', random_state=42).fit_transform(df)

droped_data=drop_highly_correlated_features(df, 0.7)


# Perform K-means clustering
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(PCA_comp)

# Create a color map with 10 distinct colors
color_map = plt.cm.get_cmap('tab10', 10)

# Get the unique colors for each item
num_items = len(data)
item_colors = [color_map(i % 10) for i in range(num_items)]

# Get the cluster labels for each item
cluster_labels = clusters[:num_items]

# Create a color map with unique background colors based on the cluster labels
background_colors = [color_map(i % 10) for i in cluster_labels]

# Plot the t-SNE-transformed data with colors based on the item and background colors based on the cluster labels
plt.figure(figsize=(8, 6))
for i, data_file in enumerate(data):
    ID = IDs[i]
    num_points = len(data_file)
    item_color = item_colors[i]
    background_color = background_colors[i]
    tsne_data_file = tsne_data[i: i + num_points]
    plt.scatter(tsne_data_file[:, 0], tsne_data_file[:, 1], c=item_color, label=f'File {i+1}', edgecolor=background_color)

plt.title('t-SNE Transformed Data with K-means Clustering')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()

# Create a color map with 5 distinct colors based on the cluster labels
cluster_colors = [color_map(i % 10) for i in clusters]

# Plot the t-SNE-transformed data with colors based on the cluster labels
plt.figure(figsize=(8, 6))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=cluster_colors)
plt.title('t-SNE Transformed Data with K-means Clustering')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

# Plot the PCA-transformed data with colors based on the item and background colors based on the cluster labels
plt.figure(figsize=(8, 6))
for i, data_file in enumerate(data):
    ID = IDs[i]
    num_points = len(data_file)
    item_color = item_colors[i]
    background_color = background_colors[i]
    PCA_comp_file = PCA_comp[i: i + num_points]
    plt.scatter(PCA_comp_file[:, 0], PCA_comp_file[:, 1], c=item_color, label=f'File {i+1}', edgecolor=background_color)

plt.title('PCA-Transformed Data with K-means Clustering')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.show()

# Plot the PCA-transformed data with colors based on the cluster labels
plt.figure(figsize=(8, 6))
plt.scatter(PCA_comp[:, 0], PCA_comp[:, 1], c=cluster_colors)
plt.title('PCA-Transformed Data with K-means Clustering')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

# Plot the UMAP-transformed data with colors based on the item and background colors based on the cluster labels
plt.figure(figsize=(8, 6))
for i, data_file in enumerate(data):
    ID = IDs[i]
    num_points = len(data_file)
    item_color = item_colors[i]
    background_color = background_colors[i]
    umap_data_file = umap_data[i: i + num_points]
    plt.scatter(umap_data_file[:, 0], umap_data_file[:, 1], c=item_color, label=f'File {i+1}', edgecolor=background_color)

plt.title('UMAP-Transformed Data with K-means Clustering')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend()
plt.show()

# Plot the UMAP-transformed data with colors based on the cluster labels
plt.figure(figsize=(8, 6))
plt.scatter(umap_data[:, 0], umap_data[:, 1], c=cluster_colors)
plt.title('UMAP-Transformed Data with K-means Clustering')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()

# Plot the raw data with colors based on the cluster labels
plt.figure(figsize=(8, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=cluster_colors)
plt.title('Raw Data with K-means Clustering (After Feature Selection)')
plt.xlabel('Attribute 1')
plt.ylabel('Attribute 2')
plt.show()
