import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA

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
    dataframe1 = np.transpose(pd.read_csv(cestaKdatam))
    return dataframe1


def remove_low_variance_features(data, threshold):
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
    pca = PCA(n_components=num_components)
    transformed_data = pca.fit_transform(data)
    return transformed_data

def compute_column_correlations(data):
    correlations = np.corrcoef(data, rowvar=False)
    return correlations

def remove_non_numeric_values(data):
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

data = np.array(eval(Data[3]))
data = data[:, :, 0]  # Convert to numeric format
kapa = remove_low_variance_features(data, 0.001)

pca_data = perform_pca(kapa, num_components=2)
preprocessed_data = remove_non_numeric_values(kapa)
correlations = compute_column_correlations(preprocessed_data.astype(float))
print(data)