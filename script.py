# ---------------------------------------------------------------------------------------------------------------------
# Author: Eric Wang
# Data:
#  1. Financial statement data: https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data
# ---------------------------------------------------------------------------------------------------------------------
# The goal of this project is to identify possible groupings of firms based on their financial statement data. Since
# the dataset contains 64 financial figures, I use unsupervised learning methods to simplify the data and cluster it.
# At the end, I compare the cluster labels to whether the firms go bankrupt within a year of the data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import arff
from sklearn.manifold import TSNE
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.preprocessing import StandardScaler
from ppca import PPCA
from sklearn.pipeline import make_pipeline
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

DATAROOT = '/Users/ericwang/Documents/GitHub datasets/Firm segmentation/'
EXPORT = '/Users/ericwang/Documents/GitHub/Firm segmentation/'

# Import the data and label the columns according to the documentation
raw = arff.loadarff(DATAROOT + '5year.arff')
colnames = pd.read_excel(DATAROOT + 'column_names.xlsx', converters={'Name': str})['name'].str.strip().tolist()
main_df = pd.DataFrame(raw[0])
main_df.columns = colnames
main_df['bankrupt'] = np.where(main_df['bankrupt'] == b'1', True, False)

# Check if there are missing values: there are many
bankrupt_firms = main_df
print(bankrupt_firms.isnull().sum())

# Remove outlier firms to facilitate ML methods. In this context, outlier values can come from missing or inaccurate
# reporting or extremely poor performance. Use mean imputation on a filtering matrix to help with this process.
for_filtering = bankrupt_firms.drop('bankrupt', 1)
for_filtering = for_filtering.fillna(for_filtering.mean())
trimmed = bankrupt_firms[(np.abs(stats.zscore(for_filtering)) < 2).all(axis=1)]

# Create a preliminary t-SNE plot to visualize the variation in the data.
# It looks like there are possibly two or three clusters on the bottom-left and top-right. However, t-SNE is known
# to create clusters from data without inherent groups.
model = TSNE(learning_rate=200)
scaler = StandardScaler()
tsne = model.fit_transform(scaler.fit_transform(trimmed.fillna(trimmed.mean()).to_numpy()))
plt.scatter(tsne[:, 0], tsne[:, 1])
plt.savefig(EXPORT + 'tsne_plot.png')
plt.show()
plt.cla()

# Create a random sample of 50 firms that will go bankrupt within the year and 50 firms that will not
sample = trimmed.query('bankrupt == True').sample(n=50)
sample = sample.append(trimmed.query('bankrupt == False').sample(n=50))
actual_labels = sample['bankrupt']
sample = sample.drop('bankrupt', 1)

# Attempt NMF to preserve the interpretability of the variables. Count columns with negative values first.
col_count = 0
for (colname, colrow) in sample.iteritems():
    if (sample[colname] < 0).any():
        col_count += 1

# Since performing NMF would require dropping 39 columns, it would be unreasonable to proceed with NMF. There is a high
# risk that some of the dropped columns can explain variation. This makes sense. Negative financial ratios are often
# taken as a sign of poor performance.
print(col_count)

# Proceed with PCA to perform dimensionality reduction.
# Normalize and model the data using a probabilistic PCA function that is robust to missing values (not the well-known
# implementation in sci-kit learn)
sample = sample.to_numpy()
pca = PPCA()
pipeline = make_pipeline(scaler, pca)
pca.fit(scaler.fit_transform(sample))

# Plot the explained variation by number of principal components
pca_features = list(range(1, pca.var_exp.shape[0] + 1))
plt.bar(pca_features, pca.var_exp)
plt.xlabel('PCA feature')
plt.ylabel('Explained variation')
plt.xticks(pca_features)
plt.savefig(EXPORT + 'pca_variation_plot.png')
plt.show()
plt.cla()

num_components = min(np.where((pca.var_exp >= 0.95) == True)[0]) + 1

# 18 PCA features explain at least 95% of the variation among firms. Rerun PCA with that number of components
# and then transform the data.
pca = PPCA()
pca.fit(scaler.fit_transform(sample), d=num_components)
pca_components = pca.transform()
print(pca_components.shape)

# Run hierarchical clustering and plot the dendrogram
clustered = linkage(pca_components, method='complete')
dendrogram(clustered, no_labels=True)
plt.savefig(EXPORT + 'dendrogram_plot.png')
plt.show()
plt.cla()

# Cut the dendrogram at two groups. With more than two groups, there seem to be groups that contain outliers as well as
# diminishing gains in proximity. This choice of centroids is also consistent with the earlier t-SNE plot.
model_labels = fcluster(clustered, 30, criterion='distance')

# Cross tabulate the cluster labels with the labelled data (similar to a confusion matrix). As seen, there is a very 
# high probability (94.4%) of a firm in the second cluster going bankrupt within the next year.
actual_labels = np.where(actual_labels == True, 'Imminently Bankrupt', 'Will Persist')
summary = pd.crosstab(model_labels, actual_labels, rownames=['Cluster'], normalize='index').round(decimals=3)
print(summary)
summary.to_excel(EXPORT + 'model_summary.xlsx')
