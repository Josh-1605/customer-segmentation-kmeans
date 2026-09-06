import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# 1. Load dataset
url = "https://raw.githubusercontent.com/sharmaroshan/Clustering-of-Mall-Customers/master/Mall_Customers.csv"

df = pd.read_csv(url)

# 2. Display basic information
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# 3. Select features for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 4. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Find optimal K using Elbow Method
inertia = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

print("\nInertia Values:")
for k, value in zip(range(2, 11), inertia):
    print(f"K = {k}: {value:.2f}")

# Elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# 6. Apply K-Means with K = 5
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 7. Calculate Silhouette Score
silhouette = silhouette_score(X_scaled, df["Cluster"])

print(f"\nSilhouette Score: {silhouette:.4f}")

# 8. Display number of customers in each cluster
print("\nCustomers in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())

# 9. Display cluster centers
centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)

print("\nCluster Centers:")
for i, center in enumerate(centers):
    print(
        f"Cluster {i}: "
        f"Annual Income = {center[0]:.2f} k$, "
        f"Spending Score = {center[1]:.2f}"
    )

# 10. Visualize K-Means clusters
plt.figure(figsize=(9, 6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    s=60
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=250
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)
plt.show()

# 11. Hierarchical Clustering - Dendrogram
plt.figure(figsize=(10, 6))

linked = linkage(X_scaled, method="ward")

dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

# 12. Apply Hierarchical Clustering
hierarchical = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)

df["Hierarchical_Cluster"] = hierarchical.fit_predict(X_scaled)

# 13. Evaluate Hierarchical Clustering
hierarchical_score = silhouette_score(
    X_scaled,
    df["Hierarchical_Cluster"]
)

print(f"\nHierarchical Clustering Silhouette Score: "
      f"{hierarchical_score:.4f}")

print("\nHierarchical Cluster Counts:")
print(df["Hierarchical_Cluster"].value_counts().sort_index())
# 14. PCA Dimensionality Reduction
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print(f"Total Variance Explained: "
      f"{pca.explained_variance_ratio_.sum():.4f}")

# PCA Cluster Visualization
plt.figure(figsize=(9, 6))

plt.scatter(
    df["PCA1"],
    df["PCA2"],
    c=df["Cluster"],
    s=60
)

plt.title("Customer Clusters using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()
# 15. Save final dataset
df.to_csv("customer_segments.csv", index=False)

print("\nFinal clustered dataset saved as customer_segments.csv")
print("\nProject completed successfully!")