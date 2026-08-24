import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset (Mall Customers Dataset via direct URL)
url = "https://raw.githubusercontent.com/vjchoudhary7/Customer-Segmentation-Tutorial-in-Python/master/Mall_Customers.csv"
df = pd.read_csv(url)

# 2. Feature Selection
# Using 'Annual Income (k$)' and 'Spending Score (1-100)' for segmentation
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# 3. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply K-Means Clustering (Optimal k = 5)
k = 5
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 5. Display Cluster Summary
print("--- Cluster Centers (Original Scale) ---")
cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
for i, center in enumerate(cluster_centers_original):
    print(f"Cluster {i}: Annual Income = ${center[0]:.2f}k, Spending Score = {center[1]:.2f}")

print("\n--- Customer Count per Cluster ---")
print(df['Cluster'].value_counts().sort_index())
