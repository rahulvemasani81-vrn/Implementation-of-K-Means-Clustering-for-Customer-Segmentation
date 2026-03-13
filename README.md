# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries
2.Import data from the csv file
3.Choose the Number of Clusters
4.Assign each data point to the nearest centroid
5.Recalculate the centroids
6.Repeat steps 3 and 4 until centroids become stable
7.Plot the data

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: RAHUL
RegisterNumber:  25003095
*/
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("Mall_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
print(data.head())
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)


data['Cluster'] = y_kmeans

print("\nClustered Data:")
print(data.head())


plt.figure()
plt.scatter(X[y_kmeans == 0]['Annual Income (k$)'], 
            X[y_kmeans == 0]['Spending Score (1-100)'], label='Cluster 0')

plt.scatter(X[y_kmeans == 1]['Annual Income (k$)'], 
            X[y_kmeans == 1]['Spending Score (1-100)'], label='Cluster 1')

plt.scatter(X[y_kmeans == 2]['Annual Income (k$)'], 
            X[y_kmeans == 2]['Spending Score (1-100)'], label='Cluster 2')

plt.scatter(X[y_kmeans == 3]['Annual Income (k$)'], 
            X[y_kmeans == 3]['Spending Score (1-100)'], label='Cluster 3')

plt.scatter(X[y_kmeans == 4]['Annual Income (k$)'], 
            X[y_kmeans == 4]['Spending Score (1-100)'], label='Cluster 4')

plt.scatter(kmeans.cluster_centers_[:,0], 
            kmeans.cluster_centers_[:,1], 
            s=200, label='Centroids')

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

## Output:
<img width="701" height="126" alt="560683199-a95cc59f-c7c6-42de-a991-939e3c9f4ce6" src="https://github.com/user-attachments/assets/94c0205c-ae41-468c-8e07-dad9fe9e84c2" />
<img width="1148" height="730" alt="image" src="https://github.com/user-attachments/assets/87ac6cb6-2b25-4e33-8e56-e6a6016851f4" />

<img width="808" height="607" alt="image" src="https://github.com/user-attachments/assets/4f29ed03-4087-4988-9559-abefbb16a3e5" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
