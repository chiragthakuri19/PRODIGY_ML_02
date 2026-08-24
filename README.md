# PRODIGY_ML_02: Customer Segmentation using K-Means

This repository contains an unsupervised machine learning model that clusters retail store customers based on their annual income and spending score.

## Dataset
- **Mall Customer Segmentation Dataset**: Contains customer demographics and purchase behavioral metrics.

## Features Used
- `Annual Income (k$)`: Annual income of the customer in thousands of dollars.
- `Spending Score (1-100)`: Score assigned based on customer behavior and purchase data.

## Methodology
1. **Data Preprocessing**: Standardized numeric features using `StandardScaler`.
2. **Clustering Algorithm**: Applied **K-Means Clustering** ($k = 5$) to segment customers into distinct behavioral groups.
3. **Evaluation & Profiling**: Extracted cluster centroids to profile target customer groups for personalized marketing strategies.
