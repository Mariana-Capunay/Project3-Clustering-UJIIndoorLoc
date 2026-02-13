# Project 3: Unsupervised Learning – WiFi Signal Clustering  
## Indoor Positioning with UJIIndoorLoc Dataset

**Date:** February 2026  

---

## Group 3

- Mariana A. Capuñay Correa
- Grace K. Inga Quispe
- Yoselyn V. Miranda Chirinos
- Lisseth J. Rondan Cantaro


---

## Project Overview

In indoor environments where GPS signals are unreliable or unavailable (e.g., shopping malls, airports, office buildings), positioning systems rely on WiFi fingerprinting techniques based on RSSI (Received Signal Strength Indicator) values.

This project applies unsupervised learning techniques to high-dimensional WiFi signal data in order to:

- Discover inherent spatial clustering structures
- Compare dimensionality reduction techniques
- Evaluate clustering performance using internal and external metrics
- Analyze separability at two hierarchical levels: buildings and floors

Unlike supervised approaches, this project focuses on structure discovery without using labels during training, while still leveraging them for evaluation.

---

## Dataset

We use the UJIIndoorLoc dataset from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/310/ujiindoorloc  

The dataset contains:

- 520 WiFi Access Point (WAP) signal strength features  
- RSSI values ranging from -104 dBm to 0  
- Value 100 indicating "no signal detected"  
- BuildingID and Floor labels  
- Geographic coordinates (longitude and latitude)  

This dataset is widely used in indoor positioning research due to its high dimensionality and real-world noise.

---

## Methodology

### 1. Data Loading and Exploration

- Loaded training dataset
- Verified dimensionality and feature consistency
- Analyzed sample distribution per BuildingID
- Explored global RSSI signal distribution
  
### 2. Data Preprocessing
The preprocessing pipeline includes:

- Replaced RSSI value 100 with -110 dBm (baseline for no signal detection)
- Applied Min-Max normalization to scale signals into the [0,1] range
- Removed zero-variance WAP features using VarianceThreshold
- Applied StandardScaler before clustering to ensure standardized feature space

This ensures numerical stability and improves clustering performance.

### 3. Dimensionality Reduction

To analyze high-dimensional signal data, multiple dimensionality reduction techniques were applied:

**PCA (Principal Component Analysis)**
- PCA retaining 95% of explained variance (~267 components)
- PCA with 27 components (reduced representation)
- 2D PCA projection for visualization

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- 2D embedding for local structure visualization

**UMAP (Uniform Manifold Approximation and Projection)**
- 2D embedding for visualization
- 10D embedding for clustering comparison

These methods allow comparison between global variance preservation (PCA) and manifold-learning approaches (t-SNE, UMAP). 

### 4. Clustering Approach

K-Means clustering was applied using:

- Number of clusters equal to the number of true labels

  - For global clustering → number of buildings
  - For intra-building clustering → number of floors

Clustering was evaluated at two levels:

**Level 1 – Global Clustering**: Clustering across all buildings to evaluate building-level separability.
**Level 2 – Intra-Building Clustering**: For each building separately, clustering was applied to analyze floor-level separability.

### 5. Cluster Evaluation Metrics

Both internal and external validation metrics were used.

#### External Metrics (label-based evaluation)

- Rand Index
- Homogeneity Score
- Completeness Score
- V-Measure
- Fowlkes-Mallows Index

These metrics compare predicted cluster assignments against true BuildingID or Floor labels.

#### Internal Metrics (structure-based evaluation)

- Silhouette Score
- Davies-Bouldin Index

These metrics evaluate compactness and separation without using ground-truth labels.

Metrics were computed:
- Without dimensionality reduction
- After PCA (27 components)
- After PCA (95% variance)
- After UMAP (10D)

This allows rigorous comparison of how dimensionality reduction affects clustering performance.

---

## Requirements

- Python 3.10+  
- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  
- umap-learn  

---

## Setup

Clone the repository:

```
git clone https://github.com/your-username/CS3061-Project3-Clustering.git  
cd CS3061-Project3-Clustering  
```


Install dependencies:

```
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn  
```

---

## How to Run


If using Jupyter Notebook:

```
jupyter notebook  
```

Then run the notebook containing the full pipeline.

---

## Output

The project generates:
- Distribution plots per building
- Histogram of RSSI signal intensities
- PCA (2D) visualization
- t-SNE visualization
- UMAP visualization
- Comparative clustering evaluation tables
- Internal and external metric summaries
- Per-building floor clustering results

Results are displayed as visualizations and structured metric outputs.

---

## Key Observations

- Building-level clustering shows strong separability.
- Floor-level separability varies significantly across buildings.
- PCA preserves global variance but may reduce local structure clarity.
- t-SNE enhances local cluster visualization but is not ideal for metric-based clustering evaluation.
- UMAP provides a balance between global and local structure preservation.
- Dimensionality reduction can either improve or degrade clustering performance depending on the method and dimensionality retained.


---


## Conclusion

This study demonstrates that WiFi fingerprint data contains meaningful spatial structure that can be uncovered through unsupervised learning techniques.

The comparison between PCA, t-SNE, and UMAP reveals that dimensionality reduction significantly impacts clustering quality, particularly when evaluated using both internal and external validation metrics.

The hierarchical evaluation (building → floor) provides deeper insight into how indoor environments influence signal-based clustering behavior.

---

## Future Work

- Explore density-based clustering methods (DBSCAN, HDBSCAN)
- Apply spectral clustering techniques
- Perform hyperparameter optimization for UMAP and K-Means
- Investigate feature selection methods to reduce WAP dimensionality
- Compare results against supervised indoor localization models
