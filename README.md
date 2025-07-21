MSCS_634_Lab_3: K-Means and K-Medoids Clustering Analysis


Purpose of Lab Work

This lab explores the performance characteristics of two centroid-based clustering algorithms:

1. K-Means Clustering - A partition-based algorithm that groups data by minimizing within-cluster sum of squares using centroids (mean points)
2. K-Medoids Clustering - A robust variant that uses actual data points (medoids) as cluster centers to minimize total dissimilarity

Objectives:
- Compare clustering performance using multiple evaluation metrics
- Analyze cluster formation patterns and algorithm behavior
- Understand when each algorithm is most effective
- Develop skills in unsupervised learning evaluation
- Gain practical experience with clustering visualization techniques
- Assess cluster quality without and with ground truth labels

Technical Implementation:
- Dataset: Wine Dataset (178 samples, 13 features, 3 classes)
- Preprocessing: Z-score standardization for consistent feature scaling
- Clustering Parameters: k = 3 (matching original wine classes)
- Evaluation Metrics: Silhouette Score, Adjusted Rand Index (ARI)
- Visualization: PCA dimensionality reduction for 2D plotting
- Custom Implementation: SimpleKMedoids class to avoid external dependencies

Key Insights and Observations

K-Means Performance Analysis:
- Silhouette Score: 0.2849 indicating moderately well-defined clusters
- Adjusted Rand Index: 0.8975 showing excellent alignment with true wine classes
- Cluster Distribution: Balanced sizes (65, 51, 62 samples per cluster)
- Inertia Value: 1277.93 representing total within-cluster sum of squares
- Algorithm Behavior: Successfully identified spherical clusters matching wine varieties

K-Medoids Performance Analysis:
- Silhouette Score: 0.1548 indicating less well-defined cluster boundaries
- Adjusted Rand Index: 0.3413 showing moderate alignment with true classes
- Cluster Distribution: Unbalanced sizes (89, 24, 65 samples per cluster)
- Inertia Value: 549.38 representing total dissimilarity from medoids
- Algorithm Behavior: More conservative clustering with actual data points as centers

Dataset Characteristics:
- High Separability: Wine classes are distinguishable in feature space (89.75% ARI for K-Means)
- Feature Diversity: 13 chemical properties with varying scales requiring standardization
- Class Balance: Relatively balanced original classes (59, 71, 48 samples)
- Dimensionality: PCA revealed 55.41% variance captured in first two components

Comparative Analysis:
- Cluster Quality: K-Means achieved 84% higher silhouette score (0.2849 vs 0.1548)
- Class Alignment: K-Means showed 163% better ARI performance (0.8975 vs 0.3413)
- Cluster Balance: K-Means produced more balanced cluster sizes
- Robustness Trade-off: K-Medoids lower performance but theoretically more robust to outliers
- Practical Preference: K-Means demonstrated superior performance for this well-structured dataset

PCA Visualization Results:
- First Principal Component: 36.2% variance explained
- Second Principal Component: 19.2% variance explained
- Total Variance Captured: 55.41% in 2D visualization
- Cluster Separation: Clear visual distinction between wine classes in reduced space

Challenges Faced and Design Decisions

1. External Library Dependency Resolution
Challenge: sklearn_extra.cluster import error preventing K-Medoids implementation
Solution: Developed custom SimpleKMedoids class implementing PAM algorithm
Implementation: Basic medoid selection, iterative assignment, convergence checking
Benefit: Self-contained solution compatible with standard scikit-learn ecosystem
Learning: Understanding core algorithmic principles through manual implementation

2. High-Dimensional Data Visualization
Challenge: Visualizing 13-dimensional clustering results meaningfully
Solution: PCA dimensionality reduction to 2D while preserving cluster patterns
Decision: Used first two principal components capturing 55.41% total variance
Trade-off: Some information loss but gained interpretable visual representation
Validation: Cluster patterns remained distinguishable in reduced space

3. Performance Metric Selection
Decision: Dual evaluation approach using Silhouette Score and Adjusted Rand Index
Rationale: Silhouette measures internal cluster quality, ARI compares with ground truth
Insight: K-Means excelled in both metrics, confirming superior performance
Alternative Considered: Additional metrics like Calinski-Harabasz index for comprehensive evaluation

4. Data Preprocessing Strategy
Decision: Z-score standardization applied to all features
Justification: Features had different scales (alcohol vs magnesium concentrations)
Validation: Mean ≈ 0, Standard deviation = 1 confirmed successful normalization
Impact: Ensured equal feature contribution to distance calculations

5. Visualization Design and Layout
Approach: 2x2 subplot layout for comprehensive comparison
Components: Original classes, K-Means results, K-Medoids results, performance metrics
Enhancement: Marked centroids and medoids for algorithm understanding
Design Choice: Color-coded clusters with performance metrics in titles

Technical Implementation Details

Libraries Used:
- sklearn: Machine learning algorithms, datasets, preprocessing, metrics
- numpy: Numerical computations and array operations
- matplotlib: Data visualization and plotting
- pandas: Data manipulation and analysis

Code Structure:
1. Data Loading & Exploration - Dataset characteristics and feature analysis
2. Data Preprocessing - Z-score standardization implementation
3. K-Means Implementation - Standard sklearn algorithm with parameter tuning
4. Custom K-Medoids - SimpleKMedoids class with PAM algorithm logic
5. Performance Evaluation - Multiple metrics calculation and comparison
6. PCA Visualization - Dimensionality reduction for interpretable plotting
7. Comparative Analysis - Side-by-side algorithm comparison with insights

Custom K-Medoids Algorithm:
- Initialization: Random medoid selection from dataset
- Assignment: Points assigned to nearest medoids using Euclidean distance
- Update: New medoids selected to minimize intra-cluster distances
- Convergence: Iteration until medoid positions stabilize
- Output: Final cluster labels and medoid indices

Reproducibility:
- Fixed random state (random_state = 42)
- Consistent preprocessing pipeline
- Documented parameter choices
- Deterministic algorithm implementations

Results Summary

| Algorithm | Silhouette Score | Adjusted Rand Index | Cluster Balance | Inertia |
|-----------|------------------|---------------------|-----------------|---------|
| K-Means | 0.2849 | 0.8975 | Balanced (65,51,62) | 1277.93 |
| K-Medoids | 0.1548 | 0.3413 | Unbalanced (89,24,65) | 549.38 |

Key Findings:
- K-Means achieved superior performance across all evaluation metrics
- 84% higher silhouette score indicates better-defined clusters for K-Means
- 163% higher ARI demonstrates K-Means better recovered original wine classes
- K-Medoids showed more conservative clustering with unbalanced cluster sizes
- Custom K-Medoids implementation successfully replicated algorithm behavior
- PCA visualization effectively reduced dimensionality while preserving cluster structure

Performance Insights:
- K-Means optimal for well-separated, spherical clusters like wine dataset
- K-Medoids theoretical robustness advantage not realized in this clean dataset
- Standardization crucial for distance-based clustering algorithms
- Visualization confirmed algorithmic differences in cluster formation patterns

Future Improvements

1. Extended Clustering Algorithms: Implement hierarchical clustering, DBSCAN for comparison
2. Advanced Medoids Selection: Implement full PAM optimization for better K-Medoids performance
3. Feature Engineering: Explore feature selection impact on clustering quality
4. Validation Techniques: Implement silhouette analysis across different k values
5. Distance Metrics: Experiment with Manhattan, Cosine similarity for alternative clustering
6. Outlier Analysis: Investigate specific data points causing performance differences
7. Cross-Validation: Implement stability analysis across multiple random initializations
8. Advanced Visualization: 3D plotting, t-SNE for alternative dimensionality reduction

Algorithm Selection Guidelines

K-Means Preferable When:
- Data exhibits spherical cluster shapes
- Computational efficiency is priority
- Dataset has minimal outliers
- Balanced cluster sizes expected
- Fast convergence required

K-Medoids Preferable When:
- Dataset contains significant outliers
- Need interpretable cluster representatives
- Non-spherical cluster shapes present
- Robust performance more important than speed
- Working with categorical or mixed data types

Repository Structure

```
MSCS_634_Lab_3/
├── README.md                           # This comprehensive analysis file
├── clustering_analysis.ipynb           # Main Jupyter notebook with complete implementation

```

Setup and Execution

Prerequisites:
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Running the Analysis:
1. Clone this repository
2. Open clustering_analysis.ipynb in Jupyter
3. Run all cells sequentially
4. Review generated visualizations and performance metrics
5. Examine detailed analysis output and insights

Alternative Execution:
```bash
python clustering_analysis.py
```

Expected Output:
- Dataset characteristics and preprocessing results
- K-Means and K-Medoids performance metrics
- Comprehensive performance comparison table
- PCA visualization with cluster plots
- Detailed algorithmic analysis and recommendations
