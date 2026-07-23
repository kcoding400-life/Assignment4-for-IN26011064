## Conclusion

This project implemented a K-Nearest Neighbors (KNN) classifier (k==5) on the Breast Cancer Wisconsin Diagnostic dataset to predict tumor malignancy. Key findings demonstrate that diagnostic cell attributes effectively differentiate between benign and malignant tumors, achieving an overall classification accuracy of 95.61% and an F1-score of 0.9383.

Feature scaling using StandardScaler is critical for KNN because the algorithm calculates distance metrics (such as Euclidean distance) between data points. Features with larger raw numerical scales (e.g., area_mean) would otherwise completely dominate the distance metric over smaller-scale features (e.g., smoothness_mean), biasing the neighbors' majority vote.

A key limitation of the KNN algorithm is its high computational complexity during inference. Because KNN is a non-parametric, lazy learner, it requires computing distances to every training instance for each query prediction, making it slow and memory-intensive as datasets scale.
