# Assignment4-for-IN26011064

## Breast Cancer Classification using K-Nearest Neighbors (KNN)

**Author:** Kushagra Raghuvanshi  

**Registration Number:** 23BSA10072

**Application Number:** IN26011064

**Batch Number:** 2B

**Email ID:** kushagra.23bsa10072@vitbhopal.ac.in 

## Objective
The objective of this project is to build a K-Nearest Neighbors (KNN) classification model ($k=5$) to accurately classify breast tumors as Malignant (M) or Benign (B) based on diagnostic measurements.

## Dataset Link
- [Kaggle: Breast Cancer Wisconsin Diagnostic Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

## Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `kaggle`

## Methodology
1. **Data Understanding**: Identified numerical features and target variable (`diagnosis`), inspecting data types and distributions.
2. **Data Preprocessing**:
   - Dropped unnecessary columns (`id` and `Unnamed: 32`).
   - Encoded target variable `diagnosis` (`M`: 1, `B`: 0).
   - Split dataset into 80% training and 20% testing sets using stratified sampling.
   - Standardized features using `StandardScaler` to equalize feature contributions across distance metrics.
3. **Model Development**: Trained a `KNeighborsClassifier` with $k=5$ on the scaled training features.
4. **Model Evaluation**: Evaluated the model using Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix heatmap.

## Results
- **Accuracy:** 95.61%
- **Precision:** 97.44%
- **Recall:** 90.48%
- **F1-Score:** 0.9383

## Conclusion
This project implemented a K-Nearest Neighbors (KNN) classifier (k==5) on the Breast Cancer Wisconsin Diagnostic dataset to predict tumor malignancy. Key findings demonstrate that diagnostic cell attributes effectively differentiate between benign and malignant tumors, achieving an overall classification accuracy of 95.61% and an F1-score of 0.9383.

Feature scaling using StandardScaler is critical for KNN because the algorithm calculates distance metrics (such as Euclidean distance) between data points. Features with larger raw numerical scales (e.g., area_mean) would otherwise completely dominate the distance metric over smaller-scale features (e.g., smoothness_mean), biasing the neighbors' majority vote.

A key limitation of the KNN algorithm is its high computational complexity during inference. Because KNN is a non-parametric, lazy learner, it requires computing distances to every training instance for each query prediction, making it slow and memory-intensive as datasets scale.
