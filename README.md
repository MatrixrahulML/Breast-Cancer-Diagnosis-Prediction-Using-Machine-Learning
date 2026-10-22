# Breast Cancer Diagnosis Prediction Using Machine Learning

## Project Overview

This project explores whether machine learning can classify breast tumor samples as benign or malignant using the Wisconsin Diagnostic Breast Cancer dataset.

**Objective:** Build and evaluate a reproducible binary classification pipeline and investigate the factors influencing its predictions.

> Educational research project only. This model is not a clinically validated diagnostic tool.

## Problem Statement

Missing a malignant case is an important error to investigate. Therefore, this project evaluates malignant-class recall alongside ROC-AUC, precision, F1-score and accuracy rather than relying on accuracy alone.

## Dataset

- 569 samples
- 30 numeric diagnostic features
- Target: `diagnosis`
- Classes: Benign (B) and Malignant (M)

## Workflow

1. Data inspection and cleaning
2. Exploratory data analysis and visualization
3. Stratified train/test split
4. Five-fold cross-validation
5. Comparison of Logistic Regression, Random Forest and SVM
6. Evaluation using ROC-AUC, recall, precision, F1-score and a confusion matrix
7. Permutation feature importance
8. Model serialization and Streamlit demonstration

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook and Streamlit.

## Results

Add your actual cross-validation and held-out test results here after running the notebook.

- Selected model: [insert model]
- Cross-validation ROC-AUC: [insert measured result]
- Test ROC-AUC: [insert measured result]
- Test malignant recall: [insert measured result]
- Test precision: [insert measured result]

## How to Run

1. Open `breast_cancer_prediction.ipynb` in Google Colab.
2. Upload the dataset file when prompted.
3. Run the cells in order.
4. Follow the application instructions in the project files to launch the demo.

## Limitations

This project uses a relatively small benchmark dataset. Results may not generalize to different populations, imaging systems, clinical settings or modern diagnostic workflows. No clinical validation has been performed.

## Future Improvements

- Hyperparameter optimization
- Probability calibration and validation of classification thresholds
- External validation on independent data
- More robust model explanation
- Automated testing and deployment

## Author

Rahul Sah

[My github](https://github.com/MatrixrahulML)