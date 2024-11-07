Heart Disease Prediction Project
This project applies machine learning models to predict the likelihood of heart disease based on patient data. It compares multiple classification algorithms to identify the most effective model for accurate prediction.

Table of Contents
Overview
Dataset
Data Preprocessing
Models Used
Evaluation Metrics
Results
How to Run
Key Takeaways
Overview
The aim of this project is to evaluate and compare various machine learning algorithms for predicting heart disease, using metrics such as accuracy, precision, recall, F1 score, and error rate. This comparison offers insights into each model's effectiveness for healthcare prediction.

Dataset
The dataset includes a variety of health indicators relevant to heart disease, such as:

Age
Gender
Chest pain type
Resting blood pressure
Serum cholesterol
Fasting blood sugar
Resting electrocardiographic results
Maximum heart rate achieved
Exercise-induced angina
Oldpeak (ST depression)
Slope of peak exercise ST segment
Number of major vessels colored by fluoroscopy
Thalassemia
Target (indicator of heart disease presence)
Data Preprocessing
Load the dataset and filter rows without missing target values.
Replace remaining missing values with 0.
Drop irrelevant columns.
Convert categorical columns into factors to prepare the dataset for classification.
Models Used
Five machine learning algorithms were implemented:

Classification Tree (ctree): A tree-based model, useful for interpretability.
Recursive Partitioning (rpart): A decision tree technique that splits data by important features.
Support Vector Machine (SVM): A robust classifier that maximizes margin between classes.
Random Forest: An ensemble model that combines multiple trees to improve accuracy.
Neural Network: A multi-layer perceptron model for capturing complex patterns.
Evaluation Metrics
Each model's performance was measured with:

Accuracy: Measures overall correctness of predictions.
Precision: Evaluates the accuracy of positive predictions.
Recall: Assesses the model's ability to capture actual positives.
F1 Score: Balances precision and recall.
Error Rate: Rate of incorrect predictions.
Results
The comparison of model performance metrics provides insights into which models perform best in predicting heart disease. These findings help determine the most suitable algorithm for this dataset.

How to Run
Clone this repository and open it in R or RStudio.
Install required packages if not already installed:
r
Copy code
install.packages(c("caTools", "party", "rpart", "rpart.plot", "e1071", "randomForest", "nnet"))
Load the dataset and run each section of the code to preprocess data, build models, and evaluate results.
Key Takeaways
This project demonstrates how various machine learning models perform on healthcare data, especially in predicting heart disease. The results highlight the importance of model selection and evaluation in achieving reliable, data-driven healthcare insights.
