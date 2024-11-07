Heart Disease Classification Project
Overview
This project aims to classify heart disease using various machine learning algorithms. The dataset used contains various attributes related to heart health, and the goal is to predict the presence or absence of heart disease based on these attributes. Several classification models were implemented, including Classification Trees, Recursive Partitioning, Support Vector Machines, Random Forests, and Neural Networks.
Dataset
The dataset used in this project is sourced from [insert source here, e.g., UCI Machine Learning Repository]. It contains features related to heart health and a target variable indicating the presence of heart disease.
Features
target: Indicates the presence (1) or absence (0) of heart disease.
chest_pain_type: Type of chest pain experienced.
fasting_blood_sugar: Fasting blood sugar level.
rest_ecg: Resting electrocardiographic results.
exercise_induced_angina: Indicates if angina was induced by exercise.
slope: Slope of the peak exercise ST segment.
vessels_colored_by_fluoroscopy: Number of major vessels colored by fluoroscopy.
thalassemia: Thalassemia status.
Installation
To run this project, ensure you have R and the necessary libraries installed. You can install the required libraries using the following commands:
text
install.packages(c("party", "caTools", "rpart", "rpart.plot", "e1071", "randomForest", "nnet"))

Usage
Load the dataset using read.csv() and preprocess it by filtering out rows with NA values in the target variable.
Convert relevant columns into factors for classification.
Split the dataset into training and testing sets using sample.split().
Train multiple classification models:
Classification Tree (ctree)
Recursive Partitioning (rpart)
Support Vector Machine (svm)
Random Forest (randomForest)
Neural Network (nnet)
Evaluate each model's performance using confusion matrices and calculate metrics such as accuracy, precision, recall, F1 score, and error rate.
Results
The performance of each model is summarized in the table below:
Metric	C_Tree	R_Part	SVM	Random_Forest	Neural_Network
Accuracy	0.8439	0.8585	0.8780	1.0000	0.8976
Precision	0.8381	0.8571	0.9048	1.0000	0.9048
Recall	0.8544	0.8654	0.8636	1.0000	0.8962
F1 Score	0.8462	0.8612	0.8837	1.0000	0.9005
Error Rate	0.1561	0.1415	0.1219	0.0000	0.1024
Conclusion
This project demonstrates the application of various machine learning techniques for predicting heart disease. The Random Forest model achieved perfect accuracy and other metrics, indicating its effectiveness for this classification task, while other models also performed well with varying degrees of success.
