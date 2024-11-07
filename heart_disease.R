heart_data <- read.csv(file.choose(), stringsAsFactors = FALSE)
View(heart_data)

# Create a copy of the data and filter out rows with NA in target variable
heart1 <- heart_data[!is.na(heart_data$target), ]

View(heart1)

# Check for NA values in target variable
table(is.na(heart1$target))

# Replace all NA values with 0
heart1[is.na(heart1)] <- 0

# Remove unnecessary columns (adjust as needed)
heart1 <- heart1[, -c(1, 2)]
View(heart1)
str(heart1)

# Convert relevant columns to factors for classification
heart1$target <- factor(heart1$target)
heart1$chest_pain_type <- factor(heart1$chest_pain_type)
heart1$fasting_blood_sugar <- factor(heart1$fasting_blood_sugar)
heart1$rest_ecg <- factor(heart1$rest_ecg)
heart1$exercise_induced_angina <- factor(heart1$exercise_induced_angina)
heart1$slope <- factor(heart1$slope)
heart1$vessels_colored_by_flourosopy <- factor(heart1$vessels_colored_by_flourosopy)
heart1$thalassemia <- factor(heart1$thalassemia)
str(heart1)

#(Repeat for any other columns that need to be factors based on your dataset)

# View distribution of target variable
table(heart1$target)

# Check the structure of the data
str(heart1)
View(heart1)

# Load necessary library for data splitting
library(caTools)

# Set a seed for reproducibility and split the data into training and test sets
set.seed(123)
pd <- sample.split(heart1$target, SplitRatio = 0.8)
pd

# Create training and test datasets
heart_train <- heart1[pd == TRUE, ]
heart_test <- heart1[pd == FALSE, ]
View(heart_train)
####### 
# Classification Tree (ctree)
library(party)

# Build the ctree model

model_1 <- ctree(target ~ ., data = heart_train, controls = ctree_control(mincriterion = 0.95, minsplit = 100))
model_1
plot(model_1)  # Plot the classification tree

# Make predictions on the test set
pred_1 <- predict(model_1, heart_test)
pred_1

# Generate confusion matrix for ctree model
table(pred_1, heart_test$target)

####### 
# Recursive Partitioning (rpart)
library(rpart)

# Build the rpart model
model_2 <- rpart(target ~ ., data = heart_train, method = "class")
model_2

# Plot the rpart model
library(rpart.plot)
rpart.plot(model_2)

# Make predictions on the test set
pred_2 <- predict(model_2, heart_test, type = "class")

# Generate confusion matrix for rpart model
table(pred_2, heart_test$target)

######## 
# Support Vector Machine (SVM)
library(e1071)

# Build the SVM model
model_3 <- svm(target ~ ., data = heart_train)
model_3

# Make predictions on the test set
pred_3 <- predict(model_3, heart_test)
pred_3

# Generate confusion matrix for SVM model
table(pred_3, heart_test$target)

########## 
# Random Forest
library(randomForest)

# Build the Random Forest model
model_4 <- randomForest(target ~ ., data = heart_train)
model_4

# Make predictions on the test set
pred_4 <- predict(model_4, heart_test)
pred_4

# Generate confusion matrix for Random Forest model
table(pred_4, heart_test$target)

############ 
# Neural Network
install.packages("nnet")  # Install nnet package if not already installed
library(nnet)

# Build the neural network model
model_nn <- nnet(target ~ ., data = heart_train, size = 5, decay = 0.01, maxit = 200)
model_nn

# Make predictions on the test set
pred_5 <- predict(model_nn, heart_test)

# Convert probabilities to binary predictions
predicted_classes <- ifelse(pred_5 > 0.5, 1, 0)
predicted_classes

# Generate confusion matrix for Neural Network model
table(predicted_classes, heart_test$target)

################################ 
# Comparing Results of Different Models 

# Create confusion matrices for each model
confusion_matrix_1 <- table(pred_1, heart_test$target)
confusion_matrix_2 <- table(pred_2, heart_test$target)
confusion_matrix_3 <- table(pred_3, heart_test$target)
confusion_matrix_4 <- table(pred_4, heart_test$target)
confusion_matrix_5 <- table(predicted_classes, heart_test$target)

# Function to calculate performance metrics
calculate_metrics <- function(confusion_matrix) {
  TP <- confusion_matrix[2, 2] # True Positives
  TN <- confusion_matrix[1, 1] # True Negatives
  FP <- confusion_matrix[1, 2] # False Positives
  FN <- confusion_matrix[2, 1] # False Negatives
  
  # Calculate accuracy
  accuracy <- (TP + TN) / sum(confusion_matrix)
  
  # Calculate precision
  precision <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
  
  # Calculate recall
  recall <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
  
  # Calculate F1 Score
  f1_score <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
  
  # Calculate error rate
  error_rate <- (FP + FN) / sum(confusion_matrix)
  
  return(c(accuracy = accuracy, precision = precision, recall = recall, f1_score = f1_score, error_rate = error_rate))
}

# Calculate metrics for each confusion matrix
metrics_1 <- calculate_metrics(confusion_matrix_1)
metrics_2 <- calculate_metrics(confusion_matrix_2)
metrics_3 <- calculate_metrics(confusion_matrix_3)
metrics_4 <- calculate_metrics(confusion_matrix_4)
metrics_5 <- calculate_metrics(confusion_matrix_5)

# Create a data frame to hold the results
results_df <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score", "Error Rate"),
  C_Tree = metrics_1,
  R_Part = metrics_2,
  SVM = metrics_3,
  Random_Forrest = metrics_4,
  Neural_Network = metrics_5
)
 
# Transpose the data frame for better readability
results_df <- as.data.frame(t(results_df))
colnames(results_df) <- results_df[1, ]  # Set the first row as column names
results_df <- results_df[-1, ]  # Remove the first row

# Print the results in a clear format
print(results_df)

