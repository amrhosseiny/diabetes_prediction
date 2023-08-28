# Read the dataset from CSV file
dataset <- read.csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
# Load the required packages
library(dplyr)

# Display the dataset
print(dataset)

# Summary statistics
summary(dataset)

# Correlation matrix
cor(dataset)

# Frequency table for categorical variables
table(dataset$Diabetes_binary)
table(dataset$HighBP)
table(dataset$HighChol)
table(dataset$CholCheck)
table(dataset$Smoker)
table(dataset$Stroke)
table(dataset$HeartDiseaseorAttack)
table(dataset$PhysActivity)

# Boxplot for continuous variables
boxplot(dataset$BMI, main = "BMI")
boxplot(dataset$Age, main = "Age")
boxplot(dataset$Education, main = "Education")
boxplot(dataset$Income, main = "Income")


#######################################
#               part two              #
#######################################

# To determine which variables have the highest predictive power for whether someone
# has diabetes or not, we can use various feature selection techniques and statistical
# tests. Here we use logistic regression and built-in R functions to assess their significance.

# Load required package
library(caret)

# Create a binary outcome variable
dataset$Diabetes_binary <- factor(dataset$Diabetes_binary)

# Train a logistic regression model
model <- train(Diabetes_binary ~ ., data = dataset, method = "glm", family = "binomial")

# Get variable importance
var_importance <- varImp(model)

# Display the variable importance
print(var_importance)

#######################################
#       parts three & four             #
#######################################

# Load required packages
library(lattice)
library(ggplot2)
library(caret)
library(Matrix)
library(glmnet)
library(pROC)
library(randomForest)
library(rpart)
library(rpart.plot)
library(kknn)



# Create a binary outcome variable
dataset$Diabetes_binary <- factor(dataset$Diabetes_binary)

# Split the data into training and validation sets
set.seed(123)  # Set seed for reproducibility
train_index <- createDataPartition(dataset$Diabetes_binary, p = 0.8, list = FALSE)
train_data <- dataset[train_index, ]
validation_data <- dataset[-train_index, ]

x_train <- as.matrix(train_data[, -1])
y_train <- train_data$Diabetes_binary

# Perform logistic regression with LASSO regularization
lasso_model <- glmnet(x_train, y_train, family = "binomial", alpha = 1)
best_lambda <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)$lambda.min
lasso_model_best <- glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = best_lambda)

# Get variable importance from the logistic regression model
var_importance <- coef(lasso_model_best)[-1]
var_importance <- as.matrix(var_importance)

# Extract the selected features
non_zero_features <- which(var_importance[-1, ] != 0)
selected_features <- colnames(train_data)[non_zero_features]

# Make predictions on the validation set
x_validation <- as.matrix(validation_data[, -1])
validation_preds <- predict(lasso_model_best, newx = x_validation, type = "response")
validation_preds <- as.factor(ifelse(validation_preds > 0.5, 1, 0))
validation_outcome <- validation_data$Diabetes_binary

# Ensure both predicted and actual values have the same levels
levels(validation_outcome) <- levels(validation_preds)

# Calculate sensitivity, specificity, and accuracy on the validation set
confusion_mat <- confusionMatrix(validation_preds, validation_outcome)
sensitivity <- confusion_mat$byClass["Sensitivity"]
specificity <- confusion_mat$byClass["Specificity"]
accuracy <- confusion_mat$overall["Accuracy"]

# Calculate the AUC on the validation set
roc_obj <- roc(as.numeric(validation_outcome) - 1, as.numeric(validation_preds) - 1)
auc <- auc(roc_obj)

# Display the selected features and their importance
print(paste("Selected Features:", non_zero_features))
print(paste("Variable Importance:", var_importance))

# Display sensitivity, specificity, accuracy, and AUC on the validation set
print(paste("Sensitivity:", sensitivity))
print(paste("Specificity:", specificity))
print(paste("Accuracy:", accuracy))
print(paste("AUC:", auc))
##################################################################################
# Random Forest

rf_model <- randomForest(x = x_train, y = y_train, ntree = 100, importance = TRUE)
var_importance <- rf_model$importance[, "MeanDecreaseGini"]
importance_threshold <- 0.01  # Adjust as needed
selected_features <- names(var_importance[var_importance > importance_threshold])

# Subset the data with selected features
selected_data <- dataset[, c("Diabetes_binary", selected_features)]

# Prepare the training and validation datasets with selected features
x_train_selected <- as.matrix(train_data[, selected_features])
y_train_selected <- train_data$Diabetes_binary

x_validation_selected <- as.matrix(validation_data[, selected_features])
y_validation_selected <- validation_data$Diabetes_binary

# Train a logistic regression model with selected features
logit_model <- glm(Diabetes_binary ~ ., data = selected_data, family = "binomial")

# Make predictions on the validation set using the logistic regression model
validation_preds <- predict(logit_model, newdata = as.data.frame(x_validation_selected), type = "response")
validation_preds <- as.factor(ifelse(validation_preds > 0.5, "1", "0"))
validation_outcome <- validation_data$Diabetes_binary

# Ensure both predicted and actual values have the same levels
levels(validation_outcome) <- levels(validation_preds)

# Calculate sensitivity, specificity, and accuracy on the validation set
confusion_mat <- confusionMatrix(validation_preds, validation_outcome)
sensitivity <- confusion_mat$byClass["Sensitivity"]
specificity <- confusion_mat$byClass["Specificity"]
accuracy <- confusion_mat$overall["Accuracy"]

# Calculate the AUC on the validation set
roc_obj <- roc(as.numeric(validation_outcome) - 1, as.numeric(validation_preds) - 1)
auc <- auc(roc_obj)

# Display the selected features
print(paste("Selected Features:", selected_features))

# Display sensitivity, specificity, accuracy, and AUC on the validation set
print(paste("Sensitivity:", sensitivity))
print(paste("Specificity:", specificity))
print(paste("Accuracy:", accuracy))
print(paste("AUC:", auc))

##########################################################################
# Tree-based

# Create a binary outcome variable
dataset$Diabetes_binary <- factor(dataset$Diabetes_binary)

# Split the data into training and validation sets
set.seed(123)  # Set seed for reproducibility
train_index <- createDataPartition(dataset$Diabetes_binary, p = 0.8, list = FALSE)
train_data <- dataset[train_index, ]
validation_data <- dataset[-train_index, ]

x_train <- as.matrix(train_data[, -1])
y_train <- train_data$Diabetes_binary

# Fit a decision tree model
tree_model <- rpart(y_train ~ ., data = as.data.frame(cbind(y_train, x_train)), method = "class")

# Visualize the decision tree
rpart.plot(tree_model)

# Prepare the validation data
x_validation <- as.matrix(validation_data[, -1])
y_validation <- validation_data$Diabetes_binary

# Make predictions on the validation set
validation_preds <- predict(tree_model, newdata = as.data.frame(x_validation), type = "class")

# Calculate confusion matrix
conf_mat <- table(validation_preds, y_validation)

# Calculate sensitivity, specificity, accuracy
sensitivity <- conf_mat[2, 2] / sum(conf_mat[2, ])
specificity <- conf_mat[1, 1] / sum(conf_mat[1, ])
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)

# Calculate AUC
roc_obj <- roc(ifelse(y_validation == "1", 1, 0), as.numeric(validation_preds) - 1)
auc <- auc(roc_obj)

# Display the evaluation metrics
print(paste("Sensitivity:", sensitivity))
print(paste("Specificity:", specificity))
print(paste("Accuracy:", accuracy))
print(paste("AUC:", auc))


###############################################################################
# KNN

# Create a binary outcome variable
dataset$Diabetes_binary <- factor(dataset$Diabetes_binary, levels = c("0", "1"), labels = c("No", "Yes"))

# Split the data into training and validation sets
set.seed(123)  # Set seed for reproducibility
train_index <- createDataPartition(dataset$Diabetes_binary, p = 0.8, list = FALSE)
train_data <- dataset[train_index, ]
validation_data <- dataset[-train_index, ]

# Define the outcome variable
y_train <- train_data$Diabetes_binary
y_validation <- validation_data$Diabetes_binary

# Define the predictor variables
x_train <- train_data[, -1]
x_validation <- validation_data[, -1]

# Define the control parameters for cross-validation
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Train the KNN model with cross-validation and grid search
knn_model <- train(x = x_train, y = y_train, method = "knn", trControl = ctrl, tuneLength = 5)

# Make predictions on the validation set using the trained model
validation_preds <- predict(knn_model, newdata = x_validation)

# Calculate evaluation metrics on the validation set
conf_mat <- confusionMatrix(validation_preds, y_validation)
sensitivity <- conf_mat$byClass["Sensitivity"]
specificity <- conf_mat$byClass["Specificity"]
accuracy <- conf_mat$overall["Accuracy"]

# Calculate AUC on the validation set
roc_obj <- roc(ifelse(y_validation == "Yes", 1, 0), ifelse(validation_preds == "Yes", 1, 0))
auc <- auc(roc_obj)

# Display the evaluation metrics
print(paste("Sensitivity:", sensitivity))
print(paste("Specificity:", specificity))
print(paste("Accuracy:", accuracy))
print(paste("AUC:", auc))
optimal_k <- knn_model$bestTune$k

print(optimal_k)
