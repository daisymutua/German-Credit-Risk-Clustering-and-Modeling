################## DATA PREPROCESSING ######################

# Load necessary libraries
library(tidyverse)
library(caret)
library(ggplot2)
library(scales)

# Replace NA values with "unknown"
df[is.na(df)] <- "unknown"

# Check for NA values
colSums(is.na(df))

# Drop the '...1' column
df <- df %>% select(-`...1`)

# Define categorical features
categorical_features <- c('Sex', 'Job', 'Housing', 'Saving accounts', 
                          'Checking account', 'Purpose', 'Risk')

# Label encode the categorical features
df[categorical_features] <- lapply(df[categorical_features], 
                                   function(x) as.numeric(factor(x)))

# Show head of new dataframe
head(df)

# LOG TRANSFORMATION OF NUMERIC FEATURES
num_df <- df %>% select(Age, Duration, `Credit amount`)
num_df_log <- log(num_df)

# Plot distributions after log transformation
par(mfrow = c(1, 3))

# Credit amount
hist(num_df_log$`Credit amount`, breaks = 40, main = "Credit Amount", 
     col = "skyblue", xlab = "", border = "white", probability = TRUE)
lines(density(num_df_log$`Credit amount`), col = "blue", lwd = 2)

# Duration
hist(num_df_log$Duration, breaks = 40, main = "Duration", 
     col = "salmon", xlab = "", border = "white", probability = TRUE)
lines(density(num_df_log$Duration), col = "red", lwd = 2)

# Age
hist(num_df_log$Age, breaks = 40, main = "Age", 
     col = "darkviolet", xlab = "", border = "white", probability = TRUE)
lines(density(num_df_log$Age), col = "black", lwd = 2)


###################### STANDARDISE ###################

# STANDARDISE THE NUMERIC FEATURES
preProc <- preProcess(num_df_log, method = c("center", "scale"))
num_df_scaled <- predict(preProc, num_df_log)

# Show dimensions and values
num_df_scaled <- as.data.frame(num_df_scaled)
num_df_scaled <- data.frame(lapply(num_df_scaled, as.numeric))

dim(num_df_scaled)
head(num_df_scaled)

#################### PREDICTIVE MODELLING #######################


# Load necessary libraries
library(xgboost)
library(caret)
library(dplyr)
library(ggplot2)
library(pROC)
library(tibble)
library(tidyr)
library(data.table)
library(parallel)
library(mlbench)
library(vip)

# Combine numeric and categorical data
num_df_scaled <- as.data.frame(num_df_scaled)
colnames(num_df_scaled) <- c("Age", "Duration", "Credit_amount")  
cat_df <- df %>% select(all_of(categorical_features))  

data <- bind_cols(cat_df, num_df_scaled)
data$Risk <- df$Risk  # Add target variable
# Display first few rows
head(data)

############# 1. XGBOOST #####################

# Split data into features (x) and target (y)
x <- data[, !grepl("Risk", names(data))]
y <- data$Risk

# Perform train-test split
set.seed(101)  # Equivalent to random_state
split_index <- sample(nrow(data), nrow(data)*0.2)
x_test <- x[split_index, ]
x_train <- x[-split_index, ]
y_test <- y[split_index]
y_train <- y[-split_index]

# Print dimensions
print(paste('xtrain shape: ', dim(x_train)[1], 'rows,', dim(x_train)[2], 'columns'))
print(paste('xtest shape: ', dim(x_test)[1], 'rows,', dim(x_test)[2], 'columns'))
print(paste('ytrain shape: ', length(y_train)))
print(paste('ytest shape: ', length(y_test)))

##### HYPERPARAMETER TUNING ####

##### Convert target to factor if it's categorical
y_train <- as.factor(y_train)
levels(y_train)
# Automatically make them valid R names
levels(y_train) <- make.names(levels(y_train))

# Combine x and y for caret
train_data <- cbind(x_train, Risk = y_train)

# Generate random hyperparameter grid (25 samples like Python's n_iter=25)
set.seed(123)
xgb_random_grid <- data.frame(
  nrounds = sample(50:300, 25, replace = TRUE),  # equivalent to n_estimators
  max_depth = sample(1:50, 25, replace = TRUE),
  eta = runif(25, 0, 2),                         # learning_rate in Python
  gamma = runif(25, 0, 1),            # similar to gamma uniform(1, 0.000001)
  colsample_bytree = runif(25, 0.5, 1),
  min_child_weight = sample(1:10, 25, replace = TRUE),
  subsample = runif(25, 0.5, 1)
)

# Define trainControl with cross-validation
control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

# Train XGBoost model using random search
set.seed(123)
xgb_model <- train(
  Risk ~ .,
  data = train_data,
  method = "xgbTree",
  metric = "ROC",
  tuneGrid = xgb_random_grid,
  trControl = control
)

########### PLOTTING ROC CURVE ##############
plot_roc_cv <- function(x, y, model_func, n_splits = 10) {
  set.seed(123)
  folds <- caret::createFolds(y, k = n_splits, list = TRUE, 
                              returnTrain = FALSE)
  
  all_roc_data <- list()
  
  for (i in seq_along(folds)) {
    test_idx <- folds[[i]]
    train_idx <- setdiff(seq_along(y), test_idx)
    
    test_y <- y[test_idx]
    
    # Skip fold if only one class is present
    if (length(unique(test_y)) < 2) {
      message(sprintf("Skipping Fold %d: Only one class present in test set.", i))
      next
    }
    
    model <- model_func(x[train_idx, ], y[train_idx])
    
    dtest <- xgboost::xgb.DMatrix(data = as.matrix(x[test_idx, ]))
    prob_pred <- predict(model, dtest)
    
    # Force levels to match full dataset (in correct order)
    roc_obj <- pROC::roc(response = test_y,
                         predictor = prob_pred,
                         levels = levels(y),  # keep original levels
                         direction = "<")     # adjust if needed
    
    fold_auc <- as.numeric(pROC::auc(roc_obj))
    n_points <- length(roc_obj$sensitivities)
    
    df <- data.frame(
      fpr = 1 - roc_obj$specificities,
      tpr = roc_obj$sensitivities,
      fold = paste("Fold", i),
      auc = rep(fold_auc, n_points)
    )
    
    all_roc_data[[i]] <- df
  }
  
  
  
  roc_df <- dplyr::bind_rows(all_roc_data)
  mean_auc <- mean(sapply(all_roc_data, function(df) unique(df$auc)))
  
  ggplot(roc_df, aes(x = fpr, y = tpr, color = fold)) +
    geom_line(alpha = 0.6, size = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
    labs(
      title = sprintf("10-Fold ROC Curve (Mean AUC = %.2f)", mean_auc),
      x = "False Positive Rate",
      y = "True Positive Rate",
      color = "Fold"
    ) +
    theme_minimal()
}

plot_roc_cv(x, y, xgb_model_func, n_splits = 10)


#############2. Random forest ###############
# Combine features and label
rf_train_data <- cbind(x_train, Risk = y_train)

# Random forest training with caret
set.seed(123)
rf_model <- train(
  Risk ~ .,
  data = rf_train_data,
  method = "rf",
  metric = "ROC",
  trControl = control,
  tuneLength = 5
)

rf_probs <- predict(rf_model, x_test, type = "prob")[, 2]
rf_preds <- ifelse(rf_probs > 0.5, levels_y[2], levels_y[1])
rf_preds <- factor(rf_preds, levels = levels_y)
confusionMatrix(data = rf_preds, reference = y_test)

rf_model_func <- function(x_train, y_train) {
  df <- cbind(x_train, Risk = y_train)
  colnames(df) <- make.names(colnames(df))
  model <- randomForest::randomForest(Risk ~ ., data = df)
  return(model)
}

plot_roc_cv_rf <- function(x, y, model_func, n_splits = 10) {
  set.seed(123)
  x <- x[, , drop = FALSE]
  colnames(x) <- make.names(colnames(x))  # Important
  
  folds <- caret::createFolds(y, k = n_splits)
  all_roc_data <- list()
  
  for (i in seq_along(folds)) {
    test_idx <- folds[[i]]
    train_idx <- setdiff(seq_along(y), test_idx)
    test_y <- y[test_idx]
    
    if (length(unique(test_y)) < 2) {
      message(sprintf("Skipping Fold %d: Only one class in test set.", i))
      next
    }
    
    model <- model_func(x[train_idx, ], y[train_idx])
    prob_pred <- predict(model, x[test_idx, , drop = FALSE], type = "prob")[, 2]
    
    roc_obj <- pROC::roc(
      response = test_y,
      predictor = prob_pred,
      levels = levels(y),
      direction = "<"
    )
    
    df <- data.frame(
      fpr = 1 - roc_obj$specificities,
      tpr = roc_obj$sensitivities,
      fold = paste("Fold", i),
      auc = rep(pROC::auc(roc_obj), length(roc_obj$sensitivities))
    )
    
    all_roc_data[[i]] <- df
  }
  
  roc_df <- dplyr::bind_rows(all_roc_data)
  mean_auc <- mean(sapply(all_roc_data, function(df) unique(df$auc)))
  
  ggplot(roc_df, aes(x = fpr, y = tpr, color = fold)) +
    geom_line(alpha = 0.6, size = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(
      title = sprintf("10-Fold ROC (Random Forest, Mean AUC = %.2f)", mean_auc),
      x = "False Positive Rate", y = "True Positive Rate"
    ) +
    theme_minimal()
}

plot_roc_cv_rf(x, y, rf_model_func)
#############3. # Logistic regression ###############
# y is a binary factor with proper levels
y <- as.factor(y)
levels(y) <- make.names(levels(y))  

# Combine features and labels
log_train_data <- cbind(x_train, Risk = y_train)

# Set up caret control
control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

# Train logistic regression
set.seed(123)
log_model <- train(
  Risk ~ .,
  data = log_train_data,
  method = "glm",
  family = "binomial",
  metric = "ROC",
  trControl = control
)

log_probs <- predict(log_model, x_test, type = "prob")[, 2]
log_preds <- ifelse(log_probs > 0.5, levels_y[2], levels_y[1])
log_preds <- factor(log_preds, levels = levels_y)
confusionMatrix(data = log_preds, reference = y_test)


log_model_func <- function(x_train, y_train) {
  df <- cbind(x_train, Risk = y_train)
  model <- glm(Risk ~ ., data = df, family = binomial)
  return(model)
}

plot_roc_cv_logreg <- function(x, y, model_func, n_splits = 10) {
  set.seed(123)
  folds <- caret::createFolds(y, k = n_splits)
  all_roc_data <- list()
  
  for (i in seq_along(folds)) {
    test_idx <- folds[[i]]
    train_idx <- setdiff(seq_along(y), test_idx)
    test_y <- y[test_idx]
    
    if (length(unique(test_y)) < 2) {
      message(sprintf("Skipping Fold %d: Only one class in test set.", i))
      next
    }
    
    model <- model_func(x[train_idx, ], y[train_idx])
    prob_pred <- predict(model, x[test_idx, ], type = "response")
    
    roc_obj <- pROC::roc(
      response = test_y,
      predictor = prob_pred,
      levels = levels(y),
      direction = "<"
    )
    
    fold_auc <- as.numeric(pROC::auc(roc_obj))
    df <- data.frame(
      fpr = 1 - roc_obj$specificities,
      tpr = roc_obj$sensitivities,
      fold = paste("Fold", i),
      auc = rep(fold_auc, length(roc_obj$sensitivities))
    )
    
    all_roc_data[[i]] <- df
  }
  
  roc_df <- dplyr::bind_rows(all_roc_data)
  mean_auc <- mean(sapply(all_roc_data, function(df) unique(df$auc)))
  
  ggplot(roc_df, aes(x = fpr, y = tpr, color = fold)) +
    geom_line(alpha = 0.6, size = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(
      title = sprintf("10-Fold ROC (LogReg, Mean AUC = %.2f)", mean_auc),
      x = "False Positive Rate", y = "True Positive Rate"
    ) +
    theme_minimal()
}

plot_roc_cv_logreg(x, y, log_model_func)


