# German Credit Risk Clustering and Modeling

This project focuses on building and evaluating machine learning models to predict credit risk based on customer financial and demographic data from the German Credit dataset. The goal is to classify individuals as either good or bad credit risks, aiding financial institutions in making more informed lending decisions.

The original code was
https://www.kaggle.com/code/delllectron/bank-credit-data-clustering-modelling#PREDICTIVE-MODELLING

<h2>Models Implemented</h2>
XGBoost: Gradient boosting for handling complex interactions and imbalanced data.
Logistic Regression: A baseline model for binary classification.
Random Forest: An ensemble model to reduce overfitting and improve generalization.

<h2>Libraries Used</h2>
This project is built in R and leverages the following libraries:
caret: Model training and cross-validation
randomForest: Random forest classifier
xgboost: XGBoost model
pROC: ROC curve and AUC computation
dplyr, ggplot2, tidyr: Data wrangling and visualization

<h2>Project Structure</h2>
- data/                  # Contains the dataset
- history/               # Scripts to show the changes done over time
- plots/                 # ROC curve and other visualizations
- README.md              # Project documentation
- credit_risk.qmd       # Main script to run the project

<h2>Environment Setup (Windows)</h2>
1. To run this project successfully on Windows, you'll need to install several essential R packages. Below is a guide to set up your environment.

2. Ensure you are using R ≥ 4.4.0 and a compatible IDE such as RStudio.

3. Run the following commands in your R console to install the core libraries used in this project:

```r
install.packages("caret")
install.packages("xgboost")
install.packages("randomForest")
install.packages("pROC")
install.packages("ggplot2")
install.packages("dplyr")
```
4. These libraries support key tasks such as:
- Data preprocessing and modeling (caret)
- Advanced modeling with XGBoost and Random Forest
- Evaluation using ROC-AUC (pROC)
- Plotting results with ggplot2
- Data manipulation using dplyr

5. All code was written and tested on Windows 10, so compatibility is confirmed for that environment.

<h2>Collaborators</h2>

- Daisy Mutua
- Alain Kalonji
