#Bank Customer Churn Prediction
This project aims to predict the customers who might leave the bank based on the historical data provided.

#The Work Plan
Two pipelines are investigated: 1) Quick Pipeline and 2) Long Pipeline with different preprocessing steps.


1.   **Short Pipeline**

  * Reading the dataset.
  * Splitting the data into training data (historical data) and testing data (future data).
  * Encoding the categorical features including the ordinal and not ordinal ones.
  * Modelling: 
      * Automatic tunning using Bayesian optimisation.
      * Cross-validation to check the generalisation error.
      * Training then evaluating the XGB classifier using different metrics. 

  * Results Discussion

2.   **Long Pipeline** 

  * Reading the dataset.
  * Splitting the data into training data (historical data) and testing data (future data).
  * Encoding the categorical features including the ordinal and not ordinal ones.
  * Imputations.
  * One-Hot Encoding.
  * Analysing the dataset (Descriptive statistics, Checking the correlation).
  * Standardisation.
  * Modelling: 
      * Automatic tunning using Bayesian optimisation.
      * Cross-validation to check the generalisation error.
      * Training then evaluating different classifiers using different metrics. 
  * Results Discussion

* Questions & Answers




