# Bank Customer Churn Prediction

This project aims to predict the customers who might leave the bank based on the historical data provided.

# The Work Plan
Two pipelines are investigated: 1) Short Pipeline and 2) Long Pipeline with different preprocessing steps.


1.   **Short Pipeline**

   * Reading the dataset.
   * Splitting the data into training data (historical data) and testing data (future data).
   * Encoding the categorical features including the ordinal and not ordinal ones.
   * Modelling: 
      * Automatic tunning using Bayesian optimisation.
      * Cross-validation to check the generalisation error.
      * Training then evaluating the XGB classifier using different metrics. 

   * Results Discussion.

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
    * Results Discussion.

# Questions & Answers

1.   Which machine learning models did you consider and why?

To determine the best approach for this classification problem, I considered different classical machine learning techniques including a) tree-based models (XGBClassifier and Random Forest); b) distance-based models ( Support Vector Machine) and probability-based model (Naive Bayes and Logistic Regression). These models have different behaviour in handling the classification problems. The tree-based models rely on a series of conditional statements to partition the data points into subsets. The distance-based models classify data points by the dissimilarity between them as measured by distance functions. Finally, the probability-based models, like Naive Bayes, rely on conditional probability to determine the likelihood of a specific class given a set of independent features.

----------------------------

2.   What are the pros and cons of using each of these models?

*   Logistic Regression

Pros

    *   Simple algorithm.
    *   Interpretable.
    *   Less prone to over-fitting (with low-dimensional data).
    *   Best option  when data is linearly separable.

Cons

    *   Sensitivity to outliers.
    *   Require data preparation such as scale, normalize data and one-hot encoding that might be computationally expensive.
    *   Bad option when data has complex relationships.


*   Random Forest

Pros

    *   Interpretable (relatively).
    *   No need to scale and normalize data (less effort in regards to preprocessing).
    * Performs well with imbalanced datasets.
    * Performs well with big data.
    * Outliers have less impact.
    * Can be used for feature selection since it provides the feature importance.

Cons

    * Comparing to other models, takes a longer time to train and expensive complexity wise.

*   SVM

Pros

    * Outliers have less impact.
    * Performs well with high dimensions ( a large number of features).
    * Very good option with binary classification and NLP applications (e.g. text classification).

Cons

    * Requires data preparation such as scale and normalize data.
    * Comparing to other models, takes a longer time to train.

*   NB

Pros

    * Very fast (suitable for real-time prediction).
    * Performs well with big data.
    * Insensitive to irrelevant features.
    * Performs well with high dimensions ( a large number of features).
    * Good for multi-class prediction.

Cons 

    * Assumption of the features independency doesn't hold.
    *  Training data should be representative.

* XGBoost

Pros

    * No need to scale and normalize data (less effort in regards to preprocessing).
    * Can be used for feature selection since it provides the feature importance.
    * Outliers have less impact.
    * Performs well with large-sized datasets.
    * Less prone to overfitting.
    * Best option for almost any classification problem.

Cons

    * Too many hyperparameters (harder to tune).



-----------------------------

3. What criteria did you use to decide which model was the best?

After the initial dataset analysis, I expect the distance-based technique would be the best (precisely XGBoost). However, it is worth evaluating the other techniques. All the models have been evaluated using a range of metrics including:

  * Accuracy
  * F1
  * Jaccard
  * Kappa
  * Log_loss
  * Classification report (recall and precision)
  * Confusion matrix
  * Cross-validation 10-fold (Kappa and F1 scores)

Mainly I used **Kappa** (Cohen's kappa coefficient). Kappa provides a measure of statistical agreement between the predicted class and the actual class that takes into consideration the probability that the model classifies correctly by chance. Kappa values > 0.75 are considered to be a strong indicator that the classifier's performance is excellent; between 0.4 to 0.75 is considered as fair to good; lower than 0.4 is understood to be weak. I chose XGBoost as the best in this situation based on these metrics and the pros mentioned in the previous question. However, if the tunning would be a big problem, then random forest lays itself as a very good candidate as well. 

-----------------------------
4. Would you change your solution in any way if the input dataset was much larger(e.g. 10GB)?

For evaluating and selecting the best model, I would sample from this big dataset a representative dataset so I can work easily. 

For training the final model, I would use progressive data loading techniques. Some of the other stages might be too computationally expensive and I might look for alternatives. Other than this, I think XGBoost and random forest can handle large size datasets. There's always room to optimise the pipeline more.

-----------------------------
5. How would you approach this challenge if the input dataset could not fit in your device memory?

I would use a big data platform such as Apache Spark which is distributed processing system that set up a cluster of multiple nodes.

Apart from Spark, these some ways to handle this problem:

  * Use a computer with more memory. :)
  * Work with a smaller sample.
  * Change the data format.
  * Stream data or use progressive loading.
  * Use a relational database.

-----------------------------
6. What other considerations would need to be taken into account if this was real customer data?

Generally, real-world data is messy and noisy and can break the trained models in weird ways.  Therefore, I think the data pipeline would include more steps to catch outliers that might affect the model performance in production.
Each stage in the pipeline should consider uncertainty and ambiguous values and prepare a proper way to handle them, i.e. an automated data validation tooling with a confidence level. Also, sensitive customer data should be considered in proper ways.







