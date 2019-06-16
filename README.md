# Credit-Card-Fraud-Detection
My solution to the Credit Card Fraud Detection detection dataset on Kaggle.

Dataset can be found at: https://www.kaggle.com/mlg-ulb/creditcardfraud

Dependencies:
* numpy
* pandas
* scikit-learn
* seaborn
* matplotlib
* imblearn

Files:
* brute_force_model_selection.py
* gradient_boosting_optimization.py


### brute_force_model_selection.py

Brute force model search implementation. Tests out Linear SVM, Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting Classifier

**Result: GradientBoostingClassifier is optimal**

### gradient_boosting_optimization.py

Exhaustive hyperparameter search implementation. Calculates recall and log loss scores for different settings of learning_rate and n_estimators for GradientBoosting Classifier. 

**Result: learning_rate = 0.5 and n_estimators = 100 is optimal**
