# A Comparative Study Between Classification Algorithms

The code is created to conduct a comparative study among several classification algorithms with its default parameters defined using three different sources of information. The first method of evaluation considered was the coefficient of determination (R2), and additional analysis was conducted using the confusion matrix technique. The datasets selected for the study were “Adult Dataset”, “Online News Popularity Dataset” and “Hotel Booking Demand Dataset”. The models used for the study were Logistic Regression, Perceptron, Decision Trees, Extra Trees Classifier, Gradient Tree Boosting, AdaBoost, Gaussian Naïve Bayes, Support Vector Classifier (SVC), Linear Support Vector Classifier (SVC), Stochastic Gradient Descent (SGD), K-Nearest Neighbors (KNN) and Random Forest. Some preprocessed techniques like attribute selection, multicollinearity and balancing were presented in this document, as well.

# Data Sources

[Adult Data Set](http://archive.ics.uci.edu/ml/datasets/Adult) - The Adult Data Set was collected from the UCI Repository. The task is to predict whether the income of an individual exceeds $50K/yr based on census data. The dataset has 48842 instances and 15 variables (features). It is also known as “Census Income” dataset.

[Online News Popularity Dataset](http://archive.ics.uci.edu/ml/datasets/Online+News+Popularity) - The Online News Popularity Dataset was collected from the UCI Repository. The task is to predict the popularity of an article based on the number of shares. The dataset has 39797 instances and 61 variables (features). It is also known as “Popularity” dataset.

[Hotel Booking Demand Dataset](https://www.kaggle.com/jessemostipak/hotel-booking-demand) - The Hotel Booking Demand Dataset was collected from the Kaggle Repository. The task is to predict if a booking is cancelled based on the features. The dataset has 119390 instances and 32 variables (features). It is also known as “Booking” dataset.

# ML Models

**Logistic Regression** - Is used where the response variable is categorical, and the idea is to find a relationship between features and probability of a particular outcome.

**Support Vector Classifier (SVC)**, **Linear Support Vector Classifier (LSVC)** - SVM is a supervised machine learning model that can be used for classification or regression problems. It has two methods: Support Vector Classifier, which is used for classification and Support Vector Regressor, which is used for regression. The main idea of this model is to find the hyperplane that separates the higher dimensional data into classes.

**Perceptron** - Perceptron has a different approach to learn a hyperplane that separates the instances of the different classes. If the data can be separated using hyperplane, it is said to be linearly separable, and there is a straightforward algorithm for finding a separating hyperplane.

**Decision Tree**, **Extra Trees Classifier**, **Gradient Tree Boosting**, **AdaBoost**, **Random Forest** - Decision tree learners are robust classifiers that utilize a tree structure to model the relationships among the features and its potential outcomes. The tree name is because it mirrors the way a real tree begins at a full trunk and splits into narrower and narrower branches as it is followed upward. Gradient Tree Boosting is an ensemble learning method and a generalization to arbitrary differentiable loss functions. Boosting is an ensemble technique in which the predictors are not made independently but sequentially. Extra Trees is an ensemble learning method, as well. It randomizes individual decisions and subsets of data to minimize over-learning from the data and overfitting. AdaBoost is used to fit a sequence of weak learners on repeatedly modified versions of the data, for instance, models that are only slightly better than random guessing, such as small decision trees. The predictions from all of them combined through a weighted majority vote (or sum) to produce the final prediction. Random Forests consists of many decision trees. Random Forests has a low classification error compared to other traditional classification algorithms because it overcomes the problem of overfitting.

**Gaussian Naïve Bayes** - Naïve Bayes is a model of linear classifiers that are known for being simple yet very efficient. The probabilistic model of naïve Bayes classifiers is based on Bayes' theorem, and the adjective naïve comes from the assumption is often violated, but naïve Bayes.

**Stochastic Gradient Descent (SGD)** - Stochastic gradient descent (SGD) is a simple yet very efficient approach to fit linear classifiers models like Support Vector Machines and Logistic Regression. It is a good option when the number of samples (and the number of features) is vast.

**K-Nearest Neighbors (KNN)** - KNN classifier is used to classify unlabeled observations by assigning them to the class of the most similar labelled examples. Features of the observations are collected for both training and test dataset. KNN is the most straightforward model to understand and apply.

# Preprocessing

**Null and Missing Values** - 

**Refactoring** -

**Feature Selection** - 

**One-Hot Enconding** - 

**Collinearity and Multicollinearity** - 

**Balancing** - 

# Usage

# Results
