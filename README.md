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

The following data mining steps were conducted in order to make all data sources ready for model's application:

## Null and Missing Values 

All data sources were checked for missing and null values. The analysis was made considering the following assumptions:
- Is the missing data related to the outcome (informative missingness)? 
- Is the percentage of missing data substantial enough to remove the predictor?
- Is the missing data concentraded on specific samples?
- Is it possible to use the training data to estimate the missing values?

## Feature Engineering

Activities regarding the selection and creation of relevant attributes which contribute most to the predicted outputs. In order to make the observations easy to understand by the models. A featuring engineering analysis was conducted in all the categorical predictors to check if it was required to reduce the number of classes (categories) on those predictors.

## Feature Selection 

One of the primary reasons to measure the strength or relevance of the predictors is to filter which should be used as inputs in a model. This supervised feature selection can be data driven based on the existing data. The results of the filtering process can be a critical step in creating an effective model.

## One-Hot Encoding 

Some machine learning models required that all predictors must be of numerical type. One-Hot Encoding is the process of translate categorical predictors into a computer-based
format.

## Collinearity and Multicollinearity 

Collinearity is the technical term for the situation where a pair of predictors variables have a substantial correlation with each other. Multicollinearity is when multiple predictors have relationships at the same time. Another way of eliminating redundant predictors as well as irrelevant ones is to select a subset of predictors that have little intercorrelation between its features, especially in classifier algorithms.

## Balancing 

Activities regarding the definition of balance of the target output of data. Sampling techniques such as downsampling and oversampling are also presented in this step, as shown on the following diagrams:

|![](/Figures/adult_balance_binary.png)     | ![](/Figures/adult_balance_binary_new.png)          |
|-------------------------------------------|-----------------------------------------------------|
| Adult Data Set without Sampling technique | Adult Data Set after downsampling technique applied |

# Development Setup

The project was created based on Anaconda environment. The following libraries are necessary in order to run the codes:
- re
- numpy
- pandas
- seaborn
- matplotlib.pyplot
- scikit-learn

All libraries are already installed in Anaconda environment default installation. For more information about how to install Anaconda environment please refer to its [documentation](https://www.anaconda.com/products/individual).

# Usage

The codes are divided by dataset, as follows:

## Preprocessing

The preprocessing steps necessary to implement the machine learning models.

- bsil19151608-DMML1_P_Prep_Dataset_1.ipynb - [Adult Data Set](http://archive.ics.uci.edu/ml/datasets/Adult)
- bsil19151608-DMML1_P_Prep_Dataset_2.ipynb - [Online News Popularity Dataset](http://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)
- bsil19151608-DMML1_P_Prep_Dataset_3.ipynb - [Hotel Booking Demand Dataset](https://www.kaggle.com/jessemostipak/hotel-booking-demand)

## Model's application

The application of the machine learning models as well as its evaluation using accuracy metric and confusion matrix.

- bsil19151608-DMML1_P_Model_Dataset_1.ipynb - [Adult Data Set](http://archive.ics.uci.edu/ml/datasets/Adult)
- bsil19151608-DMML1_P_Model_Dataset_2.ipynb - [Online News Popularity Dataset](http://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)
- bsil19151608-DMML1_P_Model_Dataset_3.ipynb - [Hotel Booking Demand Dataset](https://www.kaggle.com/jessemostipak/hotel-booking-demand)

# Results

All models were applied considering the same procedure and using default parameters setting to avoid any errors in interpretation of the results phase. After the model's application, the results were ranked based on the Accuracy score, a common approach in the data science field. By collecting the two best Accuracy results, we break down the models by using the Confusion Matrix method to check if the model does not show any inconsistencies such as higher occurrence of false negatives instead of false positives.
The following diagram shows an example of the Confusion Matrix evaluation technique:

![](/Figures/adult_confusion_matrix.png) 
