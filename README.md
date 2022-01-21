# Ben Fuqua Senior Project
## Outlier Detection Methods
This repository will hold all of the various research materials, links, code chunks, and programs developed during my senior project

## Organization
You will find 2 folders within the main page of the repo. The first one being for global methods and the second being about local methods. Each folder will have a README page that describes what is within that folder, links to articles found on various methods, and python scripts that will have code examples of the various methods.

This README will have links to general notes about outlier detection methods and hold my timeline for the project
## Time Line
Week 1: Project Definition
Weeks 2-4: Research Global and local outlier detection models
Week 5-7: Finish up research on various methods/organize information
Week 8-10: Start the recommendation program
Week 11-12: Start making presentation and finish up program
Week 13: Final preparation and presentation

## General Notes


#### Brief Overview
- Outlier 
    - The training data contains outliers which are defined as observations that are far from the others. 
    - Outlier detection estimators thus try to fit the regions where the training data is the most concentrated, ignoring the deviant observations.
    - Unsupervised anomaly detection
    - cannot form dense cluster/assumption is they are located in low density regions.
    
- Novelty
    - The training data is not polluted by outliers and we are interested in detecting whether a new observation is an outlier. 
    - In this context an outlier is also called a novelty.
    - Semi-supervised anomaly detection.
    - Possible to form dense cluster, but must be in a low density region of the training data.

#### Strategy Implementation
The scikit-learn project provides a set of machine learning tools that can be used for both novelty and outlier detection. 
- Outlier detection occurs in the `estimator.fit(X_train)
- Novelty detection occurs in the `estimator.predict(X_test)
- Inliers are labeled '1'
- Outliers are labeled '-1'

The predict method makes use of a threshold on the raw scoring function respective to the estimator. This function is accessible through the score_samples method, and the threshold is controlled by the contamination parameter. 
The decision_function method is also defined from the scoring function, in such a way that negative values are outliers and non-negative ones are inliers

**Special Note**: Local Outlier FActor does not support predict, decision function, and score_samples methods but only a fit_predict method, as this estimator was originally meant to be applied for outlier detection(not outlier and novelty). You can access the individual scores through the negative_outlier_factor_ attribute. If you really want to use this estimator for novelty detection, you can instantiate the estimator with the novelty parameter = True, but in this case fit_predict is not available. When set to true, you must only use predict and score_samples on new unseen data or else you will get incorrect scores.

# Overview of Methods
A side by side comparison of the detection algorithms in scikit-learn. Local Outlier Factor (LOF) does not show a decision boundary because it is no predict method. 

![](model_examples.png)
###### Picture Credit: SciKit-Learn 2.7.1

#### Novelty Detection
Consider a sample with *n* observations (rows) and *p* features (columns). If we add another observation, how do we know if it is regular? (i.e. does it come from the same distribution?) Novelty detection addresses this question

Simply put, it is about to learn a threshold of the initial observation's distribution, plotted in *p*-dimensional space. Then, if our observation lies within the threshold, we assume they come from the same population (distribution). Else, we can say they are abnormal with a given confidence in our assessment

#### Outlier Detection
Similar in goal to Novelty Detection, the goal is to separate a core of regular observations from polluting ones, called outliers. Yet, in the case of outlier detection, we don't have a clean data set representing the population of regular observations that can be used to train any tool.







# Appendix: Links

[Overview of various models](https://scikit-learn.org/stable/modules/outlier_detection.html)
- Click on the picture and names of the estimators to learn specific information about them. 
