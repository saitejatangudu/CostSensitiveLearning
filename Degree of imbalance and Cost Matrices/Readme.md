# Analysis of Cost-Sensitive Algorithms for Degree of Imbalancing

This repository contains the data and code used in the experiments performed in the paper titled "Analysis of Cost-Sensitive Algorithms for Degree of Imbalancing".

## Reference

- Paper: [Analysis of Cost-Sensitive Algorithms for Degree of Imbalancing](https://link.springer.com/chapter/10.1007/978-3-031-38296-3_6)
- Authors: Sai Teja Tangudu, Rajeev Kumar
- Conference: International Conference on Computational Intelligence in Data Science, 2023

## Abstract

Class imbalance, coupled with other data characteristics such as feature set and class separability, can significantly impact the performance of machine learning algorithms. In this paper, we focus on cost-sensitive learning applied at the algorithmic level, particularly using the Cost-Sensitive Logistic Regression (CSLR) algorithm. We propose a methodology to empirically evaluate the performance of cost-sensitive algorithms across varying degrees of imbalanced data. This involves inducing a cost matrix into the training process, which forces the model to penalize misclassification errors according to the data's skewness, thereby reducing bias towards the majority class. We present empirical evaluations of the CSLR algorithm on four popular datasets and analyze its behavior using Mean Absolute Error (MAE) and Kappa values.

## Datasets

This section describes the datasets used for empirical analysis, including attribute information, preprocessing steps, and class distribution summaries. All attributes have been scaled to the range [0, 1] using MinMaxScaler from Python's scikit-learn library.

### Vehicle Dataset

- Description: A four-class dataset with eighteen attributes and four classes: Opel, bus, Saab, and van. The dataset is reclassified into "Van" and "Non Van" classes to induce class imbalance.
- Class Distribution: 199 minority samples vs. 647 majority samples, imbalance ratio (IR) of 0.307.

### Pima Indians Diabetes Dataset

- Description: Used for binary classification tasks with class labels "diabetic" and "non-diabetic". It contains diagnostic metrics of patients.
- Class Distribution: 268 positive samples (diabetic) vs. 768 negative samples (non-diabetic).

### Ionosphere Dataset

- Description: Binary class dataset used to classify radar as "good" or "bad".
- Class Distribution: 225 samples in the majority class ("good radar"), 126 samples in the minority class ("bad radar").

### Abalone Dataset

- Description: Classifies the age of abalone based on physical characteristics. It has seven attributes excluding the discrete feature "sex".
- Class Distribution: 42 samples in the minority class (class "18") vs. 689 samples in the majority class (class "9").

## Notebooks

For each dataset, two notebooks are provided:
1. Error Bar Analysis: Illustrates the performance of logistic regression across different degrees of imbalance and variable performance concerning different cost matrices.
2. Kappa Score Analysis: Examines the performance of the model based on Kappa score variations.

Feel free to explore the code and data provided in this repository to replicate the experiments and further analyze the results. If you have any questions or inquiries, feel free to contact the authors. 