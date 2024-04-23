# ADASYN based Cost-Matrix-Design

## Overview
This repository contains the implementation and experimental results of a novel approach for designing cost matrices in imbalanced classification problems. The proposed method utilizes an ADASYN-based complexity measure to assess the difficulty of imbalanced datasets and assign appropriate misclassification cost weights to samples. The aim is to enhance the performance of cost-sensitive classifiers, by designing more robust cost matrices particularly in scenarios where class imbalances pose challenges.

## Contents
1. **Imported Dependencies**: This directory contains all the necessary dependencies imported from the master repository of the paper titled "An Interpretable Measure of Dataset Complexity for Imbalanced Classification Problems".The cost matrices incorporated in the study have been prepared based on the complexity measures mentioned in this paper. 

2. **Resampling NoteBooks**: Here, you can find the scripts and code used to conduct experiments and evaluate the performance of different complexity measures in cost matrix design and how they performed in camparison to the proposed adasyn based cost matrices along side preparing the necessary imbalanced varients for the study. 

3. **Results and analysis notebook**: In this analysis notebook, various methods were assessed across multiple datasets using different evaluation metrics. Data visualization facilitated comparative analysis through line charts, depicting performance metrics across datasets, methods, and evaluation metrics with a consistent color palette. Insights were drawn by identifying top-performing methods based on average performance, aiding practitioners in method selection. The conclusion highlighted the significance of the findings in guiding method selection, while future directions suggested exploring additional datasets, methods, and hyperparameter impact for further validation and insights.
