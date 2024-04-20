# ADASYN based Cost-Matrix-Design

## Overview
This repository contains the implementation and experimental results of a novel approach for designing cost matrices in imbalanced classification problems. The proposed method utilizes an ADASYN-based complexity measure to assess the difficulty of imbalanced datasets and assign appropriate misclassification cost weights to samples. The aim is to enhance the performance of classifiers, particularly in scenarios where class imbalances pose challenges.

## Contents
1. **Imported Dependencies**: This directory contains all the necessary dependencies imported from the master repository of the paper titled "An Interpretable Measure of Dataset Complexity for Imbalanced Classification Problems".

2. **NoteBooks**: Here, you can find the scripts and code used to conduct experiments and evaluate the performance of different complexity measures in cost matrix design. 

3. **Experiment Results**: This holds the results obtained from the experiments conducted using various complexity measures. It includes performance metrics such as accuracy, precision, recall, F1-score, Kappa, GMean, AUC, and PR Score, analyzed across multiple datasets.

## Introduction
The introduction section provides a detailed overview of the research undertaken. It introduces the problem of imbalanced classification and the importance of accurate complexity measures for designing effective cost matrices. Additionally, it outlines the motivation behind the proposed ADASYN-based complexity measure and its potential benefits in improving classifier performance.

## Review of Existing Complexity Measures
This section reviews several existing complexity measures used in imbalanced classification tasks. It discusses measures such as N1, N3, TLCM, and the Imbalance Ratio (IR), highlighting their strengths, limitations, and areas of focus. This review serves as a basis for comparing these measures with the proposed ADASYN-based approach.

## Conclusion
The conclusion summarizes the findings of the study, emphasizing the effectiveness of the proposed ADASYN-based complexity measure in comparison to existing measures. It discusses the superior performance of the proposed method across multiple datasets and metrics, reaffirming its reliability and potential in handling class imbalance effectively.

## How to Use
Users can replicate the experiments conducted in the "Experiments" directory by following the provided scripts and guidelines. Additionally, they can explore the experiment results in the "Experiment Results" directory to gain insights into the comparative performance of different complexity measures.