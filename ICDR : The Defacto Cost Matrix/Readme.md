# ICDR : The Defacto Cost Matrix
## Introduction
Most real-world applied Machine Learning (ML) classification tasks involve imbalanced data. Addressing class imbalance is crucial for improving classifier performance. While many existing techniques focus on data level and algorithmic level solutions, their effectiveness in handling the skewness aspect of imbalanced data, alongside factors such as class overlapping, often receive lesser attention. This repository aims to bridge this gap by empirically reviewing a set of Cost-Sensitive Learning (CSL) algorithms that utilize Inverse Class Distribution Ratio (ICDR) based cost matrices. We assess their performance against cost-insensitive (CSIL) algorithms, considering various factors such as class overlapping, minority class sample size, diverse data domains, and ML models.

## Objective
The primary objective of this repository is to provide a comprehensive empirical evaluation of CSL and CSIL algorithms in handling class imbalance. By conducting experiments across various datasets and considering different factors influencing class imbalance, we aim to enhance the understanding of effective strategies for tackling imbalanced data.

## Content
### 1. Use Case Wise Results
This section contains detailed information about the experimental results for each specific use case. It provides insights into the performance of CSL and CSIL algorithms across different scenarios.

### 2. Dataset Folders
Each of the 12 folders represents a dataset used in our experiments. Within each dataset folder, you'll find:

- **Data Files:** Raw data files used for training and evaluation.
- **Methods:** Implementation of CSL and CSIL algorithms.
- **Models:** Trained ML models for each algorithm.
- **Results:** Experimental results, including performance metrics and analysis.

## How to Use
1. Clone the repository to your local machine:
   ```
   git clone https://github.com/your_username/your_repository.git
   ```

2. Navigate to the desired dataset folder:
   ```
   cd dataset_name
   ```

3. Explore the contents of the folder, including data files, implemented methods, trained models, and experimental results.

4. Run the provided scripts or notebooks to reproduce the experiments or integrate the algorithms into your own projects.

## Citation
If you find this repository useful in your research or work, please consider citing our paper:

[Insert citation details here]

## Contributions
Contributions to this repository are welcome! If you have any suggestions, improvements, or additional datasets/algorithms to include, feel free to open an issue or submit a pull request.
