from audioop import minmax
import numpy as np
from src.utils import base_skf_test, load_real_datasets
from src.utils import load_synthetic_datasets
from sklearn.naive_bayes import ComplementNB as NaiveBayes
from sklearn.preprocessing import MinMaxScaler
"""
Since there is no parameter space to search through for GaussianNaiveBayes, we only have the unoptimized experiment here.
"""

def unoptimized_experiment(datasets, write_result=False): 
    gmean_scores, f1_scores, rec_scores, prec_scores= [],[],[],[]
    for dataset, dataset_name in datasets: 
        minmax_scaler = MinMaxScaler()
        normalized_data = minmax_scaler.fit_transform(dataset.data)
        print(f"{dataset_name}")
        clf = NaiveBayes()
        gmeans, f1s, mac_recs, mac_precs = base_skf_test(normalized_data, dataset.target, clf)
        if write_result:  
            with open("./results/unopt/nb.csv","a") as f:
                f.write(f"{dataset_name}, {round(np.mean(gmeans),3)}, {round(np.mean(f1s),3)}, {round(np.mean(mac_recs),3)}, {round(np.mean(mac_precs),3)} \n")
        gmean_scores.append(np.mean(gmeans))
        f1_scores.append(np.mean(f1s))
        rec_scores.append(np.mean(mac_recs))
        prec_scores.append(np.mean(mac_precs))
    return gmean_scores, f1_scores, rec_scores, prec_scores
    

if __name__=="__main__": 
    datasets = load_synthetic_datasets() + load_real_datasets()
    unoptimized_experiment(datasets, write_result=True)
