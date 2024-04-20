import numpy as np
from hyperopt import fmin, tpe, Trials, space_eval
from hyperopt import STATUS_OK
from src.utils import base_skf_test
from src.utils import load_synthetic_datasets, load_real_datasets
from src.utils import write_result
from sklearn.ensemble import RandomForestClassifier
from src.parameter_spaces import get_rf_space
datasets = load_real_datasets()

def optimized_experiment(datasets, write_result_flag=False, max_evals=30, resampler=None):
    gmean_scores, f1_scores, rec_scores, prec_scores = [],[],[],[]
    for dataset, dataset_name in datasets:
        print(f"{dataset_name}", end=" ")
        m = len(dataset.data[0]) # Number of variables

        space = get_rf_space(m)
        result_dict = {}

        def objective(params):
            clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_features=params['max_features'])
            gmeans, f1s, mac_recs, mac_precs= base_skf_test(dataset.data, dataset.target, clf, resampler=resampler)
            result_dict[1-np.mean(mac_recs)] = (gmeans, f1s, mac_recs, mac_precs)
            return {'loss': 1-np.mean(mac_recs), 'status': STATUS_OK}

        trials = Trials()
        best = fmin(objective,
                space = space,
                algo=tpe.suggest,
                max_evals = max_evals,
                trials = trials
                )
        if write_result_flag:  
            write_result("rf",dataset_name, trials, space_eval(space, best), result_dict)
        
        best_loss = trials.best_trial['result']['loss']
        gmean, f1, mac_rec, mac_prec= result_dict[best_loss] 
        gmean_scores.append(np.mean(gmean))
        f1_scores.append(np.mean(f1))
        rec_scores.append(np.mean(mac_rec))
        prec_scores.append(np.mean(mac_prec))
    return gmean_scores, f1_scores, rec_scores, prec_scores


def unoptimized_experiment(datasets, write_result=False): 
    gmean_scores, f1_scores, rec_scores = [],[],[] 
    for dataset, dataset_name in datasets: 
        print(f"{dataset_name}",end=" ")
        clf = RandomForestClassifier()
        gmeans, f1s, mac_recs = base_skf_test(dataset.data, dataset.target, clf)
        if write_result: 
            with open("./results/unopt/rf.csv","a") as f:
                f.write(f"{dataset_name}, {round(np.mean(gmeans),3)}, {round(np.mean(f1s),3)}, {round(np.mean(mac_recs),3)} \n")
        gmean_scores.append(np.mean(gmeans))
        f1_scores.append(np.mean(f1s))
        rec_scores.append(np.mean(mac_recs))
    return gmean_scores, f1_scores, rec_scores

if __name__=="__main__": 
    datasets = load_synthetic_datasets() + load_real_datasets()
    optimized_experiment(datasets,write_result_flag=True)
    #unoptimized_experiment()
