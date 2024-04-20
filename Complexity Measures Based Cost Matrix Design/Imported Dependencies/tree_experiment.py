import numpy as np
from src.utils import base_skf_test, load_real_datasets
from src.utils import load_synthetic_datasets, write_result
from src.parameter_spaces import tree_space
from sklearn import tree
from hyperopt import fmin, tpe, Trials, space_eval
from hyperopt import STATUS_OK

def optimized_experiment(datasets, write_result_flag=False, max_evals=30, resampler=None):
    gmean_scores, f1_scores, rec_scores, prec_scores = [],[],[],[]
    for dataset, dataset_name in datasets:
        print(f"{dataset_name}", end=" ")
        m = len(dataset.data[0]) # Number of variables

        result_dict = {}

        def objective(params):
            clf = tree.DecisionTreeClassifier(min_samples_leaf=params['min_samples_leaf'])
            gmeans, f1s, mac_recs, mac_precs= base_skf_test(dataset.data, dataset.target, clf, resampler=resampler)
            result_dict[1-np.mean(mac_recs)] = (gmeans, f1s, mac_recs, mac_precs)
            return {'loss': 1-np.mean(mac_recs), 'status': STATUS_OK}

        trials = Trials()
        best = fmin(objective,
                space = tree_space,
                algo=tpe.suggest,
                max_evals = max_evals,
                trials = trials
                )
        if write_result_flag:  
            write_result("tree",dataset_name, trials, space_eval(tree_space, best), result_dict)
        
        best_loss = trials.best_trial['result']['loss']
        gmean, f1, mac_rec, mac_prec= result_dict[best_loss] 
        gmean_scores.append(np.mean(gmean))
        f1_scores.append(np.mean(f1))
        rec_scores.append(np.mean(mac_rec))
        prec_scores.append(np.mean(mac_prec))
    return gmean_scores, f1_scores, rec_scores, prec_scores



def unoptimized_experiment(datasets, write_result_flag=False): 
    gmean_scores, f1_scores, rec_scores, prec_scores= [],[],[],[]
    for dataset, dataset_name in datasets: 
        print(f"{dataset_name}",end=" ")
        clf = tree.DecisionTreeClassifier()
        gmeans, f1s, mac_recs, mac_precs = base_skf_test(dataset.data, dataset.target, clf)
        if write_result_flag:  
            with open("./results/unopt/tree.csv","a") as f:
                f.write(f"{dataset_name}, {round(np.mean(gmeans),3)}, {round(np.mean(f1s),3)}, {round(np.mean(mac_recs),3)}, {round(np.mean(mac_precs),3)} \n")
        gmean_scores.append(np.mean(gmeans))
        f1_scores.append(np.mean(f1s))
        rec_scores.append(np.mean(mac_recs))
        prec_scores.append(np.mean(mac_precs))
    return gmean_scores, f1_scores, rec_scores, prec_scores
    

if __name__=="__main__": 
    datasets = load_synthetic_datasets() + load_real_datasets()
    optimized_experiment(datasets, write_result_flag=True)
