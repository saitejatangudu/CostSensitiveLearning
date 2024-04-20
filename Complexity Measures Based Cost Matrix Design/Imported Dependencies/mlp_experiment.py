from sklearn.neural_network import MLPClassifier
from src.parameter_spaces import get_mlp_space
from src.utils import write_result
import numpy as np
from hyperopt import fmin, tpe, Trials, space_eval
from hyperopt import STATUS_OK
from src.utils import base_skf_test
from src.utils import load_real_datasets,load_synthetic_datasets
from src.utils import write_result
import warnings

warnings.simplefilter("ignore")

datasets = load_real_datasets()

n_iterations = 1000
def optimized_experiment(datasets, write_result_flag=False, max_evals=100, resampler=None): 
    gmean_scores, f1_scores, rec_scores, prec_scores = [],[],[],[]
    
    for dataset, dataset_name in datasets: 
        m = len(dataset.data[0]) # Number of variables
        space = get_mlp_space(m)
        result_dict = {}
        
        def objective(params):
            clf = MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=params['learning_rate_init'], max_iter=1000)
            gmeans, f1s, mac_recs, mac_precs = base_skf_test(dataset.data, dataset.target, clf, resampler=resampler)
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
            write_result("mlp",dataset_name, trials, space_eval(space, best), result_dict)
        
        best_loss = trials.best_trial['result']['loss']
        gmean, f1, mac_rec, mac_prec = result_dict[best_loss] 
        gmean_scores.append(np.mean(gmean))
        f1_scores.append(np.mean(f1))
        rec_scores.append(np.mean(mac_rec))
        prec_scores.append(np.mean(mac_prec))
    return gmean_scores, f1_scores, rec_scores, prec_scores



def unoptimized_experiment(datasets, write_results=False): 
    gmean_scores, f1_scores, rec_scores = [],[],[] 
    for dataset, dataset_name in datasets: 
        print(f"{dataset_name}",end=" ")
        clf = MLPClassifier(max_iter=n_iterations)
        gmeans, f1s, mac_recs = base_skf_test(dataset.data, dataset.target, clf)
        if write_result:  
            with open(f"./results/unopt/mlp_{n_iterations}.csv","a") as f:
                f.write(f"{dataset_name}, {round(np.mean(gmeans),3)}, {round(np.mean(f1s),3)}, {round(np.mean(mac_recs),3)} \n")
        gmean_scores.append(np.mean(gmeans))
        f1_scores.append(np.mean(f1s))
        rec_scores.append(np.mean(mac_recs))
    return gmean_scores, f1_scores, rec_scores

        


if __name__=="__main__": 
    datasets = load_synthetic_datasets() + load_real_datasets()
    optimized_experiment(datasets,write_result_flag=True)
    #unoptimized_experiment()



