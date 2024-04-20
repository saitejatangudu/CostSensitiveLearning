from copy import copy
import gc
import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score,recall_score,precision_score
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from random import sample
from ds_pipe.utils.helper_functions import key_with_min_val, key_with_max_val
from ds_pipe.datasets.dataset_utils import generate_bunch_dataset
from ds_pipe.datasets.dataset_loader import load_dataset_to_bunch
from os import listdir 
import warnings
warnings.filterwarnings('once')

def get_minority_class(y): 
    counts = Counter(y)
    return key_with_min_val(counts)

def get_majority_class(y): 
    counts = Counter(y)
    return key_with_max_val(counts)

def undersampler(dataset, IR): 
    """
    # Description
    Undersamples a dataset by removing points from the minority to reach the desired imbalance ratio.

    # Keyword arguments: 
    - dataset : A bunch dataset with dataset.data and dataset.target 
    - IR : The Imbalance ratio to achieve on the dataset
    """
    counts = Counter(dataset.target)
    org_ir = max(counts.values())/min(counts.values())
    assert IR <= max(counts.values())/1, "Desired IR is not achievable without Oversampling the majority class"
    
    
    c_min = key_with_min_val(counts)
    c_maj = key_with_max_val(counts)
    
    number_of_minority_samples = round(max(counts.values())/IR)
    number_of_majority_samples = round(IR*min(counts.values()))

    minority_indices = [i for i,yi in enumerate(dataset.target) if  yi == c_min]
    majority_indices = set(range(len(dataset.target)))-set(minority_indices)


    if IR > org_ir: 
        rus = RandomUnderSampler(sampling_strategy={c_min: number_of_minority_samples, c_maj: max(counts.values())})
        X, y = rus.fit_resample(dataset.data, dataset.target)
    elif IR < org_ir: 
        rus = RandomUnderSampler(sampling_strategy={c_min: min(counts.values()), c_maj: number_of_majority_samples})
        X, y = rus.fit_resample(dataset.data, dataset.target)
    else: 
        return dataset
    
    #reordering the dataset
    return generate_bunch_dataset(X,y)

def load_synthetic_datasets(): 
    synthetic_datasets = [
    (load_dataset_to_bunch("./src/datasets/02a.csv"), "02a"),
    (load_dataset_to_bunch("./src/datasets/02b.csv"), "02b"),
    (load_dataset_to_bunch("./src/datasets/03subcl5.csv"), "subcl5"),
    (load_dataset_to_bunch("./src/datasets/03subcl5-4000-noise.csv"), "subcl5-noise"),
    (load_dataset_to_bunch("./src/datasets/04clover5.csv"), "clover"),
    (load_dataset_to_bunch("./src/datasets/04clover5-noise.csv"), "clover-noise"),
    #(load_dataset_to_bunch("./src/datasets/flower-3d.csv"), "flower"),
    (load_dataset_to_bunch("./src/datasets/paw3-2d-border-center.csv"), "paw-2d"),
    (load_dataset_to_bunch("./src/datasets/paw3-2d-border-dense-center.csv"), "paw-2d-border-dense-center"),
    (load_dataset_to_bunch("./src/datasets/paw3-2d-only-border.csv"), "paw-2d-only-border"),
    (load_dataset_to_bunch("./src/datasets/paw3-2d-very-dense-center.csv"), "paw-2d-very-dense-center"),
    (load_dataset_to_bunch("./src/datasets/gaussian_overlap_0.83_0.17_1000_1_1.csv"), "gaussian_overlap_1std"),
    (load_dataset_to_bunch("./src/datasets/gaussian_overlap_0.83_0.17_1000_1_2.csv"), "gaussian_overlap_2std"),
    (load_dataset_to_bunch("./src/datasets/gaussian_overlap_0.83_0.17_1000_1_3.csv"), "gaussian_overlap_3std"),
    (load_dataset_to_bunch("./src/datasets/gaussian_overlap_0.83_0.17_1000_1_4.csv"), "gaussian_overlap_4std"),
    (load_dataset_to_bunch("./src/datasets/local_imbalance_degree_0.83_0.17_0.05_1000.csv"), "local_imbalance_degree_005"),
    (load_dataset_to_bunch("./src/datasets/local_imbalance_degree_0.83_0.17_0.1_1000.csv"), "local_imbalance_degree_01"),
    (load_dataset_to_bunch("./src/datasets/local_imbalance_degree_0.83_0.17_0.2_1000.csv"), "local_imbalance_degree_02"),
    (load_dataset_to_bunch("./src/datasets/local_imbalance_degree_0.83_0.17_0.5_1000.csv"), "local_imbalance_degree_05"),
    (load_dataset_to_bunch("./src/datasets/uniform_overlap_0.83_0.17_10_1000.csv"), "uniform_overlap_10"),
    (load_dataset_to_bunch("./src/datasets/uniform_overlap_0.83_0.17_20_1000.csv"), "uniform_overlap_20"),
    (load_dataset_to_bunch("./src/datasets/uniform_overlap_0.83_0.17_40_1000.csv"), "uniform_overlap_40"),
    (load_dataset_to_bunch("./src/datasets/uniform_overlap_0.83_0.17_60_1000.csv"), "uniform_overlap_60"),
    (load_dataset_to_bunch("./src/datasets/uniform_overlap_0.83_0.17_80_1000.csv"), "uniform_overlap_80"),
    (load_dataset_to_bunch("./src/datasets/uniform_only_boundary_no_overlap_0.83_0.17_1000.csv"), "uniform_only_boundary_no_overlap"),
    (load_dataset_to_bunch("./src/datasets/multi_modal_no_overlap_0.83_0.17_1000.csv"), "multi_model_no_overlap"),
    (load_dataset_to_bunch("./src/datasets/multi_modal_overlap_0.83_0.17_1000.csv"), "multi_modal_overlap")
    ] 

    return synthetic_datasets

def load_all_synthetic_datasets(): 
    synthetic_datasets = []
    for fname in listdir("./src/datasets/"): 
        synthetic_datasets.append((load_dataset_to_bunch(f"./src/datasets/{fname}"), fname.split(".c")[0]))
    return synthetic_datasets

def load_real_datasets():
    from ds_pipe.datasets.dataset_loader import DatasetCollections
    from src.measures import imbalance_ratio
    dc = DatasetCollections()
    return [(dataset, dataset_name) for dataset, dataset_name in dc.all_datasets() if len(dataset.target) <= 10000 and len(set(dataset.target)) == 2 and imbalance_ratio(dataset) > 1.1]

def load_dataset_from_collection(dataset_name, collection):
    return [dataset for dataset, name in collection if name == dataset_name][0] 

def round_array(l,num_digits = 4): 
    """
    Danger Danger, this method mutates the given list.
    # Keyword arguments: 
    - l : A list of floats 
    - num_digits the number of digits to round the elements too
    """
    for i in range(len(l)): 
        l[i] = round(l[i],num_digits)
    return l

def base_skf_test(X, y, classifier, num_random_samples=2,resampler=None): 
    gmean_scores = []
    f1_scores = []
    macro_recall = []
    macro_precision = []
    c_min = get_minority_class(y) 
    for _ in range(num_random_samples): 
        skf=StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if resampler != None: 
                rt = resampler()
                X_train, y_train = rt.fit_resample(X_train, y_train)
            clf = copy(classifier)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            gmean_scores.append(geometric_mean_score(y_test, y_pred))
            f1_scores.append(f1_score(y_true=y_test, y_pred=y_pred, pos_label=c_min))
            macro_recall.append(recall_score(y_true=y_test, y_pred=y_pred, pos_label=c_min))
            macro_precision.append(precision_score(y_true=y_test, y_pred=y_pred, pos_label=c_min))
            #except: 
            #    print("Exception on macro recall score")
            #    macro_recall.append(0)
            #Deleting every single trace of the clf
            del clf
            gc.collect()

    return gmean_scores, f1_scores, macro_recall,macro_precision



def write_result(classifier_name, dataset_name, trials, space_e, result_dict): 
    with open(f"./results/hyperopt/{classifier_name}.csv", mode="a+") as f: 
        best_loss = trials.best_trial['result']['loss']
        gmean, f1, mac_rec, mac_prec = result_dict[best_loss] 
        f.write(f"{dataset_name}, {round(np.mean(gmean),5)}, {round(np.mean(f1),5)}, {round(np.mean(mac_rec),5)}, {round(np.mean(mac_prec),5)}, {','.join([str(x) for x in space_e.items()])}\n")

