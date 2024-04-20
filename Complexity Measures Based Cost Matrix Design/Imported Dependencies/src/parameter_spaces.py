from hyperopt import hp
import numpy as np 
import math

tree_space = {"min_samples_leaf": hp.choice('min_samples_leaf', range(2,11)), 
              "min_impurity_decrease": hp.choice('min_impurity_decrease', [0.0,0.1,0.2,0.3,0.4,0.5])} 

knn_space = {'n_neighbors': hp.choice('n_neighbors', range(1,33,2))}

svm_space = {'kernel': hp.choice('kernel', ["rbf", "sigmoid", "linear"]), #We removed the polykernel, from the Barella setup, since it throws errors. 
            'gamma': hp.choice('gamma', [2**i for i in range(-10,11)]),  
            'degree': hp.choice('degree', [2,3,4,5])}

def get_mlp_space(m):
    """
    # Keyword arguments
    - m: The number of features in the dataset  
    """ 
    a = int((m+2)/2)
    mlp_space = {'learning_rate_init': hp.choice('learning_rate_init', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]),
                'hidden_layer_sizes':hp.choice('hidden_layer_sizes', [int(x) for x in np.arange(a-3,a+4,1) if int(x) > 0])}
    return mlp_space

def get_rf_space(m): 
    """
    # Keyword arguments
    - m: The number of features in the dataset
    """ 
    if m == 1:
        max_features_list = [1]
    elif m == 2:
        max_features_list = [1,2]
    else:
        max_features_list = [math.ceil(math.sqrt(m)/2), math.ceil(math.sqrt(m)),math.ceil(2*math.sqrt(m))]
    
    rf_space = {'n_estimators': hp.choice('n_estimators', np.arange(100,1100,100)),
                'max_features':hp.choice('max_features', max_features_list)}
    return rf_space