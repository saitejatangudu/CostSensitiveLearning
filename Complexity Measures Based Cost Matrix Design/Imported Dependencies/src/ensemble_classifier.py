# In this file, I define a simple ensemble classifier consisting of many other classifiers 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from collections import Counter 
import numpy as np 
import pickle 
import src.parameter_spaces as ps
import sys
from hyperopt import space_eval

class ParameterOptimizedEnsembleClassifier: 
    def __init__(self, dataset_name): 

        # Load all the optimized parameters given the datasets. 
        with open(f"./results/hyperopt/{dataset_name}_svm.hyperopt", "rb") as f:
            self.svc_params = pickle.load(f)
         
        with open(f"./results/hyperopt/{dataset_name}_knn.hyperopt", "rb") as f:
            self.knn_params = pickle.load(f)
        
        with open(f"./results/hyperopt/{dataset_name}_mlp.hyperopt", "rb") as f:
            self.mlp_params = pickle.load(f)
        
        with open(f"./results/hyperopt/{dataset_name}_rf.hyperopt", "rb") as f:
            self.rf_params = pickle.load(f)
        
        self.svc = None
        self.knn = None #self.get_knns()
        self.nb = GaussianNB() # No parameter space to search for NB  
        self.nn = None
        self.rfs = None

    def get_svc(self): 
        #print(self.svc_params.best_trial['misc']['vals'])
        #print(self.svc_params.best_trial.keys())
        space_eval(space, self.svc_params.best_trial) 
        svc_param_dict = self.svc_params.best_trial['misc']['vals']
        degree = ps.svm_space['degree'][svc_param_dict['degree'][0]]
        kernel = ps.svm_space['kernel'][svc_param_dict['kernel'][0]]
        gamma = ps.svm_space['gamma'][svc_param_dict['gamma'][0]]
        #print(f"loaded svc with {gamma}, {kernel}, {degree} parameters")
        return SVC(kernel = kernel, gamma = gamma, degree=degree)
    
    
    def fit(self, X, y): 
        self.svc = self.get_svc() 
        self.svc.fit(X,y)        

    
    def predict(self, X): 
        svc_pred = self.svc.predict(X)
        knn_pred = self.knn.predict(X)
        nn_pred = self.nn.predict(X)
        nb_pred = self.nb.predict(X)
        rf_pred = self.rf.predict(X)
        
        res_matrix = np.array([svc_pred, knn_pred, nn_pred, nb_pred, rf_pred])
        predictions = [] 
        for i in range(len(svc_pred)): 
            predictions.append(max(Counter(res_matrix[:,i]))) 
        return predictions
    
    def save_results(self):
        pass

class EnsembleClassifier:
    #TODO This classifier needs a way of saving the individual results
    def __init__(self): 
        self.svc = SVC()
        self.kNN = KNeighborsClassifier()
        self.nn = MLPClassifier()
        self.nb = GaussianNB()
        self.lda = LinearDiscriminantAnalysis()
        self.rf = RandomForestClassifier()


    def fit(self, X, y): 
        self.svc.fit(X=X, y=y)
        self.kNN.fit(X=X, y=y)
        self.nn.fit(X=X, y=y)
        self.nb.fit(X=X, y=y)
        self.lda.fit(X=X, y=y)
        self.rf.fit(X=X, y=y)
    
    def predict(self, X): 
        svc_pred = self.svc.predict(X)
        knn_pred = self.kNN.predict(X)
        nn_pred = self.nn.predict(X)
        nb_pred = self.nb.predict(X)
        lda_pred = self.lda.predict(X)
        rf_pred = self.rf.predict(X)
        res_matrix = np.array([svc_pred, knn_pred, nn_pred, nb_pred, lda_pred, rf_pred])
        predictions = []
        for i in range(len(svc_pred)): 
            predictions.append(max(Counter(res_matrix[:,i])))
        return predictions
