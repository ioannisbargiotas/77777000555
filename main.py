#!/usr/bin/env python3
# -*- coding:utf-8 -*-



from time import time
import argparse
import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import mannwhitneyu as mww
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import multiprocessing as mp
from joblib import Parallel, delayed

    
def OOBAUC(estimator,X,y):
    PositiveOOBPosterior = np.zeros([y.shape[0],BOiterations])*np.nan 
    for j in np.arange(0,BOiterations):
        "Create the forest"
        estimator.fit(X, y)
        clf = estimator
        "Take the OOB posteror probabilities"
        OOBPosterior = clf.oob_decision_function_
        PositiveOOBPosterior[:,j] = OOBPosterior[:,-1]
    
    "Calculate AUC"
    PositiveOOB = PositiveOOBPosterior.mean(axis=1)
    n1 = np.count_nonzero(y==clf.classes_[-1])
    n2 = y.size - n1;
    TiedRank = rankdata(PositiveOOB)
    W1 = np.sum(TiedRank[y == clf.classes_[-1]])
    "W2 = np.sum(TiedRank[Y == clf.classes_[0]])"
    AUC = (W1-n1*(n1+1)/2)/(n1*n2)
    
    "Error to be maximized"
    Final_error = AUC
    
    return Final_error    


# Function of Predictor Importance (see Genuer et al 2010, Bargiotas et al 2020, Bargiotas et al 2021)
def InterpretImportance1(X,Y,params,iters,model):
    
    def InitImport(clf,X,Y,i):
        clf.fit(X, Y)
        importance = clf.feature_importances_
        return importance
    
    def SecondImport(clf,X,Y,j):
        clf.fit(X,Y)
        oobPred = clf.oob_decision_function_[:,-1]
            
        #Calculate AUC
        n1 = np.count_nonzero(Y==clf.classes_[-1])
        n2 = Y.size - n1;
        TiedRank = rankdata(oobPred)
        W1 = np.sum(TiedRank[Y == clf.classes_[-1]])
        AUC = (W1-n1*(n1+1)/2)/(n1*n2);
        imptemp = clf.feature_importances_    
        return imptemp, AUC;
    
    
    #Run multiple times and keep Importance
    importance = np.zeros([iters,X.shape[1]]) 
    clf = RandomForestClassifier(n_estimators = Ntrees,max_features = params['max_features'],min_samples_leaf = params['min_samples_leaf'],oob_score=True)
    
    
    importance = Parallel(n_jobs=-1)(delayed(InitImport)(clf,X,Y,i) for i in np.arange(0,iters))
    # with parallel_backend(backend="multiprocessing", n_jobs=-1):
        # importance = Parallel((delayed(InitImport)(clf,X,Y,i) for i in np.arange(0,iters)))
    #importance = pool.starmap(InitImport, [(clf,X,Y,i) for i in np.arange(0,iters)])
    # for i in np.arange(0,iters):
    #     clf.fit(X, Y)
    #     importance[i,:] = clf.feature_importances_
    AvImp = np.mean(importance, axis = 0)
    StdImp = np.std(importance, axis = 0)
    
    
    
    #Definition of Threshold (Check Genuer 2010)    
    DescendAvImpIndex =  (-AvImp).argsort()
    DescendAvImpValue =  AvImp[DescendAvImpIndex]
    
    AscendStdImpIndex = StdImp.argsort()
    AscendStdImpValue = StdImp[AscendStdImpIndex]
    
    rgf = RandomForestRegressor(n_estimators = Ntrees,oob_score=True)
    rgf.fit(np.arange(0,AscendStdImpValue.size).reshape(-1, 1), AscendStdImpValue.reshape(-1, 1).ravel())
    oobThres = rgf.oob_prediction_
    Threshold = oobThres.min()
    
    
    #Keep the variables that passes the Threshold
    PassedVariables = DescendAvImpIndex[DescendAvImpValue>Threshold]
    
    
    
    #Keep model depending on its mean and std of AUC 
    jsize = PassedVariables.size
    MeanAUC = np.zeros(jsize)
    StdAUC = np.zeros(jsize)
    ImportanceCell = np.zeros([jsize,X.shape[1]])
    
    for j in np.arange(0,PassedVariables.size):
        Xtemp = X[:,DescendAvImpIndex[0:j+1]]
        if Xtemp.shape[1]<params['max_features']:
            clf = RandomForestClassifier(n_estimators = Ntrees,max_features = Xtemp.shape[1],min_samples_leaf = params['min_samples_leaf'],oob_score=True)
        else:
            clf = RandomForestClassifier(n_estimators = Ntrees,max_features = params['max_features'],min_samples_leaf = params['min_samples_leaf'],oob_score=True)    
        
        imptemp = np.zeros([iters,Xtemp.shape[1]])
        AUC = np.zeros(iters)
        
        res = Parallel(n_jobs=-1)(delayed(SecondImport)(clf,Xtemp,Y,i) for i in np.arange(0,iters))
        
        # for jj in np.arange(0,iters):
        #     clf.fit(Xtemp,Y)
        #     oobPred = clf.oob_decision_function_[:,-1]
            
        #     #Calculate AUC
        #     n1 = np.count_nonzero(Y==clf.classes_[-1])
        #     n2 = Y.size - n1;
        #     TiedRank = rankdata(oobPred)
        #     W1 = np.sum(TiedRank[Y == clf.classes_[-1]])
        #     AUC[jj] = (W1-n1*(n1+1)/2)/(n1*n2);
        #     imptemp[jj,:] = clf.feature_importances_
        
        # AUC = np.array(res.AUC)
        # imptemp = np.array(res.imptemp)
        imptemp = np.array([item[0] for item in res])
        AUC = np.array([item[1] for item in res])
        MeanAUC[j] = AUC.mean()
        StdAUC[j] = AUC.std()
        ImportanceCell[j,DescendAvImpIndex[0:j+1]] =  imptemp.mean(axis = 0)
    
    #Find the most parsimonious model which is higher than MaxAUC - StdAUC[ArgMaxAUC]
    ArgMaxAUC = MeanAUC.argmax()
    MaxAUC = MeanAUC.max()
    ThresAUC = MaxAUC - StdAUC[ArgMaxAUC] 
    ImpOfPossibleNestedModels = ImportanceCell[MeanAUC>ThresAUC,:]
    
    #from the selected complexity model (usually the most parsimonious in position 0) 
    FinalImportance = ImpOfPossibleNestedModels[model,:]
    return FinalImportance    


#Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


#MAIN CODE


if __name__ == "__main__":
    #mp.freeze_support()
    #Import data
    start = time()
    
    args = argparse.ArgumentParser()
    #args.add_argument('input', metavar='input', nargs='+',help='path of the input')    
    args.add_argument('Ntrees', metavar='Ntrees', type=int, nargs='+',help='n of trees')
    args.add_argument('BOiterations', metavar='BOiterations', type=int, nargs='+',help='Iteration of Optimization')
    
    #
    #    
    #    
    error_message = ""
    #    
    parsed_args = args.parse_args() 
    #file_address = parsed_args.input[0]
    Ntrees = parsed_args.Ntrees[0]
    BOiterations = parsed_args.BOiterations[0]
    path = 'C:\\Users\\ibargiotas\\Documents\\MATLAB\\Two-sample AUC maximization\\Parkinsons\\'
    #df = pd.read_excel(path+'OpenEyes.xlsx')
    #df = pd.read_excel(file_address)
    df = pd.read_excel('input_0.xlsx')    
    #    	 
    #    
    #Labels in the Last column 
    Features = np.array(df)
    X = Features[:,:-1]
    Y = Features[:,-1]
    #    
    #    
    #Set Parameters' and ranges to be otimized
    maxMinLS = X.shape[0]
    maxnumPTS = X.shape[1]
    #    
    min_samples_leaf  =  np.arange(1,int(np.round(0.3*maxMinLS)+1))
    max_features =  np.arange(1,int(np.round(0.7*maxnumPTS)+1))
    #    
    params = dict(min_samples_leaf=min_samples_leaf,max_features=max_features)
    #    
    #    
    #Random Forest Classifier and Random or Grid Optimization
    #Ntrees = 2*Y.size
    #if Ntrees>50:
    #    Ntrees=50
    #
    cv = [(slice(None), slice(None))]
    clf = RandomForestClassifier(n_estimators = Ntrees,oob_score=True)
    grid_rf = GridSearchCV(estimator=clf, param_grid = params, scoring = OOBAUC, cv = cv, n_jobs=-1,verbose = 1)
    #clf = RandomForestClassifier(n_estimators = Ntrees ,oob_score=True)
    #grid_rf = GridSearchCV(clf, param_grid = params, scoring = 'roc_auc', cv = 5, n_jobs=-1,verbose = 1)
    #grid_rf = RandomizedSearchCV(clf, params,scoring = 'roc_auc', cv = 5, n_jobs=8,verbose = 1, n_iter = 50)
    #    
    #Evaluation
    
    grid_rf.fit(X, Y)

    #print("GridSearchCV or RandomizedSearchCV took %.2f seconds for %d candidate parameter settings."
    #      % (Timepvalue, len(grid_rf.cv_results_['params'])))
    #report(grid_rf.cv_results_)
    #    
    params = grid_rf.best_params_
    Starclf = RandomForestClassifier(n_estimators = Ntrees,max_features = params['max_features'],min_samples_leaf = params['min_samples_leaf'],oob_score=True)
    Starclf.fit(X, Y)
    #Starclf = grid_rf.best_estimator_
    #    
    #Manual greedy approach with OOB cross validation
    #Starclf = GreedySearchAUCMaxOOB(min_samples_leaf,max_features)
    #    
    #    
    #Take OOB
    Scores = Starclf.oob_decision_function_[:,-1]
    #    
    #U statistics
    U = mww(Scores[Y==grid_rf.classes_[0]],Scores[Y==grid_rf.classes_[1]],alternative='less')
    #Result 1 - p value estimations
    pvalue = U.pvalue
    #    
    #Result 2 - Predictor Importance
    importance = InterpretImportance1(X,Y,grid_rf.best_params_,50,0)
    #
    #importance = Starclf.feature_importances_
    #
    #Figure of sorted Importance
    Imp_ind = (-importance).argsort()
    x_pos = np.arange(len(importance))
    plt.bar(x_pos, importance[Imp_ind])
    Variable_names = df.columns[0:-1]
    Variable_names = Variable_names[Imp_ind] 
    plt.xticks(x_pos, tuple(Variable_names),rotation=90)
    #    
    #    
    ######################################################################
    #    
    #OPTIONAL : More accurate predictor importance but very heavy coding
    #iterations = 20
    #importance = InterpretImportance1(X,Y,grid_rf.best_params_,iterations)
    # print(time() - start)
    end = time()
    elapsedTime = end-start
    with open('elapsedTime.txt', 'w') as et:
        et.write(str(elapsedTime))
    with open('pvalue.txt', 'w') as pv:
        pv.write(str(pvalue))
    with open('importance.txt', 'w') as imp:
        imp.write(str(importance))
    plt.savefig("output_importance.png")
    
    
