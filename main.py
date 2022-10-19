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
from sklearn import metrics


#OOBAUC calculates the AUC from the OOBscores and use it as maximization criterion
def OOBAUC(estimator,X,y):
    PositiveOOBPosterior = np.zeros([y.shape[0],BOiterations])*np.nan 
    AUC = np.zeros([BOiterations])*np.nan
    for j in np.arange(0,BOiterations):
        "Create the forest"
        estimator.fit(X, y)
        clf = estimator
        "Take the OOB posteror probabilities"
        OOBPosterior = clf.oob_decision_function_
        PositiveOOBPosterior[:,j] = OOBPosterior[:,-1]
        
        "Calculate AUC"
        PositiveOOB = OOBPosterior[:,-1]#PositiveOOBPosterior.mean(axis=1)
        fpr, tpr, thresholds = metrics.roc_curve(y, PositiveOOB, pos_label=1)
        AUC[j] = metrics.auc(fpr, tpr)
        #n1 = np.count_nonzero(y==clf.classes_[-1])
        #n2 = y.size - n1;
        #TiedRank = rankdata(PositiveOOB)
        #W1 = np.sum(TiedRank[y == clf.classes_[-1]])
        "W2 = np.sum(TiedRank[Y == clf.classes_[0]])"
        #AUC[j] = (W1-n1*(n1+1)/2)/(n1*n2)
    
    "Error to be maximized"
    Final_error = np.mean(AUC) - np.std(AUC)
    
    #"Calculate AUC"
    #PositiveOOB = PositiveOOBPosterior.mean(axis=1)
    #n1 = np.count_nonzero(y==clf.classes_[-1])
    #n2 = y.size - n1;
    #TiedRank = rankdata(PositiveOOB)
    #W1 = np.sum(TiedRank[y == clf.classes_[-1]])
    #"W2 = np.sum(TiedRank[Y == clf.classes_[0]])"
    #AUC = (W1-n1*(n1+1)/2)/(n1*n2)
    #
    #"Error to be maximized"
    #Final_error = AUC
    #
    return Final_error    


# Function of Predictor Importance (see Genuer et al 2010, Bargiotas et al 2020, Bargiotas et al 2021)
def InterpretImportance1(X,Y,params,iters,complexity):
    
    #Local Function (for parallel processing) to calculate initial importance tendencies from OOB importance    
    def InitImport(clf,X,Y,i):
        clf.fit(X, Y)
        importance = clf.feature_importances_
        return importance
    
    #Local Function (for parallel processing) to calculate oob decision, importance and  AUC   
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
    #clf = RandomForestClassifier(n_estimators = Ntrees,max_depth = params['max_depth'],max_features = params['max_features'],min_samples_leaf = params['min_samples_leaf'],min_samples_split = params['min_samples_split'],oob_score=True)
    clf = RandomForestClassifier(n_estimators = Ntrees,max_features = params['max_features'],min_samples_leaf = params['min_samples_leaf'],oob_score=True)
    #
    #Run InitImport in parallel
    importance = Parallel(n_jobs=-1)(delayed(InitImport)(clf,X,Y,i) for i in np.arange(0,iters))
    AvImp = np.mean(importance, axis = 0)
    StdImp = np.std(importance, axis = 0)
    #
    #Definition of Threshold (Check Genuer 2010)    
    DescendAvImpIndex =  (-AvImp).argsort()
    DescendAvImpValue =  AvImp[DescendAvImpIndex]
    AscendStdImpIndex = StdImp.argsort()
    AscendStdImpValue = StdImp[AscendStdImpIndex]
    
    #Create a linear regression with Sorted STDs (Check Genuer 2010) 
    rgf = RandomForestRegressor(n_estimators = Ntrees,oob_score=True)
    rgf.fit(np.arange(0,AscendStdImpValue.size).reshape(-1, 1), AscendStdImpValue.reshape(-1, 1).ravel())
    
    oobThres = rgf.oob_prediction_
    Threshold = oobThres.min()
    #  
    #Keep the dimensions that passes the significance Threshold
    PassedVariables = DescendAvImpIndex[DescendAvImpValue>Threshold]
    #
    #Keep model depending on its mean and std of AUC 
    jsize = PassedVariables.size
    MeanAUC = np.zeros(jsize)
    StdAUC = np.zeros(jsize)
    ImportanceCell = np.zeros([jsize,X.shape[1]])
    
    for j in np.arange(0,PassedVariables.size):
        Xtemp = X[:,DescendAvImpIndex[0:j+1]]
        
        if Xtemp.shape[1]<params['max_features']:
            clf = RandomForestClassifier(n_estimators = Ntrees,max_depth = params['max_depth'],max_features = Xtemp.shape[1],min_samples_leaf = params['min_samples_leaf'],min_samples_split = params['min_samples_split'],oob_score=True)
            #clf = RandomForestClassifier(n_estimators = Ntrees,max_features = Xtemp.shape[1],min_samples_leaf = params['min_samples_leaf'],oob_score=True)
        else:
            clf = RandomForestClassifier(n_estimators = Ntrees,max_depth = params['max_depth'],max_features = params['max_features'],min_samples_leaf = params['min_samples_leaf'],min_samples_split = params['min_samples_split'],oob_score=True)    
            #clf = RandomForestClassifier(n_estimators = Ntrees,max_features = params['max_features'],min_samples_leaf = params['min_samples_leaf'],oob_score=True)    
        
        #For every meaningful combination, calculate AUC and importance
        imptemp = np.zeros([iters,Xtemp.shape[1]])
        AUC = np.zeros(iters)
        res = Parallel(n_jobs=-1)(delayed(SecondImport)(clf,Xtemp,Y,i) for i in np.arange(0,iters))
        
        #Save Mean(AUC) std(AUC) and Mean Importance 
        imptemp = np.array([item[0] for item in res])
        AUC = np.array([item[1] for item in res])
        MeanAUC[j] = AUC.mean()
        StdAUC[j] = AUC.std()
        ImportanceCell[j,DescendAvImpIndex[0:j+1]] =  imptemp.mean(axis = 0)
    #
    #Find the most parsimonious model which is higher than MaxAUC - StdAUC[ArgMaxAUC]
    ArgMaxAUC = MeanAUC.argmax()
    MaxAUC = MeanAUC.max()
    ThresAUC = MaxAUC - StdAUC[ArgMaxAUC] 
    ImpOfPossibleNestedModels = ImportanceCell[MeanAUC>ThresAUC,:]
    #
    #from the selected complexity model (usually the most parsimonious in position 0) 
    FinalImportance = ImpOfPossibleNestedModels[complexity,:]
    return FinalImportance    

# MAIN CODE

if __name__ == "__main__":
    start = time()
    #Inputs from demo
    args = argparse.ArgumentParser()
    args.add_argument('Ntrees', metavar='Ntrees', type=int, nargs='+',help='n of trees')
    args.add_argument('BOiterations', metavar='BOiterations', type=int, nargs='+',help='Iteration of Optimization')
    #  
    error_message = ""
    #    
    parsed_args = args.parse_args() 
    Ntrees = parsed_args.Ntrees[0]
    BOiterations = parsed_args.BOiterations[0]
    df = pd.read_excel('input_0.xlsx')    
    #    	   
    #Labels in the Last column 
    Features = np.array(df)
    X = Features[:,:-1]
    Y = Features[:,-1]
    #    
    #    
    #Set Parameters' and ranges to be optimized
    maxMinLS = X.shape[0]
    maxnumPTS = X.shape[1]
    #
    # Number of features to consider at every split
    max_features = [int(x) for x in np.linspace(1,np.ceil(np.round(0.7*maxnumPTS)),num = 10)]#np.arange(1,int(np.round(0.7*maxnumPTS)+1))#
    # Maximum number of levels in tree
    max_depth = np.arange(3,8,2)
    # Minimum number of samples required to split a node
    min_samples_split = np.linspace(0, 0.2, num = 3)
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [int(x) for x in np.linspace(1, int(np.round(0.1*maxMinLS)+1), num = 10)]#np.arange(1,int(np.round(0.2*maxMinLS)+1))#
    # Create the random grid
    random_grid = {'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    
    #Grid search with 1fold as CV schema (reminder:we use the OOB decision score for Validation) (see Silke Janitza and Roman Hornung 2018)
    cv = [(slice(None), slice(None))]
    clf = RandomForestClassifier(n_estimators = Ntrees,oob_score=True)
    grid_rf = GridSearchCV(estimator=clf, param_grid = random_grid, scoring = OOBAUC, cv = cv, n_jobs=-1,verbose = 1)
    
    #Evaluation
    grid_rf.fit(X, Y)
    
    params = grid_rf.best_params_
    
    #Creation of Star Model
    Starclf = RandomForestClassifier(n_estimators = Ntrees,max_depth = params['max_depth'],max_features = params['max_features'],min_samples_leaf = params['min_samples_leaf'],min_samples_split = params['min_samples_split'],oob_score=True)
    #Starclf = RandomForestClassifier(n_estimators = Ntrees,max_features = params['max_features'],min_samples_leaf = params['min_samples_leaf'],oob_score=True)
    Starclf.fit(X, Y)
    
    #Take OOB probabilities of POSITIVE class 
    Scores = Starclf.oob_decision_function_[:,-1]
    #    
    #U statistics - alternative = less because we took scores of POSITIVE class 
    U = mww(Scores[Y==Starclf.classes_[0]],Scores[Y==Starclf.classes_[1]],alternative='less')
    #Result 1 - p value estimations
    pvalue = U.pvalue
    pvalue = np.append(pvalue,grid_rf.best_score_)
    #    
    #Result 2 - Predictor Importance
    final_importance = InterpretImportance1(X,Y,params,50,0)
    starmodel_importance = Starclf.feature_importances_
    #
    #Result 3 - Size Effect
    N0 = len(Scores[Y==Starclf.classes_[0]])
    N1 = len(Scores[Y==Starclf.classes_[1]])
    N = N1+N0
    
    #Cohen's D
    Scores0 = Scores[Y==Starclf.classes_[0]]
    Scores1 = Scores[Y==Starclf.classes_[1]]
    s0 = np.nanstd(Scores0)
    s1 = np.nanstd(Scores1)
    s = np.sqrt(((N0 - 1)*s0**2 + (N1 - 1)*s1**2)/(N - 2))
    CohenD = np.abs((np.nanmean(Scores1) - np.nanmean(Scores0))/s);
    # Biserial Size effect
    Biserial = np.sqrt(CohenD**2*N1*N0/(N*(N-2)+CohenD**2*N1*N0))#Diana Kornbrot 2014, eq.3
    size_effect = np.append(CohenD,Biserial)
    #rho = np.corrcoef(Scores, Y)
    #Pearson = rho[0,1] 
    #size_effect = np.append(size_effect,Pearson)
    #    
    # Exports
    ######################################################################
    #  
    #Figure of Importance
    x_pos = np.arange(len(final_importance))
    plt.bar(x_pos, final_importance)
    Variable_names = df.columns[0:-1]
    Variable_names = Variable_names 
    plt.xticks(x_pos, tuple(Variable_names),rotation=90)
    plt.savefig("final_importance.png")
    
    #Imp_ind = (-starmodel_importance).argsort()
    plt.bar(x_pos, starmodel_importance)
    plt.xticks(x_pos, tuple(Variable_names),rotation=90)
    plt.savefig("starmodel_importance.png") 
    #    
    end = time()
    elapsedTime = end-start
    with open('elapsedTime.txt', 'w') as et:
        et.write(str(elapsedTime))
    with open('pvalue.txt', 'w') as pv:
        pv.write(str(pvalue))
    with open('size_effect.txt', 'w') as se:
        se.write(str(size_effect))
    with open('params.txt', 'w') as hp:
        hp.write(str(params))    
    
    
