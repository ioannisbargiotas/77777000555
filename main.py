#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from time import time
import argparse
import openpyxl
import numpy as np
import sklearn as sk
import pandas as pd
from scipy.stats import wilcoxon as wilc
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
    #clf = RandomForestClassifier(n_estimators = Ntrees,max_depth = params['max_depth'],max_features = params['max_features'],min_samples_leaf = params['min_samples_leaf'],min_samples_split = params['min_samples_split'],oob_score=True)
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
            clf = RandomForestClassifier(n_estimators = Ntrees,max_depth = params['max_depth'],max_features = Xtemp.shape[1],min_samples_leaf = params['min_samples_leaf'],min_samples_split = params['min_samples_split'],oob_score=True)
            #clf = RandomForestClassifier(n_estimators = Ntrees,max_features = Xtemp.shape[1],min_samples_leaf = params['min_samples_leaf'],oob_score=True)
        else:
            clf = RandomForestClassifier(n_estimators = Ntrees,max_depth = params['max_depth'],max_features = params['max_features'],min_samples_leaf = params['min_samples_leaf'],min_samples_split = params['min_samples_split'],oob_score=True)    
            #clf = RandomForestClassifier(n_estimators = Ntrees,max_features = params['max_features'],min_samples_leaf = params['min_samples_leaf'],oob_score=True)    
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
    
    Inst_id = Features[:,0]
    X = Features[:,1:-1]
    Y = Features[:,-1]
    #    
    Classes = np.unique(Y)
    X0 = X[Y==Classes[0],:]
    Inst_id0 = Inst_id[Y==Classes[0]]
    X1 = X[Y==Classes[1],:]
    Inst_id1 = Inst_id[Y==Classes[1]]
    #
    N = X0.shape[0]    
    
    Instances = np.arange(0,N)
    tempBeta = np.zeros((N,X0.shape[1]))
    tempscore0 = np.zeros((N,N))*np.nan
    tempscore1 = np.zeros((N,N))*np.nan
    score0 = np.zeros((N,1))*np.nan
    score1 = np.zeros((N,1))*np.nan
    Models = np.zeros((N,X0.shape[1]+1))*np.nan
    
    Drop_values = np.ones((N,N))*np.nan
    N_classifiers = int(np.round((Instances.shape[0]-1)/0.68))
    
    kernel='linear'
    
    if kernel=='linear':
        for i in Instances:
            #select pair of dataset
            Inst_id0[i] 
            XX0 = X0[Inst_id0==Inst_id0[i],:]
            XX1 = X1[Inst_id1==Inst_id0[i],:]

            XX = np.concatenate((XX0, XX1),axis=0)
        
            Response0 = -1
            Response1 = 1
            Response = np.concatenate(([Response0], [Response1]),axis=0)
        
            #Train svm
            clf = sk.svm.SVC(kernel=kernel,probability=True)
            clf.fit(XX, Response)
            
            Models[i,:] = np.append(clf.coef_, clf.intercept_)
        
            Index = Instances
            Index = np.delete(Index,i)
            
            #choose randomly
            Choices = np.random.choice(Index,N_classifiers)
            Drop_values[i,Choices] = 1
            
            
            temp = clf.predict_proba(X0[Index,:])
            tempscore0[Index,i] = temp[:,-1]
            temp = clf.predit_proba(X1[Index,:])
            tempscore1[Index,i] = temp[:,-1]
            tempBeta[i,:] = clf.coef_

            # model = []
            # for i in Instances:
                
            #     XX0 = X0[i,:]
            #     XX1 = X1[i,:]
            #     XX = np.concatenate(([XX0.T], [XX1.T]),axis=0)
            
            #     Response0 = -1
            #     Response1 = 1
            #     Response = np.concatenate(([Response0], [Response1]),axis=0)
            
            #     #Train svm
            #     clf = sk.svm.SVC(kernel=kernel,probability=True)
            #     clf.fit(XX, Response)
            #     model.append(clf)
            
            # N_classifiers = int(np.round(Instances.shape[0]/0.68))
            # for i in Instances
            #     XX0 = X0[i,:]
            #     XX1 = X1[i,:]
            #     XX = np.concatenate(([XX0.T], [XX1.T]),axis=0)
            
            #     Response0 = -1
            #     Response1 = 1
            #     Response = np.concatenate(([Response0], [Response1]),axis=0)
                
            #     Index = Instances
            #     Index = np.random.choice(Instances,N_classifiers)
                
            #     scoreX0 = np.zeros((N,1))*np.nan
            #     scoreX1 = np.zeros((N,1))*np.nan
            #     for ii in Index:
            #         clf = model[ii]
            #         scoreX0[ii,0] = clf.
            #         tempscore0[Index,0] = temp
                
            #     temp = clf.decision_function(X0[Index,:])
            #     tempscore0[Index,i] = temp
            #     temp = clf.decision_function(X1[Index,:])
            #     tempscore1[Index,i] = temp #temp[:,-1]
            #     tempBeta[i,:] = clf.coef_  
    elif kernel=='rbf':
        for i in Instances:
            Inst_id0[i] 
            XX0 = X0[Inst_id0==Inst_id0[i],:]
            XX1 = X1[Inst_id1==Inst_id0[i],:]
            #Generate normal datasets with centers given by each instance
            d = XX1-XX0
            cov_all = np.abs(np.diag(d)/10)
            XX0 = np.random.multivariate_normal(X0[i,:].T, cov_all, N)
            XX1 = np.random.multivariate_normal(X1[i,:].T, cov_all, N)
            XX = np.concatenate((XX0, XX1),axis=0)
            
            Response0 = -np.ones((N, 1))
            Response1 = np.ones((N, 1))
            Response = np.concatenate((Response0, Response1),axis=0)
            Response = Response.ravel()
            
            #Train svm
            clf = sk.svm.SVC(kernel=kernel,probability=True)
            clf.fit(XX, Response)
            
            Index = Instances
            Index = np.delete(Index,i)
            
            temp = clf.decision_function(X0[Index,:])
            tempscore0[Index,i] = temp
            temp = clf.decision_function(X1[Index,:])
            tempscore1[Index,i] = temp #temp[:,-1]
            XX_test = np.concatenate((X0[Index,:], X1[Index,:]),axis=0)
            Response_Test = np.delete(Response,[i,i+N])
            r = permutation_importance(clf,XX_test, Response_Test,scoring = 'roc_auc',n_repeats=10)
            for ii in r.importances_mean.argsort()[::-1]:
                if r.importances_mean[ii] - 2 * r.importances_std[ii] > 0:
                    tempBeta[i,ii] = r.importances_mean[ii] 

    
    final_importance = np.nanmean(tempBeta,axis = 0)
    Scores0 = np.nanmean(tempscore0,axis = 1)
    Scores1 = np.nanmean(tempscore1,axis = 1)
    
    d = Scores1 - Scores0
    w, pvalue = wilc(d, alternative='greater')
    
    #Result 3 - Size Effect
    s0 = np.nanstd(Scores0)
    s1 = np.nanstd(Scores1)
    s = np.sqrt(((N - 1)*s0**2 + (N - 1)*s1**2)/(2*N - 2))
    CohenD = np.abs((np.nanmean(Scores1) - np.nanmean(Scores0))/s);
    # Biserial Size effect
    Biserial = np.sqrt(CohenD**2*N*N/(2*N*(2*N-2)+CohenD**2*N*N))#Diana Kornbrot 2014, eq.3
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
    Variable_names = df.columns[1:-1]
    Variable_names = Variable_names 
    plt.xticks(x_pos, tuple(Variable_names),rotation=90)
    plt.savefig("final_importance.png")
    
    #Imp_ind = (-starmodel_importance).argsort()
    plt.bar(x_pos, final_importance)
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
    
