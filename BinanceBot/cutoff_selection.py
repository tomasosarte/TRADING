
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="Solver terminated early.*")
warnings.filterwarnings("error", category=RuntimeWarning)

def find_best_params(folds):
    """
    This function is used to find what is the best model, with the best possible params. The function evaluates
    some models with some parameters to test wich is the best and stores all the results in a csv file.
    Arguments: 
        * folds: A list of dataframes represnting the folds
    """

    # All models taken into account
    clf1 = SVC(probability=True, max_iter=10000)
    clf2 = LogisticRegression(max_iter=3500)
    clf3 = XGBClassifier()

    # Parametrization of the models
    param1 = {}
    param1['classifier__C'] = [0.1, 0.2, 0.5, 1, 2, 5, 10,100]
    param1['classifier__kernel'] = ["rbf", "linear"]
    param1['classifier'] = [clf1]

    param2 = {}
    param2['classifier__C'] = [0.1, 0.5, 1, 2, 10, 1000, 10000]
    param2['classifier__penalty'] = ['l1', 'l2']
    param2['classifier__solver'] = ['liblinear']
    param2['classifier'] = [clf2]

    param3 = {}
    param3['classifier__gamma'] = [0.5, 1, 1.5, 2, 5]
    param3['classifier__colsample_bytree'] = [0.6, 0.8, 1.0]
    param3['classifier__max_depth'] = [3, 5, 4]
    param3['classifier__min_child_weight'] = [1, 5, 10]
    param3['classifier__subsample'] = [0.6, 0.8, 1]
    param3['classifier'] = [clf3]

    models = []
    mean_specificties = []

    for c in param1['classifier__C']:
        for kernel in param1['classifier__kernel']:
            estimator = SVC(C=c, kernel=kernel, probability=True, max_iter=1000, degree=5)
            model, mean_spec = get_params_10fCV(folds,estimator)
            models.append(model)
            mean_specificties.append(mean_spec)


    for c in param2['classifier__C']:
        for penalty in param2['classifier__penalty']:
            for solver in param2['classifier__solver']:
                estimator = LogisticRegression(C=c, penalty=penalty, solver=solver, max_iter=3500)
                model, mean_spec = get_params_10fCV(folds,estimator)
                models.append(model)
                mean_specificties.append(mean_spec)

    for gamma in param3['classifier__gamma']:
        for colsample_bytreee in param3['classifier__colsample_bytree']:
            for max_depth in param3['classifier__max_depth']:
                for min_child_weight in param3['classifier__min_child_weight']:
                    for subsample in param3['classifier__subsample']:
                        estimator = XGBClassifier(gamma=gamma, colsample_bynode=colsample_bytreee,
                                                  min_child_weight=min_child_weight, subsample=subsample,
                                                  max_depth=max_depth)
                        model, mean_spec = get_params_10fCV(folds,estimator)
                        models.append(model)
                        mean_specificties.append(mean_spec)
    
    # Saving data in csv file
    dictionary = {'selected_model':models,'mean_Spec_Sens=1':mean_specificties}
    modelsAndSpecificities = pd.DataFrame(dictionary)
    modelsAndSpecificities.to_csv("results_quality10f.csv")

    return True

def get_params_10fCV(folds, model):
    """
    10-fold CrossValidation (10-f CV).
    Arguments: 
        * folds: A list of dataframes represnting the folds
        * model: A machine learning model with hyperparametrization
    Output:
        * str(model): A string containing the model ans it's parametrization
        * mean_spec: A float containing the mean specificity taking into account the 
        specificty in each fold.
    """
    results = {'cutoff':[],'spec':[]}
    for i in range(len(folds)):
        rand = random.randint(1, 100)
        X = folds[i].drop('Target',axis=1)
        y =  folds[i]['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rand)
        model = model.fit(X_train, y_train)
        prediction = model.predict_proba(X_test)
        #-----------------------------------------
        probabilityOf1 = []
        for predict in prediction: probabilityOf1.append(predict[1])
        #-----------------------------------------
        cutoff, spec = find_cutoff(probabilityOf1, y_test)
        results['cutoff'].append(cutoff)
        results['spec'].append(spec)
        # plot_hist(probabilityOf1, y_test,cutoff, i)

    print('-'*25)
    print('MODEL:')
    print(model)
    print('Mean specificity:')
    mean_spec = np.mean(results['spec'])
    print(mean_spec)
    print('All cutoffs:')
    print(results['cutoff'])
    print('-'*25)

    return str(model), mean_spec

def plot_hist(scorings,Target,cutoff,i):
    """
    This function is used to save plot the given scorings divided by the Target, knowing the cutoff.
    Arguments: 
        * scorings: The probability that the number is a 1 (a negative data)
        * Target: The real label that correponds to the data
        * cutoff: A float containing the maximum threshold that preserves the sensibility to 1 for 1 fold
        * i: An integer determining wich fold are we using
    """
    dictionary = {'scoring':scorings,'quality':Target}
    df = pd.DataFrame(dictionary)
    A = df.loc[df['quality'] == 0, 'scoring']
    B = df.loc[df['quality'] == 1, 'scoring']
    plt.hist(A, alpha=0.5, label='Good', bins=10)
    plt.hist(B, alpha=0.5, label='Bad', bins=10)

    plt.title(f'Predicted values at iteration {i}. Cutoff = {cutoff}')
    plt.xlabel('Scoring')
    plt.ylabel('Frequency')

    # add legend
    plt.legend(title='Quality')
    plt.savefig(f'scorings/scorings_iteration{i}.png')
    plt.clf()
    return True


def find_cutoff(scoring, label):
    """
    This function given a scoring and label finds what's the best cutoff that preserves to 1 
    the sensibility and returns the cutoff and the respective secificty.
    Arguments: 
        * scoring: The probability that the number is a 1 (a negative data)
        * label: The real label that correponds to the data
    Output:
        * cutoff: A flotat containing the maximum threshold that preserves the sensibility to 1 for 1 fold
        * spec: A float containing the specificity assigned to the treshold
    """
    for threshold in np.linspace(0,1,1001):
        prediction = np.where(scoring < threshold, 0, 1)
        sens = recall_score(label.values,prediction)
        confusion_m = confusion_matrix(label, prediction)
        try:spec = confusion_m[0, 0] / sum(confusion_m[0,])
        except RuntimeWarning: spec = 0    
        if sens < 1:
            cutoff = threshold-0.001
            prediction = np.where(scoring < threshold, 0, 1)
            confusion_m = confusion_matrix(label, prediction)
            try:spec = confusion_m[0, 0] / sum(confusion_m[0,])
            except RuntimeWarning: spec = 0
            return cutoff, spec
        else:
            pass

if __name__ == '__main__':

    X= "INTRODUCIR X DONDE Target: 0 o 1 (buena/mala)"
    find_best_params(X)