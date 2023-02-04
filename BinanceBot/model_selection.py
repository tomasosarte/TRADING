import os
import numpy as np
import pandas as pd
import random
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def find_best_params(folds):
    """
    This function is used to find what is the best model, with the best possible params. The function evaluates
    some models with some parameters to test wich is the best and stores all the results in a csv file.
    Arguments: 
        * folds: A list of dataframes represnting the folds
    Output: 
        * 
    """

    # All models taken into account
    # --> XGBClassifier()

    # Parametrization of the models
    param1 = {
        # 'gamma': [0.5],
        # 'learning_rate': [0.01],
        # 'n_estimators': [25],
        # 'max_depth': [2],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2],
        'n_estimators': [25, 50, 100, 150],
        'max_depth': [2, 3, 4, 5, 6],
    }

    models = []
    mean_accuracy = []

    for gamma in param1['gamma']:
        for max_depth in param1['max_depth']:
            for n_estimators in param1['n_estimators']:
                for learning_rate in param1['learning_rate']:
                    rand = random.randint(1, 100)
                    estimator = XGBClassifier(gamma=gamma, n_estimators=n_estimators,
                                              learning_rate=learning_rate, max_depth=max_depth,
                                              objective = 'binary:logistic', eval_metric = 'auc', 
                                              booster = 'gbtree', subsample=0.8, colsample_bytree= 1,
                                              use_label_encoder = False, random_state=rand)
                    model, mean_acc = get_params_10fCV(folds,estimator)
                    models.append(model)
                    mean_accuracy.append(mean_acc)
                    #--------------------------------------------------------------------------------------------------------------------
                    print(f'Parameters: Gamma={gamma}, max_depth:{max_depth}, n_estimators:{n_estimators}, learning_rate:{learning_rate}')
                    print(f'MODEL: {model}')
                    print(f'Mean accuracy: {mean_acc}')        
                    print('-'*25)
                    #---------------------------------------------------------------------------------------------------------------------
                    
                
                

    # Saving data in csv file
    dictionary = {'selected_model':models,'mean_accuracy':mean_accuracy}
    modelsAndSpecificities = pd.DataFrame(dictionary)
    modelsAndSpecificities.to_csv("results_quality10fCV.csv")
    i = modelsAndSpecificities['mean_accuracy'].idxmax()
    best_model = modelsAndSpecificities.iloc[i]

    print()
    print()
    print('The best model is:')
    print(best_model)
    print()
    print()

    return best_model

def get_params_10fCV(folds, model):
    """
    10-fold CrossValidation (10-f CV).
    Arguments: 
        * folds: A list of dataframes represnting the folds
        * model: A machine learning model with hyperparametrization
    Output:
        * str(model): A string containing the model ans it's parametrization
        * mean_accuracy: A float containing the mean accuracy of the 10fCV.
    """
    results = {'accuracy':[]}

    for i in range(len(folds)):
        partition = int(len(folds[i]) * 0.9)

        train = folds[i].iloc[:partition]
        test = folds[i].iloc[partition:]

        X_train = train.drop('target',axis=1)
        X_test =  test.drop('target',axis=1)
        y_train = train['target']
        y_test =  test['target']

        model = model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        accuracy = accuracy_score(y_test, prediction)

        results['accuracy'].append(accuracy)


        mean_accuracy = np.mean(results['accuracy'])

    return str(model), mean_accuracy

if __name__ == '__main__':

    #Exporting folds
    folder = 'folds'
    folds = []
    for file in os.listdir(folder):
        fold = pd.read_csv(f'{folder}/{file}')
        folds.append(fold)

    print(f"FOLD SHAPE: {folds[0].shape}")
    print()
    print(fold)

    # Find best params and saving data
    find_best_params(folds)

