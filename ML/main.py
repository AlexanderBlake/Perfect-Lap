'''
This module conducts machine learning experiments to find the best model and
the most important features.
'''

from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from sklearn import tree
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split

RANDOM = 42
SCORE = 'neg_root_mean_squared_error'

def preprocess_data(data):
    '''
    This function performs some operation.

    Parameters:
    data (int): Description of param1.

    Returns:
    bool: Description of the return value.
    '''
    data = data.groupby(['Race 1 DateTime'])
    scaler = MinMaxScaler()

    scaled_groups = []
    for _, group in data:
        group['Race 1 Fastest Time'] = scaler.fit_transform(
            group['Race 1 Fastest Time'].values.reshape(-1, 1))
        group['Race 1 Gap to Leader'] = scaler.fit_transform(
            group['Race 1 Gap to Leader'].values.reshape(-1, 1))
        scaled_groups.append(group)

    data = pd.concat(scaled_groups)

    for col in ['Race 1 Kart Num', 'Final Kart Num', 'Driver Name']:
        one_hot_encoded = pd.get_dummies(data[col], prefix=col)
        data = pd.concat([data, one_hot_encoded], axis=1)

    x_data = data[['Final Starting Pos', 'Race 1 Fastest Time', 'Race 1 Gap to Leader'] +
                  list(data.filter(regex='Driver Name_').columns)]
    y_data = data['Final Finish Pos']

    return x_data, y_data

def conduct_experiments(x_data, y_data) -> list:
    '''
    This function performs some operation.

    Parameters:
    x_data (int): Description of param1.
    y_data (str): Description of param2.

    Returns:
    list: Description of the return value.
    '''
    results = []

    models = [(KNeighborsRegressor(), 'KNN'),
              (LinearSVR(random_state=RANDOM, dual='auto', max_iter=8000), 'Linear SVM')]
    for model, name in models:
        results.append((-1 * cross_val_score(model, x_data, y_data, scoring=SCORE).mean(), name))

    for depth in range(1, 9):
        results.append((-1 * cross_val_score(
            RandomForestRegressor(random_state=RANDOM, max_depth=depth), x_data, y_data,
            scoring=SCORE).mean(), 'Random Forest Max Depth: ' + str(depth)))

    for kern in ['rbf', 'linear', 'poly', 'sigmoid']:
        results.append((-1 * cross_val_score(NuSVR(kernel=kern), x_data, y_data,
                                             scoring=SCORE).mean(), 'NU SVM Kernel: ' + kern))
        results.append((-1 * cross_val_score(SVR(kernel=kern), x_data, y_data,
                                             scoring=SCORE).mean(), 'SVM Kernel: ' + kern))

    k_fold = KFold()
    temp = 0
    for _, (_, test_index) in enumerate(k_fold.split(x_data)):
        temp += sqrt(mean_squared_error(np.full(len(test_index), 6), y_data.iloc[test_index]))
    results.append((temp / 5, 'BASELINE: Median Guessing'))

    '''
    for ac in ['relu', 'identity', 'logistic', 'tanh']:
        for so in ['lbfgs', 'sgd', 'adam']:
            for lr in ['constant', 'invscaling', 'adaptive']:
                if so == 'sgd':
                    results.append((-1 * cross_val_score(
                        MLPRegressor(random_state=RANDOM, max_iter=6400, activation=ac,
                                     solver=so, learning_rate=lr), x_data, y_data,
                                     scoring='neg_root_mean_squared_error').mean(),
                                     'Neural Network ' + ac + ' ' + so + ' ' + lr))
                else:
                    results.append((-1 * cross_val_score(
                        MLPRegressor(random_state=RANDOM, max_iter=6400, activation=ac,
                                     solver=so), x_data, y_data,
                                     scoring='neg_root_mean_squared_error').mean(),
                                     'Neural Network ' + ac + ' ' + so + ' ' + lr))
                    break
    '''

    results.sort()
    return results

def find_importances(x_data, regr) -> list:
    '''
    This function performs some operation.

    Parameters:
    x_data (int): Description of param1.
    regr (str): Description of param2.

    Returns:
    list: Description of the return value.
    '''

    results = []
    importances = {'Driver Name': 0, 'Final Kart Num': 0, 'Track': 0}

    for i, col in enumerate(x_data.columns):
        if 'Driver Name' in col:
            importances['Driver Name'] += regr.feature_importances_[i]
        elif 'Final Kart Num' in col:
            importances['Final Kart Num'] += regr.feature_importances_[i]
        elif 'Track' in col:
            importances['Track'] += regr.feature_importances_[i]
        else:
            results.append((regr.feature_importances_[i], col))

    for key, value in importances.items():
        results.append((value, key))

    results.sort(reverse=True)
    return results

def display_table(header: list[str], results: list) -> None:
    '''
    This function prints a pretty table.

    Parameters:
    header (list[str]): Description of param1.
    results (list): Description of param2.

    Returns:
    None
    '''
    pt_table = PrettyTable(header)

    for i, result in enumerate(results):
        value, name = result
        if 'BASELINE' in name:
            pt_table.add_row(['','',''])
        pt_table.add_row([i + 1, name, value])
        if 'BASELINE' in name:
            pt_table.add_row(['','',''])

    print(pt_table)
    print()

def main() -> None:
    '''
    This function runs the program.

    Parameters:
    None

    Returns:
    None
    '''

    data = pd.read_csv('VSK - Sheet1.csv')
    x_data, y_data = preprocess_data(data)
    results = conduct_experiments(x_data, y_data)

    display_table(['Rank', 'Model Description', 'Root Mean Square Error (RMSE)'], results)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=RANDOM)
    regr = RandomForestRegressor(random_state=RANDOM, max_depth=2).fit(x_train, y_train)

    y_pred = regr.predict(x_test)

    x_test['predicted'] = y_pred
    x_test['actual'] = y_test
    x_test['error'] = y_test - y_pred
    pd.set_option('display.max_columns', None)
    #print(X_test)

    plt.figure(figsize=(100, 100))
    tree.plot_tree(regr.estimators_[0], feature_names=x_data.columns, filled=True)
    plt.savefig('decision_tree_plot.png')

    results = find_importances(x_data, regr)
    display_table(['Rank', 'Feature', 'Importances'], results)

    print(len(x_data.columns))

main()
