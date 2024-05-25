import numpy as np
import pandas as pd
from math import sqrt
from sklearn import tree
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split

def main():
    df = pd.read_csv('VSK - Sheet1.csv')

    df = df.groupby(['Race 1 DateTime'])
    scaler = MinMaxScaler()
    
    scaled_groups = []
    for name, group in df:
        group['Race 1 Fastest Time'] = scaler.fit_transform(group['Race 1 Fastest Time'].values.reshape(-1, 1))
        scaled_groups.append(group)

    df = pd.concat(scaled_groups)

    for col in ['Race 1 Kart Num', 'Final Kart Num', 'Driver Name']:
        one_hot_encoded = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, one_hot_encoded], axis=1)

    X = df[['Final Starting Pos', 'Race 1 Fastest Time'] +
           list(df.filter(regex='Track|Final Kart Num_|Driver Name_').columns)]
    y = df['Final Finish Pos']

    results = []

    models = [(LinearRegression(), 'Linear Regression'),
              (KNeighborsRegressor(), 'KNN'),
              (RandomForestRegressor(random_state=42), 'Random Forest'),
              (LinearSVR(random_state=42, dual='auto', max_iter=4000), 'Linear SVM')]
    for model, name in models:
        results.append((-1 * cross_val_score(model, X, y, scoring='neg_root_mean_squared_error').mean(), name))

    for depth in range(1, 17):
        results.append((-1 * cross_val_score(tree.DecisionTreeRegressor(random_state=42, max_depth=depth), X, y, scoring='neg_root_mean_squared_error').mean(), 'Decision Tree Max Depth: ' + str(depth)))
    
    for kern in ['rbf', 'linear', 'poly', 'sigmoid']:
        results.append((-1 * cross_val_score(NuSVR(kernel=kern), X, y, scoring='neg_root_mean_squared_error').mean(), 'NU SVM Kernel: ' + kern))
        results.append((-1 * cross_val_score(SVR(kernel=kern), X, y, scoring='neg_root_mean_squared_error').mean(), 'SVM Kernel: ' + kern))

    kf = KFold()
    for guesses in range(10, 15):
        temp = 0
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            temp += mean_squared_error(np.random.randint(1, guesses, size=len(X.iloc[test_index])), y.iloc[test_index])
        results.append((temp / 5, 'Random Guessing 1 - ' + str(guesses)))

    '''
    for ac in ['relu', 'identity', 'logistic', 'tanh']:
        for so in ['lbfgs', 'sgd', 'adam']:
            for lr in ['constant', 'invscaling', 'adaptive']:
                if so == 'sgd':
                    results.append((-1 * cross_val_score(MLPRegressor(random_state=42, max_iter=12800, activation=ac, solver=so, learning_rate=lr), X, y, scoring='neg_root_mean_squared_error').mean(), 'Neural Network ' + ac + ' ' + so + ' ' + lr))
                else:
                    results.append((-1 * cross_val_score(MLPRegressor(random_state=42, max_iter=12800, activation=ac, solver=so), X, y, scoring='neg_root_mean_squared_error').mean(), 'Neural Network ' + ac + ' ' + so + ' ' + lr))
                    break
    '''

    results.sort()
    pt_table = PrettyTable(['Rank', 'Model Description', 'Root Mean Square Error (RMSE)'])

    for i in range(len(results)):
        score, model = results[i]
        pt_table.add_row([i + 1, model, score])

    print(pt_table)
    print()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    regr = tree.DecisionTreeRegressor(random_state=42, max_depth=2).fit(X_train, y_train)
    
    y_pred = regr.predict(X_test)

    X_test['predicted'] = y_pred
    X_test['actual'] = y_test
    X_test['error'] = y_test - y_pred
    pd.set_option('display.max_columns', None)
    #print(X_test)

    plt.figure(figsize=(100, 100))
    tree.plot_tree(regr, feature_names=X.columns, filled=True)
    plt.savefig('decision_tree_plot.png')

    results = []
    importances = {'Driver Name': 0, 'Final Kart Num': 0, 'Track': 0}
    for i in range(len(X.columns)):
        if 'Driver Name' in X.columns[i]:
            importances['Driver Name'] += regr.feature_importances_[i]
        elif 'Final Kart Num' in X.columns[i]:
            importances['Final Kart Num'] += regr.feature_importances_[i]
        elif 'Track' in X.columns[i]:
            importances['Track'] += regr.feature_importances_[i]
        else:
            results.append((regr.feature_importances_[i], X.columns[i]))

    for key, value in importances.items():
        results.append((value, key))

    results.sort(reverse=True)
    pt_table = PrettyTable(['Rank', 'Feature', 'Importances'])

    for i in range(len(results)):
        importance, feature = results[i]
        pt_table.add_row([i + 1, feature, importance])

    print(pt_table)
    print()

    print(len(X.columns))

main()
