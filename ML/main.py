import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from sklearn import tree
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

DEPTH = 5
RANDOM = 42
SCORE = 'neg_root_mean_squared_error'

def predict(final_starting_pos, driver_name, race1_fastest_time, race1_min_fastest_time, race1_max_fastest_time, final_temp):
    columns = []
    model = joblib.load('random_forest_model.pkl')

    with open('column_names.txt', 'r') as file:
        for line in file:
            column_name = line.strip()
            columns.append(column_name)

    race1_fastest_time = (race1_fastest_time - race1_min_fastest_time) / (race1_max_fastest_time - race1_min_fastest_time)

    manual_input = {'Final Starting Pos': final_starting_pos, 'Race 1 Fastest Time': race1_fastest_time, 'Final Temp': final_temp}

    manual_input['Track - Lauda'] = False
    manual_input['Track - Senna V1'] = False
    manual_input['Track - Senna V2'] = False
    manual_input['Track - VSK'] = False
    manual_input['Track - Speed Vegas'] = True

    for col in columns:
        if col == 'Driver Name_' + driver_name:
            manual_input[col] = True
        elif 'Driver Name_' in col:
            manual_input[col] = False

    input_df = pd.DataFrame([manual_input])

    return model.predict(input_df)

def preprocess_data(data):
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
    data['Final Temp'] = scaler.fit_transform(data[['Final Temp']])

    for col in ['Final Kart Num', 'Race 1 Kart Num', 'Driver Name']:
        one_hot_encoded = pd.get_dummies(data[col], prefix=col)
        data = pd.concat([data, one_hot_encoded], axis=1)

    x_data = data[['Final Starting Pos', 'Race 1 Fastest Time', 'Final Temp'] +
                  list(data.filter(regex='Driver Name_|Track').columns)]
    y_data = data['Final Finish Pos']

    return x_data, y_data

def conduct_experiments(x_data, y_data):
    results = []

    models = [(DummyRegressor(), 'BASELINE: Dummy Regressor'),
              (KNeighborsRegressor(n_neighbors=21, weights='distance'), 'KNN: K-Nearest Neighbors'),
              (SVR(kernel='linear'), 'SVM'),
              (GradientBoostingRegressor(random_state=RANDOM), 'Gradient Boosting'),
              (RandomForestRegressor(random_state=RANDOM, max_depth=DEPTH), 'Random Forest'),
              (MLPRegressor(random_state=RANDOM, activation='identity', max_iter=3200), 'Neural Network')]
    for model, name in models:
        results.append((-1 * cross_val_score(model, x_data, y_data, scoring=SCORE).mean(), name))

    results.sort()
    return results

def find_importances(x_data, regr):
    results = []
    importances = {'Driver Name': 0, 'Race 1 Kart Num': 0, 'Final Kart Num': 0, 'Track': 0}

    for i, col in enumerate(x_data.columns):
        key_present = False
        for key in importances:
            if key in col:
                importances[key] += regr.feature_importances_[i]
                key_present = True
                break

        if not key_present:
            results.append((regr.feature_importances_[i], col))

    for key, value in importances.items():
        results.append((value, key))

    results.sort(reverse=True)
    return results

def display_table(header, results):
    pt_table = PrettyTable(header)

    for i, result in enumerate(results):
        value, name = result
        if 'BASELINE' in name:
            pt_table.add_row(['','',''])
        pt_table.add_row([i + 1, name, round(value, 4)])
        if 'BASELINE' in name:
            pt_table.add_row(['','',''])

    print(pt_table)
    print()

def main():
    data = pd.read_csv('VSK - Sheet1.csv')
    x_data, y_data = preprocess_data(data)
    results = conduct_experiments(x_data, y_data)

    display_table(['Rank', 'Model Description', 'Root Mean Square Error (RMSE)'], results)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=RANDOM)
    regr = RandomForestRegressor(random_state=RANDOM, max_depth=DEPTH).fit(x_train, y_train)

    joblib.dump(regr, 'random_forest_model.pkl')

    y_pred = regr.predict(x_test)

    x_test['predicted'] = y_pred
    x_test['actual'] = y_test
    x_test['error'] = y_test - y_pred
    pd.set_option('display.max_columns', None)
    #print(X_test)

    plt.figure(figsize=(100, 100))
    tree.plot_tree(regr.estimators_[0], feature_names=x_data.columns, filled=True)
    plt.savefig('decision_tree_plot.png')
    plt.close()

    plt.figure(figsize=(15, 12))
    sns.heatmap(data[['Final Starting Pos', 'Race 1 Fastest Time', 'Race 1 Gap to Leader', 'Final Temp', 'Final Finish Pos', 'Pos Change']].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.savefig('heatmap.png')
    plt.close()

    results = find_importances(x_data, regr)
    display_table(['Rank', 'Feature', 'Importance'], results)

    with open('column_names.txt', 'w') as file:
        for column in x_data.columns:
            file.write(column + '\n')

    # print(len(x_data.columns))

# main()
print(predict(3, 'Alexander B.', 45.5, 45, 47, 90))
