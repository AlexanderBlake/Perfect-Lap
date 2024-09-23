import pandas as pd
from sklearn.svm import SVR
from prettytable import PrettyTable as pt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

def main():
    df = pd.read_csv('VSK - Sheet8.csv')

    X = pd.get_dummies(df['Kart'], prefix='Kart')
    y = df['Performance Index']

    k_fold = KFold()

    linear_svc = SVR(kernel='linear')
    polynomial_svc = SVR(kernel='poly')
    rbf_svc = SVR()
    knn = KNeighborsRegressor()

    pt_table = pt(['K-Fold', 'Linear Kernel Accuracy', 'Polynomial Kernel Accuracy', 'Radial Basis Function Accuracy', 'KNN'])

    for fold_num, (train_idx, test_idx) in enumerate(k_fold.split(X)):
        X_training, X_testing = X.iloc[train_idx], X.iloc[test_idx]
        y_training, y_testing = y.iloc[train_idx], y.iloc[test_idx]

        linear_svc.fit(X_training, y_training)
        polynomial_svc.fit(X_training, y_training)
        rbf_svc.fit(X_training, y_training)
        knn.fit(X_training, y_training)

        pt_row = [fold_num + 1, linear_svc.score(X_testing, y_testing), polynomial_svc.score(X_testing, y_testing), rbf_svc.score(X_testing, y_testing), knn.score(X_testing, y_testing)]
        pt_table.add_row(pt_row)

    print(pt_table)

    unique_X_values = df['Kart'].unique()

    pred_table = pt(['Kart', 'Radial Basis Function Prediction'])

    # Make predictions for each unique X value
    for value in unique_X_values:
        prediction_value = pd.DataFrame([[value]], columns=['Kart'])
        prediction_value = pd.get_dummies(prediction_value, prefix='Kart')
        
        # Align columns to match the training data
        prediction_value = prediction_value.reindex(columns=X.columns, fill_value=0)
        prediction_value['Kart_' + str(value)] = 1

        linear_pred = linear_svc.predict(prediction_value)[0]
        poly_pred = polynomial_svc.predict(prediction_value)[0]
        rbf_pred = rbf_svc.predict(prediction_value)[0]
        knn_pred = knn.predict(prediction_value)[0]

        # Add predictions to table
        pred_table.add_row([value, rbf_pred])

    print(pred_table)

if __name__ == "__main__":
    main()
