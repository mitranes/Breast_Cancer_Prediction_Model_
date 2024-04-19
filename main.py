import copy
import math

import numpy as np
from pandas import read_csv
import csv

def load_data():
    d = read_csv('Breast_cancer_data.csv')
    data_array = d.values
    return data_array

def get_training_data(data_array):
    X_train = data_array[:400,:5]
    y_train = data_array[:400,5:6]
    return X_train, y_train

def get_testing_data(data_array):
    X_test = data_array[400:,:5]
    y_test = data_array[400:,5:6]
    return X_test, y_test

def normalize_testing_data(X_test, mu, sigma):
    X_test_norm = (X_test - mu)/sigma
    return X_test_norm
def z_score_norm_data(X_train):
    mu = np.mean(X_train, axis=0)   #mu and sigma will have shape (n,)
    sigma = np.std(X_train, axis=0)
    X_train_norm = (X_train - mu)/sigma
    print("Here is mu ", mu)
    print("Here is sigma ", sigma)
    print("Here is X Train Normalized first 5: ", X_train_norm[:5])
    return (X_train_norm, mu, sigma)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic(X, y, w, b):
    m,n= X.shape
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost = cost + -y[i]*np.log(f_wb_i) - (1 - y[i])*np.log(1-f_wb_i)
    cost = cost/m
    return cost

def compute_gradient_logistic(X, y, w_in, b_in):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        z_i = np.dot(X[i], w_in) + b_in
        f_wb_i = sigmoid(z_i)
        err = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw, dj_db

def compute_gradient_descent_logistic(X,y, w_in,b_in, compute_cost_logistic, compute_gradient_logistic, alpha, iterations):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(iterations):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        #make simultaneous updates
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 1000000:
            cost = compute_cost_logistic(X,y,w,b)
            J_history.append(cost)
            if i % math.ceil(iterations / 10) == 0:
                print("Iteration ", i, "Cost is ", cost)
    return w,b,J_history

def prediction_breast_cancer(X,y,w,b):
    m = X.shape[0]
    results =np.empty((m,2))
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb = sigmoid(z)
        print("Prediction of Breast Cancer for this patient is ",  f_wb, " and the actual is ", y[i])
        results[i,0] = f_wb
        results[i,1] = y[i]
    return results

def percent_accuracy(results):
    m = results.shape[0]
    correct_count = 0
    for i in range(m):
        if results[i,0] <= 0.5:
            results[i,0] = 0
        else:
            results[i,0]=1
        if results[i,0] == results[i,1]:
            correct_count= correct_count+1
    return correct_count/m

if __name__ == "__main__":
    full_data = load_data()
    X_train, y_train = get_training_data(full_data)
    X_train_norm, mu, sigma = z_score_norm_data(X_train)
    w_in = np.zeros_like(X_train_norm[0])
    b_in = 0.0
    alpha = 0.1
    iters = 1000
    w_final, b_final, J_history = compute_gradient_descent_logistic(X_train_norm,y_train,w_in,b_in,compute_cost_logistic,compute_gradient_logistic,alpha,iters)
    print("Final W is ", w_final)
    print("Final b is ", b_final)
    print("NOW LETS SEE OUR PREDICTIONS!!! -------")
    X_test, y_test = get_testing_data(full_data)
    X_test_norm = normalize_testing_data(X_test,mu,sigma)
    results = prediction_breast_cancer(X_test_norm,y_test,w_final, b_final)
    print("percent accuracy is ", percent_accuracy(results))
