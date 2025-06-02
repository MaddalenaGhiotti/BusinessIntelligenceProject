import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, make_scorer
import matplotlib.pyplot as plt
#from IPython.display import display
import sys
sys.path.insert(1, '../Data')
from linear_r2 import HyperplaneR2
from metrics import performances


##############
# TRAINING SVM
##############

def lsvm_training(X_train, y_train, X_test, y_test, data_type=''):
    # Trains the Linear SVM model and evaluate its performance.

    # Parameters initialization
    C_hard = 300
    loss_hard = 'squared_hinge'
    dual_hard = False
    C_soft = 1
    loss_soft = 'hinge'
    dual_soft = True
    random_seed = 20000131

    # Prepare the model
    lsvm_hard = LinearSVC(C = C_hard, loss = loss_hard, dual = dual_hard, random_state = random_seed, max_iter=20000)
    lsvm_soft = LinearSVC(C = C_soft, loss = loss_soft, dual = dual_soft, random_state = random_seed, max_iter=20000)
    
    # Train the model
    lsvm_hard.fit(X_train, y_train)
    lsvm_soft.fit(X_train, y_train)

    # Evaluace model performance
    lsvm_hard_performance = performances(lsvm_hard, X_test, y_test, f'Linear SVM HARD - {data_type}')
    lsvm_soft_performance = performances(lsvm_soft, X_test, y_test, f'Linear SVM SOFT - {data_type}')
    """
    if (data_type == 'PCA'):
        
        # Hyperplane coefficients
        w_hard = lsvm_hard.coef_
        b_hard = lsvm_hard.intercept_
        w_soft = lsvm_soft.coef_
        b_soft = lsvm_soft.intercept_
        
        # Separation lines
        line_hard = HyperplaneR2(w_hard, b_hard)
        line_soft = HyperplaneR2(w_soft, b_soft)

        # Plot hyperplanes
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=20)
        axs[0].plot([-5., 7.], [line_hard.line_x2(-5.), line_hard.line_x2(7.)])
        axs[0].plot([-5., 7.], [line_hard.margin_x2(-5.)[0], line_hard.margin_x2(7.)[0]], 'r--', label = 'Margin border 1')
        axs[0].plot([-5., 7.], [line_hard.margin_x2(-5.)[1], line_hard.margin_x2(7.)[1]], 'r--', label = 'Margin border 2')
        axs[0].set_title('SVM PCA HARD')
        axs[1].scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=20)
        axs[1].plot([-5., 7.], [line_soft.line_x2(-5.), line_soft.line_x2(7.)])
        axs[1].plot([-5., 7.], [line_soft.margin_x2(-5.)[0], line_soft.margin_x2(7.)[0]], 'r--', label = 'Margin border 1')
        axs[1].plot([-5., 7.], [line_soft.margin_x2(-5.)[1], line_soft.margin_x2(7.)[1]], 'r--', label = 'Margin border 2')
        axs[1].set_title('SVM PCA SOFT')
        plt.show()
    """
    # Return metrics
    return lsvm_hard_performance, lsvm_soft_performance

##########################################################################################

def ksvm_gridsearch(X_train, y_train, hparameters):
    # Trains the SVM model with a specified kernel and evaluates its performance using GridSearchCV.
    # Set random seed for reproducibility
    random_seed = 20000131
    # SVM initialization
    svm = SVC(random_state=random_seed)
    # GridSearch initialization
    svm_gs = GridSearchCV(estimator=svm,
                        param_grid=hparameters,
                        scoring='f1_weighted',
                        cv=KFold(10, random_state=random_seed, shuffle=True),
                        return_train_score=True,
                        verbose=1)
        
    # Run the grid search
    svm_gs.fit(X_train, y_train)

    #  Save hyperparameters to a DataFrame
    results = pd.DataFrame(svm_gs.cv_results_)

    return results

##########################################################################################

def ksvm_train(X_train, y_train, X_test, y_test, kernel_type, C = 1, gamma='auto', data_type=''):
    # Train the SVM model with a specified kernel and evaluate its performance on test.
    # Accepted kernel types: 'rbf', 'poly', 'sigmoid'

    # Random seed initialization
    random_seed = 20000131

    # Prepare the model
    if kernel_type == 'sigmoid':
        svm_model = SVC(kernel='sigmoid', C=C, gamma=gamma, random_state=random_seed)
    elif kernel_type == 'rbf':
        svm_model = SVC(kernel='rbf', C=C, gamma=gamma, random_state=random_seed)
    elif kernel_type == 'poly':
        svm_model = SVC(kernel='poly', C=C, gamma=gamma, random_state=random_seed)
    else:
        raise ValueError("Invalid kernel type. Choose 'rbf', 'poly' or 'sigmoid'.")

    # Train the model
    svm_model.fit(X_train, y_train)

    # Evaluate model performance
    svm_performance_train = performances(svm_model, X_train, y_train, f'Train - C = {C} - gamma = {gamma} - {data_type}')
    svm_performance_test = performances(svm_model, X_test, y_test, f'Test - C = {C} - gamma = {gamma} - {data_type}')
    svm_performance = (svm_performance_train, svm_performance_test)
    # Return metrics
    return svm_performance

##########################################################################################