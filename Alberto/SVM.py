import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import sys
sys.path.insert(1, '/home/alberto/Documenti/Materiale scuola Alberto/BusinessIntelligenceProject/Data')
from preprocessing import preprocessing_diabetes
from linear_r2 import HyperplaneR2
from metrics import performances

#import warnings
#warnings.filterwarnings(action='ignore')

##########################################################################################

###############
# FETCHING DATA
###############

# Importing the dataset
trainingData = pd.read_csv('/home/alberto/Documenti/Materiale scuola Alberto/BusinessIntelligenceProject/Data/diabetes_train.csv')
testData = pd.read_csv('/home/alberto/Documenti/Materiale scuola Alberto/BusinessIntelligenceProject/Data/diabetes_test.csv')

# Preprocessing the data
df_train_scal, df_test_scal, y_train, y_test = preprocessing_diabetes(trainingData, testData)
df_train_noFeat, df_test_noFeat, _, _ = preprocessing_diabetes(trainingData, testData, option='Delete')
df_train_PCA, df_test_PCA, _, _ = preprocessing_diabetes(trainingData, testData, option='PCA')

#No smoking
df_train_noSmok = df_train_scal[[col for col in df_train_scal.columns if 'smoking' not in col]]
df_test_noSmok = df_test_scal[[col for col in df_test_scal.columns if 'smoking' not in col]]

##########################################################################################

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
        
    # Return metrics
    return lsvm_hard_performance, lsvm_soft_performance

##########################################################################################

def ksvm_gridsearch(X_train, y_train, hparameters):
    # Trains the SVM model with a specified kernel and evaluates its performance using GridSearchCV.
    # SVM initialization
    svm = SVC()
    # GridSearch initialization
    svm_gs = GridSearchCV(estimator=svm,
                        param_grid=hparameters,
                        scoring='f1_weighted',
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

####################################
# EVALUATE THE MODEL WITH GRIDSEARCH
####################################

# LINEAR SVM
# Train Linear SVM with hard and soft margins
lsvm_results = {
    'Scaled': lsvm_training(df_train_scal, y_train, df_test_scal, y_test, 'Scaled'),
    'PCA': lsvm_training(df_train_PCA, y_train, df_test_PCA, y_test, 'PCA'),
    'No Features': lsvm_training(df_train_noFeat, y_train, df_test_noFeat, y_test, 'No Features'),
    'No Smoking': lsvm_training(df_train_noSmok, y_train, df_test_noSmok, y_test, 'No Smoking'),
}

# Display Linear SVM results
print('Linear SVM Results (Hard and Soft margins):')
for key, (hard_perf, soft_perf) in lsvm_results.items():
    # Display metrics for hard margin
    display(hard_perf[0])
    # Display confusion matrix for hard margin
    plt.figure()
    sns.heatmap(hard_perf[1], annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - Hard Margin - {key}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Display metrics for soft margin
    display(soft_perf[0])
    # Display confusion matrix for soft margin
    plt.figure()
    sns.heatmap(soft_perf[1], annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - Hard Margin - {key}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

##########################################################################################

# KERNEL SVM (GRIDSEARCH)
"""
# Definizione delle liste di valori tra i quali "scorrere" per gli iper-parametri:
C_list = [1, 5, 10, 25, 50, 100]
gamma_list = [1, 0.5, 0.1, 0.01, 'scale', 'auto']
ker_list = ['rbf', 'sigmoid', 'poly']
hparameters = {'kernel':ker_list, 'C':C_list, 'gamma':gamma_list}

# Perform grid search for each data type
hp_results_scal = ksvm_gridsearch(df_train_scal, y_train, hparameters)
hp_results_scal = hp_results_scal.sort_values(by='mean_test_score', ascending=False)
hp_results_PCA = ksvm_gridsearch(df_train_PCA, y_train, hparameters)
hp_results_PCA = hp_results_PCA.sort_values(by='mean_test_score', ascending=False)
hp_results_noFeat = ksvm_gridsearch(df_train_noFeat, y_train, hparameters)
hp_results_noFeat = hp_results_noFeat.sort_values(by='mean_test_score', ascending=False)
hp_results_noSmok = ksvm_gridsearch(df_train_noSmok, y_train, hparameters)
hp_results_noSmok = hp_results_noSmok.sort_values(by='mean_test_score', ascending=False)

# Displaying the results of the grid search
print('Grid Search Results:')
display(hp_results_scal[['params', 'mean_test_score', 'std_test_score']])
display(hp_results_PCA[['params', 'mean_test_score', 'std_test_score']])
display(hp_results_noFeat[['params', 'mean_test_score', 'std_test_score']])
display(hp_results_noSmok[['params', 'mean_test_score', 'std_test_score']])
"""

##########################################################################################

# KERNEL SVM (GRIDSEARCH) - RBF ONLY
# Performing GridSearch for the best kernel (rbf)
C_list = [1, 5, 10, 15, 20, 25, 30, 40, 60, 90]
gamma_list = [5, 1, 0.5, 0.1, 0.01, 'scale', 'auto']
ker_list = ['rbf']
hparameters = {'kernel':ker_list, 'C':C_list, 'gamma':gamma_list}

# Perform grid search for each data type
hp_rbf_results_scal = ksvm_gridsearch(df_train_scal, y_train, hparameters)
hp_rbf_results_scal = hp_rbf_results_scal.sort_values(by='mean_test_score', ascending=False)
hp_rbf_results_PCA = ksvm_gridsearch(df_train_PCA, y_train, hparameters)
hp_rbf_results_PCA = hp_rbf_results_PCA.sort_values(by='mean_test_score', ascending=False)
hp_rbf_results_noFeat = ksvm_gridsearch(df_train_noFeat, y_train, hparameters)
hp_rbf_results_noFeat = hp_rbf_results_noFeat.sort_values(by='mean_test_score', ascending=False)
hp_rbf_results_noSmok = ksvm_gridsearch(df_train_noSmok, y_train, hparameters)
hp_rbf_results_noSmok = hp_rbf_results_noSmok.sort_values(by='mean_test_score', ascending=False)

# Displaying the results of the grid search
print('Grid Search Results:')
display(hp_rbf_results_scal[['params', 'mean_test_score', 'std_test_score']])
display(hp_rbf_results_PCA[['params', 'mean_test_score', 'std_test_score']])
display(hp_rbf_results_noFeat[['params', 'mean_test_score', 'std_test_score']])
display(hp_rbf_results_noSmok[['params', 'mean_test_score', 'std_test_score']])

##########################################################################################

# TESTING RBF KERNEL WITH BEST PARAMETERS
# Extracting best parameters from the grid search results
C_scal = hp_rbf_results_scal.iloc[0]['params']['C']
gamma_scal = hp_rbf_results_scal.iloc[0]['params']['gamma']
C_PCA = hp_rbf_results_PCA.iloc[0]['params']['C']
gamma_PCA = hp_rbf_results_PCA.iloc[0]['params']['gamma']
C_noFeat = hp_rbf_results_noFeat.iloc[0]['params']['C']
gamma_noFeat = hp_rbf_results_noFeat.iloc[0]['params']['gamma']
C_noSmok = hp_rbf_results_noSmok.iloc[0]['params']['C']
gamma_noSmok = hp_rbf_results_noSmok.iloc[0]['params']['gamma']

# Tesiting best parameters on all data types
rbf_kernel_results_all = {
    'Scaled': ksvm_train(df_train_scal, y_train, df_test_scal, y_test, 'rbf', C_scal, gamma_scal, 'Scaled'),
    'PCA': ksvm_train(df_train_PCA, y_train, df_test_PCA, y_test, 'rbf', C_PCA, gamma_PCA, 'PCA'),
    'No Feature': ksvm_train(df_train_noFeat, y_train, df_test_noFeat, y_test, 'rbf', C_noFeat, gamma_noFeat, 'No Feature'),
    'No Smoking': ksvm_train(df_train_noSmok, y_train, df_test_noSmok, y_test, 'rbf', C_noSmok, gamma_noSmok, 'No Smoking'),
}
# Display kernel results
print(f'Kernel rbf SVM test results on all data types:')
for key in rbf_kernel_results_all.keys():
    # Display metrics
    display(rbf_kernel_results_all[key][0][0])
    display(rbf_kernel_results_all[key][1][0])
    # Display confusion matrix
    plt.figure()
    sns.heatmap(rbf_kernel_results_all[key][1][1], annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - Kernel rbf - {key}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()