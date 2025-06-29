import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import sys
sys.path.insert(1, '../Data')
from preprocessing_v2 import preprocessing_diabetes_v2
from svm_utils import lsvm_training, ksvm_gridsearch, ksvm_train
import random
import numpy as np

##########################################################################################

###############
# FETCHING DATA
###############

# Setting a random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Importing the dataset
trainingData = pd.read_csv('../Data/diabetes_train.csv')
testData = pd.read_csv('../Data/diabetes_test.csv')

# Preprocessing the data (with oversample)
df_train_scal, df_test_scal, y_train, y_test = preprocessing_diabetes_v2(trainingData, testData, oversample=True)
df_train_noFeat, df_test_noFeat, _, _ = preprocessing_diabetes_v2(trainingData, testData, option='Delete', oversample=True) 
df_train_PCA, df_test_PCA, _, _ = preprocessing_diabetes_v2(trainingData, testData, option='PCA', oversample=True)

#No smoking
df_train_noSmok = df_train_scal[[col for col in df_train_scal.columns if 'smoking' not in col]]
df_test_noSmok = df_test_scal[[col for col in df_test_scal.columns if 'smoking' not in col]]

##########################################################################################

####################################
# EVALUATE THE MODEL WITH GRIDSEARCH
####################################

# LINEAR SVM
# Train Linear SVM with hard and soft margins
lsvm_results = {
    'Scaled': lsvm_training(df_train_scal, y_train, df_test_scal, y_test, 'Scaled', random_seed=SEED),
    'PCA': lsvm_training(df_train_PCA, y_train, df_test_PCA, y_test, 'PCA', random_seed=SEED),
    'No Features': lsvm_training(df_train_noFeat, y_train, df_test_noFeat, y_test, 'No Features', random_seed=SEED),
    'No Smoking': lsvm_training(df_train_noSmok, y_train, df_test_noSmok, y_test, 'No Smoking', random_seed=SEED),
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

# KERNEL SVM (GRIDSEARCH) - ALL KERNELS
"""
# Definizione delle liste di valori tra i quali "scorrere" per gli iper-parametri:
C_list = [1, 5, 10, 25, 50, 100]
gamma_list = [1, 0.5, 0.1, 0.01, 'scale', 'auto']
ker_list = ['rbf', 'sigmoid', 'poly']
hparameters = {'kernel':ker_list, 'C':C_list, 'gamma':gamma_list}

# Perform grid search for each data type
hp_results_scal = ksvm_gridsearch(df_train_scal, y_train, hparameters, random_seed=SEED)
hp_results_scal = hp_results_scal.sort_values(by='mean_test_score', ascending=False)
hp_results_PCA = ksvm_gridsearch(df_train_PCA, y_train, hparameters, random_seed=SEED)
hp_results_PCA = hp_results_PCA.sort_values(by='mean_test_score', ascending=False)
hp_results_noFeat = ksvm_gridsearch(df_train_noFeat, y_train, hparameters, random_seed=SEED)
hp_results_noFeat = hp_results_noFeat.sort_values(by='mean_test_score', ascending=False)
hp_results_noSmok = ksvm_gridsearch(df_train_noSmok, y_train, hparameters, random_seed=SEED)
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
C_list = [1, 5, 10, 20, 30, 50, 80, 90, 100, 110, 120]
gamma_list = [5, 1, 0.5, 0.1, 0.01, 'scale', 'auto']
ker_list = ['rbf']
hparameters = {'kernel':ker_list, 'C':C_list, 'gamma':gamma_list}

# Perform grid search for each data type
hp_rbf_results_scal = ksvm_gridsearch(df_train_scal, y_train, hparameters, random_seed=SEED)
hp_rbf_results_scal = hp_rbf_results_scal.sort_values(by='mean_test_score', ascending=False)
hp_rbf_results_PCA = ksvm_gridsearch(df_train_PCA, y_train, hparameters, random_seed=SEED)
hp_rbf_results_PCA = hp_rbf_results_PCA.sort_values(by='mean_test_score', ascending=False)
hp_rbf_results_noFeat = ksvm_gridsearch(df_train_noFeat, y_train, hparameters, random_seed=SEED)
hp_rbf_results_noFeat = hp_rbf_results_noFeat.sort_values(by='mean_test_score', ascending=False)
hp_rbf_results_noSmok = ksvm_gridsearch(df_train_noSmok, y_train, hparameters, random_seed=SEED)
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
    'Scaled': ksvm_train(df_train_scal, y_train, df_test_scal, y_test, 'rbf', C_scal, gamma_scal, 'Scaled', random_seed=SEED),
    'PCA': ksvm_train(df_train_PCA, y_train, df_test_PCA, y_test, 'rbf', C_PCA, gamma_PCA, 'PCA', random_seed=SEED),
    'No Feature': ksvm_train(df_train_noFeat, y_train, df_test_noFeat, y_test, 'rbf', C_noFeat, gamma_noFeat, 'No Feature', random_seed=SEED),
    'No Smoking': ksvm_train(df_train_noSmok, y_train, df_test_noSmok, y_test, 'rbf', C_noSmok, gamma_noSmok, 'No Smoking', random_seed=SEED),
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