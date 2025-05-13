import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import sys
sys.path.insert(1, '/home/alberto/Documenti/Materiale scuola Alberto/BusinessIntelligenceProject/Data')
from preprocessing import preprocessing_diabetes
from linear_r2 import HyperplaneR2

#import warnings
#warnings.filterwarnings(action='ignore')
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

##############
# TRAINING SVM
##############

def lsvm_training(X_train, y_train, X_test, y_test, data_type):
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
    lsvm_hard = LinearSVC(C = C_hard, loss = loss_hard, dual = dual_hard, random_state = random_seed, max_iter=10000)
    lsvm_soft = LinearSVC(C = C_soft, loss = loss_soft, dual = dual_soft, random_state = random_seed, max_iter=10000)
    
    # Train the model
    lsvm_hard.fit(X_train, y_train)
    lsvm_soft.fit(X_train, y_train)

    # Make predictions
    y_pred_hard = lsvm_hard.predict(X_test)
    y_pred_soft = lsvm_soft.predict(X_test)
    
    # Calculate metrics
    f1_hard = f1_score(y_test, y_pred_hard)
    f1_soft = f1_score(y_test, y_pred_soft)
    precision_hard = precision_score(y_test, y_pred_hard)
    precision_soft = precision_score(y_test, y_pred_soft)
    recall_hard = recall_score(y_test, y_pred_hard)
    recall_soft = recall_score(y_test, y_pred_soft)

    # Plot confusion matrix for hard margin
    cm_hard = confusion_matrix(y_test, y_pred_hard)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_hard, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - Hard Margin - {data_type}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Plot confusion matrix for soft margin
    cm_soft = confusion_matrix(y_test, y_pred_soft)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_soft, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - Soft Margin - {data_type}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    if (data_type == 'PCA'):
        
        # Hyperplane coefficients
        w_hard = lsvm_hard.coef_
        b_hard = lsvm_hard.intercept_
        w_soft = lsvm_soft.coef_
        b_soft = lsvm_soft.intercept_
        
        # Separation lines
        line_hard = HyperplaneR2(w_hard, b_hard)
        line_soft = HyperplaneR2(w_soft, b_soft)

        """
        # Plot hyperplanes
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=20)
        plt.title(f'Hyperplanes - Hard Margin - {data_type}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.plot(line_hard.line_x2, line_hard.line_x2, color='blue', label='Hard Margin Hyperplane')
        plt.plot(line_soft.line_x2, line_soft.line_x2, color='red', label='Soft Margin Hyperplane')
        plt.legend()
        plt.show()
        """
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
        
    # Return metrics
    return f1_hard, f1_soft, precision_hard, precision_soft, recall_hard, recall_soft

def ksvm_train(X_train, y_train, X_test, y_test, kernel_type):
    # Train the SVM model with a specified kernel and evaluate its performance.
    # Accepted kernel types: 'rbf', 'poly', 'sigmoid'

    # Parameters initialization
    C = 1
    random_seed = 20000131

    # Prepare the model
    if kernel_type == 'sigmoid':
        svm_model = SVC(kernel='sigmoid', C=C, random_state=random_seed)
    elif kernel_type == 'rbf':
        svm_model = SVC(kernel='rbf', C=C, random_state=random_seed)
    elif kernel_type == 'poly':
        svm_model = SVC(kernel='poly', C=C, random_state=random_seed)
    else:
        raise ValueError("Invalid kernel type. Choose 'rbf', 'poly' or 'sigmoid'.")

    # Train the model
    svm_model.fit(X_train, y_train)

    # Make predictions
    y_pred = svm_model.predict(X_test)

    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {kernel_type} Kernel')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Return metrics
    return f1, precision, recall

####################
# EVALUATE THE MODEL
####################

# Linear SVM with hard and soft margins
results = {
    'Scaled': lsvm_training(df_train_scal, y_train, df_test_scal, y_test, 'Scaled'),
    'PCA': lsvm_training(df_train_PCA, y_train, df_test_PCA, y_test, 'PCA'),
    'No Features': lsvm_training(df_train_noFeat, y_train, df_test_noFeat, y_test, 'No Features'),
    'No Smoking': lsvm_training(df_train_noSmok, y_train, df_test_noSmok, y_test, 'No Smoking'),
}
# Display Linear SVM results
results_df = pd.DataFrame(results, index=['F1 Score Hard', 'F1 Score Soft', 'Precision Hard', 'Precision Soft', 'Recall Hard', 'Recall Soft']).T
results_df.index.name = 'Linear SVM'
results_df.reset_index(inplace=True)
display(results_df)

# Kernel SVM
kernel_results = {
    'Sigmoid': ksvm_train(df_train_scal, y_train, df_test_scal, y_test, 'sigmoid'),
    'RBF': ksvm_train(df_train_scal, y_train, df_test_scal, y_test, 'rbf'),
    'Polynomial': ksvm_train(df_train_scal, y_train, df_test_scal, y_test, 'poly'),
}
# Display kernel results
kernel_results_df = pd.DataFrame(kernel_results, index=['F1 Score', 'Precision', 'Recall']).T
kernel_results_df.index.name = 'Kernel SVM'
kernel_results_df.reset_index(inplace=True)
display(kernel_results_df)

# Trying best (rbf) on all data types
rbf_kernel_results_all = {
    'Scaled': ksvm_train(df_train_scal, y_train, df_test_scal, y_test, 'rbf'),
    'PCA': ksvm_train(df_train_PCA, y_train, df_test_PCA, y_test, 'rbf'),
    'No Features': ksvm_train(df_train_noFeat, y_train, df_test_noFeat, y_test, 'rbf'),
    'No Smoking': ksvm_train(df_train_noSmok, y_train, df_test_noSmok, y_test, 'rbf'),
}
# Display kernel results
print('RBF Kernel Results on all dataset preprocessings:')
rbf_kernel_results_all_df = pd.DataFrame(rbf_kernel_results_all, index=['F1 Score', 'Precision', 'Recall']).T
rbf_kernel_results_all_df.index.name = 'Kernel rbf SVM'
rbf_kernel_results_all_df.reset_index(inplace=True)
display(rbf_kernel_results_all_df)