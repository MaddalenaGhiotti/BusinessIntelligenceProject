import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import time
import sys

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(1, '../Data')
from preprocessing import preprocessing_diabetes
from preprocessing_v2 import preprocessing_diabetes_v2

# Importing the dataset
trainingData = pd.read_csv('../Data/diabetes_train.csv')
testData = pd.read_csv('../Data/diabetes_test.csv')

# Preprocessing the data
df_train_scal, df_test_scal, y_train, y_test = preprocessing_diabetes(trainingData, testData)
df_train_noFeat, df_test_noFeat, _, _ = preprocessing_diabetes(trainingData, testData, option='Delete')
df_train_PCA, df_test_PCA, _, _ = preprocessing_diabetes(trainingData, testData, option='PCA')

# No smoking
df_train_noSmok = df_train_scal[[col for col in df_train_scal.columns if 'smoking' not in col]]
df_test_noSmok = df_test_scal[[col for col in df_test_scal.columns if 'smoking' not in col]]

from sklearn.ensemble import RandomForestClassifier

# Define performances function
def performances(model, data, y_true, title=None):
    start = time.time()
    y_pred = model.predict(data)
    stop = time.time()
    totalTime = stop - start
    acc = model.score(data, y_true)
    prec = precision_score(y_true, y_pred, average='binary')
    rec = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    df = pd.DataFrame({'Accuracy': [acc], 
                        'Precision': [prec ], 
                        'Recall': [rec ],
                        'F1': [f1 ]
                       },
                      index=[title])
    cmat = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=model.classes_))
    return df, cmat, totalTime

def random_forest_train(
    X_train, y_train, X_test, y_test,
    data_type='', n_estimators=100, verbose=False
):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=20000131)
    rf_model.fit(X_train, y_train)

    # Get performance metrics using the performances function
    metrics_df, cm, total_time = performances(rf_model, X_test, y_test, title=data_type)
    
    if verbose:
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - RF - {data_type} ({n_estimators} Trees)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    return metrics_df, cm, total_time


n_tree_options = [10, 50, 100, 200, 300]
data_variants = {
    'Scaled': (df_train_scal, df_test_scal),
    'PCA': (df_train_PCA, df_test_PCA),
    'No Features': (df_train_noFeat, df_test_noFeat),
    'No Smoking': (df_train_noSmok, df_test_noSmok),
}

best_results = {}

for name, (X_train, X_test) in data_variants.items():
    best_f1 = 0
    best_n = 0
    best_metrics = None

    for n in n_tree_options:
        metrics_df, cm, total_time = random_forest_train(X_train, y_train, X_test, y_test,
                                                         data_type=name, n_estimators=n, verbose=False)
        test_f1 = metrics_df['F1'].iloc[0]
        
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_n = n
            best_metrics = (metrics_df, cm, total_time)

    best_results[name] = {
        'Best n_estimators': best_n,
        'Best Metrics': best_metrics
    }

# Format results into a DataFrame
best_results_df = pd.DataFrame({
    'Best n_estimators': [best_results[name]['Best n_estimators'] for name in best_results],
    'Test F1': [best_results[name]['Best Metrics'][0]['F1'].iloc[0] for name in best_results],
    'Test Accuracy': [best_results[name]['Best Metrics'][0]['Accuracy'].iloc[0] for name in best_results],
    'Test Precision': [best_results[name]['Best Metrics'][0]['Precision'].iloc[0] for name in best_results],
    'Test Recall': [best_results[name]['Best Metrics'][0]['Recall'][0] for name in best_results],
    'Total Time': [best_results[name]['Best Metrics'][2] for name in best_results]
}, index=best_results.keys())

best_results_df.index.name = 'Data Variant'
best_results_df.reset_index(inplace=True)
display(best_results_df)

# Bar plot for best F1 scores by data variant
best_f1_scores = {name: best_results[name]['Best Metrics'][0]['F1'].iloc[0] for name in best_results}

plt.figure(figsize=(8, 6))
plt.bar(best_f1_scores.keys(), best_f1_scores.values(), color='skyblue')
plt.title('Best Test F1 Score by Data Variant')
plt.xlabel('Data Variant')
plt.ylabel('Test F1 Score')
plt.show()

# Plot confusion matrix for the best model of each variant
for name, (X_train, X_test) in data_variants.items():
    # Get the best number of trees for this variant
    best_n = best_results[name]['Best n_estimators']
    
    # Get the best metrics (including the confusion matrix) for the best model
    metrics_df, cm, total_time = best_results[name]['Best Metrics']
    
    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - Best Model ({name}) - {best_n} Trees')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()




