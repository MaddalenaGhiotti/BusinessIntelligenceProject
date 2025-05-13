import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import sys
sys.path.insert(1, '/home/stekap/Scrivania/BusinessIntelligence/BusinessIntelligenceProject/Data')
from preprocessing import preprocessing_diabetes


#import warnings
#warnings.filterwarnings(action='ignore')
###############
# FETCHING DATA
###############

# Importing the dataset
trainingData = pd.read_csv('/home/stekap/Scrivania/BusinessIntelligence/BusinessIntelligenceProject/Data/diabetes_train.csv')
testData = pd.read_csv('/home/stekap/Scrivania/BusinessIntelligence/BusinessIntelligenceProject/Data/diabetes_test.csv')

# Preprocessing the data
df_train_scal, df_test_scal, y_train, y_test = preprocessing_diabetes(trainingData, testData)
df_train_noFeat, df_test_noFeat, _, _ = preprocessing_diabetes(trainingData, testData, option='Delete')
df_train_PCA, df_test_PCA, _, _ = preprocessing_diabetes(trainingData, testData, option='PCA')

#No smoking
df_train_noSmok = df_train_scal[[col for col in df_train_scal.columns if 'smoking' not in col]]
df_test_noSmok = df_test_scal[[col for col in df_test_scal.columns if 'smoking' not in col]]

from sklearn.ensemble import RandomForestClassifier

def random_forest_train(
    X_train, y_train, X_test, y_test,
    data_type='', n_estimators=100, verbose=False
):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=20000131)
    rf_model.fit(X_train, y_train)

    # Predictions
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)

    # Train metrics
    train_f1 = f1_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)

    # Test metrics
    test_f1 = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)

    if verbose:
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - RF - {data_type} ({n_estimators} Trees)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    return {
        'Train F1': train_f1,
        'Train Precision': train_precision,
        'Train Recall': train_recall,
        'Test F1': test_f1,
        'Test Precision': test_precision,
        'Test Recall': test_recall
    }


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
        metrics = random_forest_train(X_train, y_train, X_test, y_test,
                                      data_type=name, n_estimators=n, verbose=True)

        if metrics['Test F1'] > best_f1:
            best_f1 = metrics['Test F1']
            best_n = n
            best_metrics = metrics

    best_results[name] = {
        'Best n_estimators': best_n,
        **best_metrics
    }

# Format results into a DataFrame
best_results_df = pd.DataFrame(best_results).T
best_results_df.index.name = 'Data Variant'
best_results_df.reset_index(inplace=True)
display(best_results_df)
f1_scores_by_variant = {}

for name, (X_train, X_test) in data_variants.items():
    scores = []
    for n in n_tree_options:
        metrics = random_forest_train(X_train, y_train, X_test, y_test, data_type=name, n_estimators=n)
        scores.append(metrics['Test F1'])
    f1_scores_by_variant[name] = scores

# Plot
plt.figure(figsize=(10, 6))
for name, scores in f1_scores_by_variant.items():
    plt.plot(n_tree_options, scores, marker='o', label=name)

plt.title('F1 Score vs n_estimators (Random Forest)')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Test F1 Score')
plt.legend(title='Data Variant')
plt.grid(True)
plt.show()
