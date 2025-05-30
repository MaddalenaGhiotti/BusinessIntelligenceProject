import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

def combination_features(df, operation_list):

    # array NumPy with number of original feature
    arr = df.to_numpy()  
    num_col = arr.shape[1] 
    
    # Initialize variables for new feature and new names
    num_new_col = (len(operation_list) * num_col * (num_col - 1) // 2) 
    new_features = np.empty((arr.shape[0], num_new_col)) 
    new_names = []
    col_idx = 0  # index
    
    # Execute operation for each couple of features
    for i in range(num_col): 
        for j in range(i+1, num_col):
            for operation in operation_list:
                if operation == 'addition': 
                    new_features[:, col_idx] = arr[:, i] + arr[:, j] 
                elif operation == 'multiplication':
                    new_features[:, col_idx] = arr[:, i] * arr[:, j]
                elif operation == 'subtraction':
                    new_features[:, col_idx] = arr[:, i] - arr[:, j]
                elif operation == 'division':
                    # Add a small value (1e-10) to the denominator to avoid division by zero.
                    new_features[:, col_idx] = arr[:, i] / (arr[:, j] + 1e-10)
            
                new_names.append(f"{operation.upper()}--{df.columns[i]}--{df.columns[j]}")
                col_idx += 1

    df_new_feat = pd.DataFrame(new_features, columns=new_names)
    df_result = pd.concat([df, df_new_feat], axis=1)

    return df_result

def preprocessing_diabetes_v3(df_train, df_test, option='', augment=False, 
                              oversample=False, delete_smoking=False):
    # Remove Errors
    df_train_filter = df_train[(df_train['age'] > 0) & (df_train['bmi'] < 70)].reset_index(drop=True)
    df_test_filter = df_test.loc[(df_test['age'] > 0) & (df_test['bmi'] < 70)].reset_index(drop=True)

    # Missing Values
    df_train_filter = df_train_filter.fillna(df_train_filter.mean(numeric_only=True))
    df_test_filter = df_test_filter.fillna(df_test_filter.mean(numeric_only=True))

    # Label encoding gender
    df_train_filter['gender'] = df_train_filter['gender'].map({'Male': 1, 'Female': 0})
    df_test_filter['gender'] = df_test_filter['gender'].map({'Male': 1, 'Female': 0})

    # Manage smoking history
    if delete_smoking:
        df_train_oh = df_train_filter.drop(columns=["smoking_history"])
        df_test_oh = df_test_filter.drop(columns=["smoking_history"])
    else:
        # people with age <= 10 no smoking
        df_train_filter.loc[df_train_filter['age'] <= 10, 'smoking_history'] = 'never'
        df_test_filter.loc[df_test_filter['age'] <= 10, 'smoking_history'] = 'never'

        # One-Hot encoding smoking info
        df_train_oh = pd.get_dummies(df_train_filter)
        df_train_oh.drop(columns=["smoking_history_No Info"], inplace=True)
        df_test_oh = pd.get_dummies(df_test_filter)
        df_test_oh.drop(columns=["smoking_history_No Info"], inplace=True)

    # Deleating too correlated feature
    if option == 'Delete':
        df_train_oh.drop(columns=["BMI_Glucose_Interaction"], inplace=True)
        df_test_oh.drop(columns=["BMI_Glucose_Interaction"], inplace=True)  

    # Dividing feature and target
    X_train = df_train_oh.drop(columns=["diabetes"])
    X_test = df_test_oh.drop(columns=["diabetes"])
    y_train = df_train_oh["diabetes"] 
    y_test = df_test_oh["diabetes"]

    # Oversample
    if oversample:
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # Dividing real vs other columns
    binary_columns = X_train.columns[X_train.map(lambda x: x in [0, 1]).all()]
    numeric_columns = X_train.drop(columns=binary_columns).columns
    X_train_numeric = X_train[numeric_columns]
    X_test_numeric = X_test[numeric_columns]
    X_train_cat = X_train.drop(columns=numeric_columns)
    X_test_cat = X_test.drop(columns=numeric_columns)

    # Augmenting feature
    if augment:
        X_train_numeric = combination_features(X_train_numeric, ['multiplication', 'division'])
        X_test_numeric = combination_features(X_test_numeric, ['multiplication', 'division'])
        numeric_columns = X_train_numeric.columns
        X_train_cat = combination_features(X_train_cat, ['addition', 'subtraction'])
        X_test_cat = combination_features(X_test_cat, ['addition', 'subtraction'])
        categorical_columns = X_train_cat.columns

        # Scaling augmented categorical data
        mmscaler = MinMaxScaler()
        X_train_cat_scaled = mmscaler.fit_transform(X_train_cat)
        X_test_cat_scaled = mmscaler.transform(X_test_cat)
        X_train_cat = pd.DataFrame(X_train_cat_scaled, columns=categorical_columns)
        X_test_cat = pd.DataFrame(X_test_cat_scaled, columns=categorical_columns)

    # Standardization
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_numeric)
    X_test_num_scaled = scaler.transform(X_test_numeric)
    X_train_combined = pd.concat([pd.DataFrame(X_train_num_scaled, columns=numeric_columns), 
                                  X_train_cat], axis=1)
    X_test_combined = pd.concat([pd.DataFrame(X_test_num_scaled, columns=numeric_columns), 
                                 X_test_cat], axis=1)
    
    if option == 'PCA':    
        # PCA
        pca = PCA(n_components=0.975)
        X_train_pca = pca.fit_transform(X_train_combined)
        X_test_pca = pca.transform(X_test_combined)

        # Convert and Export
        PCs = [f"PC {i}" for i in range(1, X_train_pca.shape[1]+1)]
        df_train_pca = pd.DataFrame(X_train_pca, columns=PCs)
        df_test_pca = pd.DataFrame(X_test_pca, columns=PCs)                                  
        return df_train_pca, df_test_pca, y_train, y_test
    
    else:
        return X_train_combined, X_test_combined, y_train, y_test
