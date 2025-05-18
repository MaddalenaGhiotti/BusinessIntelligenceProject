import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocessing_diabetes(df_train, df_test, option=''):
    # Remove Errors
    df_train_filter = df_train[(df_train['age'] > 0) & (df_train['bmi'] < 70)].copy().reset_index(drop=True)
    df_test_filter = df_test.loc[(df_test['age'] > 0) & (df_test['bmi'] < 70)].copy().reset_index(drop=True)

    # Missing Values
    df_train_filter = df_train_filter.fillna(df_train_filter.mean(numeric_only=True))
    df_test_filter = df_test_filter.fillna(df_test_filter.mean(numeric_only=True))

    # Label encoding gender
    df_train_filter['gender'] = df_train_filter['gender'].map({'Male': 1, 'Female': 0})
    df_test_filter['gender'] = df_test_filter['gender'].map({'Male': 1, 'Female': 0})

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

    # Dividing real vs other columns
    binary_columns = X_train.columns[X_train.map(lambda x: x in [0, 1]).all()]
    numeric_columns = X_train.drop(columns=binary_columns).columns
    X_train_numeric = X_train[numeric_columns]
    X_test_numeric = X_test[numeric_columns]
    X_train_cat = X_train.drop(columns=numeric_columns)
    X_test_cat = X_test.drop(columns=numeric_columns)

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)
    X_train_combined = pd.concat([pd.DataFrame(X_train_scaled, columns=numeric_columns), 
                                  X_train_cat], axis=1)
    X_test_combined = pd.concat([pd.DataFrame(X_test_scaled, columns=numeric_columns), 
                                 X_test_cat], axis=1)
    
    if option == 'PCA':    
        # PCA
        pca = PCA(n_components=0.975)
        X_train_pca = pca.fit_transform(X_train_combined)
        X_test_pca = pca.transform(X_test_combined)

        # Convert and Export
        df_train_pca = pd.DataFrame(X_train_pca)
        df_test_pca = pd.DataFrame(X_test_pca)
        return df_train_pca, df_test_pca, y_train, y_test
    
    else:
        # Convert and Export
        df_train_scal = pd.DataFrame(X_train_combined)
        df_test_scal = pd.DataFrame(X_test_combined)
        return df_train_scal, df_test_scal, y_train, y_test
