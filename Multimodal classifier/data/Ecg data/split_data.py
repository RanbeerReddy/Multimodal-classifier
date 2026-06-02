import numpy as np
from transform_data import preprocess_ecg_batch 

def get_split(Y_df, X_arr, folds):
    mask = np.isin(Y_df.strat_fold.values, folds)
    return X_arr[mask], Y_df[mask].copy()
    
def split_data(Y_clean, X):
    x_train_raw, y_train = get_split(Y_clean, X, list(range(1,9)))
    x_valid_raw, y_valid = get_split(Y_clean, X, [9])
    x_test_raw, y_test = get_split(Y_clean, X,[10])

    x_train = preprocess_ecg_batch(x_train_raw)
    x_valid = preprocess_ecg_batch(x_valid_raw)
    x_test  = preprocess_ecg_batch(x_test_raw)

    # Normalize numeric metadata (fit on train, apply on all)
    NUM_META = ['age', 'weight']
    meta_mean = y_train[NUM_META].mean()
    meta_std  = y_train[NUM_META].std() + 1e-8
    for df in [y_train, y_valid, y_test]:
        df[NUM_META] = (df[NUM_META] - meta_mean) / meta_std

    return x_train, x_valid, x_test, y_train, y_valid , y_test