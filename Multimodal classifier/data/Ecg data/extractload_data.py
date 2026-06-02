import ast
import numpy as np
import pandas as pd
import wfdb
import tqdm

from configs.dataset.ecgdata_config import SUPERCLASSES, COLS_TO_DROP, path, sampling_rate


def load_annotation_data(path):
    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    return Y


def load_raw_data(df, sampling_rate, path):
    filenames = df.filename_lr if sampling_rate == 100 else df.filename_hr
    data = [wfdb.rdsamp(path + f)[0] for f in tqdm.tqdm(filenames)]
    return np.array(data, dtype=np.float32)


def aggregate_superclass(y_dic, agg_df):
    return list(set(
        agg_df.loc[key].diagnostic_class
        for key in y_dic.keys()
        if key in agg_df.index
    ))


def prepare_dataset(path=path, sampling_rate=sampling_rate):
    Y = load_annotation_data(path)

    gg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
    agg_df = gg_df[gg_df.diagnostic == 1]

    Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: aggregate_superclass(x, agg_df))
    Y['n_superclass'] = Y['diagnostic_superclass'].apply(len)

    Y_labeled = Y[Y['n_superclass'] > 0].copy()
    X_labeled = load_raw_data(Y_labeled, sampling_rate, path)

    for cls in SUPERCLASSES:
        Y_labeled[cls] = Y_labeled['diagnostic_superclass'].apply(lambda x: int(cls in x))

    keep_cols = [c for c in Y_labeled.columns if c not in COLS_TO_DROP]
    Y_clean = Y_labeled[keep_cols].copy()

    if 'age' in Y_clean.columns:
        age_median = Y_clean['age'].median()
        Y_clean['age'] = Y_clean['age'].fillna(age_median)

    if 'weight' in Y_clean.columns:
        weight_median = Y_clean['weight'].median()
        Y_clean['weight'] = Y_clean['weight'].fillna(weight_median)

    for col in ['nurse', 'site']:
        if col in Y_clean.columns:
            col_mode = Y_clean[col].mode()
            if not col_mode.empty:
                Y_clean[col] = Y_clean[col].fillna(col_mode[0])

    if 'device' in Y_clean.columns:
        if Y_clean['device'].isna().any():
            device_mode = Y_clean['device'].mode()
            if not device_mode.empty:
                Y_clean['device'] = Y_clean['device'].fillna(device_mode[0])
        Y_clean['device'] = Y_clean['device'].astype('category').cat.codes

    return X_labeled, Y_clean


if __name__ == '__main__':
    X, Y_clean = prepare_dataset()
    print(f'Loaded dataset: X shape {X.shape}, Y_clean shape {Y_clean.shape}')


