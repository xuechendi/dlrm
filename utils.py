import timeit
import sklearn.metrics
import numpy as np
import torch
import os
import pandas as pd
import os
from tqdm import tqdm

class Timer:
    level = 0
    viewer = None
    def __init__(self, name):
        self.name = name
        if Timer.viewer:
            Timer.viewer.display(f"{name} started ...")
        else:
            print(f"{name} started ...")

    def __enter__(self):
        self.start = timeit.default_timer()
        Timer.level += 1

    def __exit__(self, *a, **kw):
        Timer.level -= 1
        if Timer.viewer:
            Timer.viewer.display(
                f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')
        else:
            print(
                f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')

def fix_na(df):
    df = df.fillna(0)
    for col in df.select_dtypes([float]):
        v = col
        if np.array_equal(df[v], df[v].astype(int)):
            df[v] = df[v].astype(int, copy = False)
    return df
   
def load_csv_to_pandasdf(dataset):
    if not isinstance(dataset, str):
        raise NotImplementedError("Only support pandas Dataframe as input")
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"{dataset} is not exists")
    if os.path.isdir(dataset):
        input_files = sorted(os.listdir(dataset))
        df = pd.read_csv(dataset + "/" + input_files[0], sep = '\t')
        for file in tqdm(input_files[1:]):
            part = pd.read_csv(dataset + "/" + file, sep = '\t')    
            df = pd.concat([df, part],axis=0)
    else:
        df = pd.read_csv(dataset, sep = '\t')
    df = fix_na(df)
    return df

def H(y, p):
    e = np.finfo(float).eps
    return -y * np.log(p + e) - (1 - y) * np.log(1 - p + e)
    
def nce_score(y_true, y_pred, verbose = False):
    avg_logloss_y_p = np.mean(sklearn.metrics.log_loss(y_true, y_pred))
    #avg_log_reci_p = np.mean(np.log(1/y_pred))
    ctr = y_true.sum() / y_true.shape[0]
    logloss_ctr = H(ctr, ctr)
    if verbose:
        print(f"avg_logloss_y_p is {avg_logloss_y_p}, y_true_num is {y_true.sum()}, y_total is {y_true.shape[0]}, logloss_ctr is {logloss_ctr}")
    return avg_logloss_y_p / logloss_ctr

class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self):
        super().__init__(reduction="mean")
        
    def forward(self, input, target):
        input = input * 2 - 1
        return super().forward(input, target)

class customBCELoss(torch.nn.BCELoss):
    def __init__(self, pos_weight = None):
        super().__init__(reduction="none" if pos_weight is not None else 'mean')
        self.pos_weigth = pos_weight

    def forward(self, input, target):
        l = super().forward(input, target)
        if self.pos_weigth is None:
            return l
        else:
            # target is one, then we need to multiple pos_weight
            mask_pos = target == 1
            l[mask_pos] = l[mask_pos] * self.pos_weigth
            return torch.mean(l)

class NCELoss(customBCELoss):
    def __init__(self, pos_weight = None):
        super().__init__(pos_weight = pos_weight)

    def forward(self, input, target):
        n = super().forward(input, target)
        ctr = target.sum() / target.shape[0]
        d = H(ctr, ctr)
        return n/d

class CombinedAdam:
    def __init__(self, params_list):
        opt_sparse = torch.optim.SparseAdam(params_list[0]["params"], lr=params_list[0]["lr"])
        opt_dense_bot = torch.optim.Adam(params_list[1]["params"], lr=params_list[1]["lr"])
        opt_dense_top = torch.optim.Adam(params_list[2]["params"], lr=params_list[2]["lr"])
        self.optimizers = [opt_sparse, opt_dense_bot, opt_dense_top]

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

def categorify(X_cat_df, label_encoders = None, inplace = False):
    from sklearn import preprocessing
    emb_dim = {}
    if label_encoders is None:
        with Timer(f"started to categorify below columns {X_cat_df.columns}"):
            label_encoders = dict()
            for v in X_cat_df.columns:
                encoder = preprocessing.LabelEncoder()
                s = encoder.fit_transform(X_cat_df[v])
                if inplace:
                    del X_cat_df[v]
                    X_cat_df[v] = s
                else:
                    X_cat_df[f"{v}_cat"] = s
                label_encoders[v] = encoder
        for v, le in label_encoders.items():
            emb_dim[v] = len(le.classes_) + 1
    else:
        label_encoders = label_encoders
        for v, le in label_encoders.items():
            le_dict = dict((v, idx) for idx, v in enumerate(le.classes_))
            s = X_cat_df[v].apply(lambda x: le_dict.get(x, len(le.classes_)))
            X_cat_df[f"{v}_cat"] = s
    
    return X_cat_df, label_encoders, emb_dim

def group_categorify(df, feature_name, grouped_features):
    k = [i for i in df.columns if i not in grouped_features]
    if len(k) == 0:
        raise NotImplementedError("df contains all the grouped keys, not support yet")
    k = k[0]
    encoder = df.groupby(by = grouped_features, as_index = False)[k].count().drop(k, axis = 1)
    encoder[feature_name] = encoder.index
    ret = df.merge(encoder, on = grouped_features)
    return ret, encoder, encoder.shape[0] + 1