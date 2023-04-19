import torch
from torch.utils.data import Dataset, RandomSampler
import numpy as np
import pandas as pd

class DLRMDataset(Dataset):
    def __init__(
        self,
        dataset,
    ):
        self.X_int = np.array()  # continuous  feature
        self.X_cat = np.array()  # categorical feature
        self.y = np.array()

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [
                self[idx]
                for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]

        i = index
        return self.X_float[i], self.X_cat[i], self.y[i] if self.y is not None else -1

    def __len__(self):
        return self.X_cat.shape[0]

class DLRMDataset_PandasDF(DLRMDataset):
    def __init__(
        self,
        emb_dim,
        dataset,
        labels,
        target_label
    ):
        if not isinstance(dataset, pd.DataFrame):
            raise NotImplementedError("Only support pandas Dataframe as input")
        df = dataset
        if target_label in df.columns:
            self.y = df[target_label].to_numpy()
        else:
            self.y = None
        feature_df = df.loc[:, ~df.columns.isin(labels)]
        self.X_cat_df = feature_df.loc[:, feature_df.columns.isin(emb_dim.keys())]
        self.X_float_df = feature_df.loc[:, ~feature_df.columns.isin(emb_dim.keys())]
        
        self.counts = np.array(list(emb_dim.values()))
        
        self.X_cat = self.X_cat_df.to_numpy()  # categorical feature
        self.X_float = self.X_float_df.to_numpy() # continuous  feature

        self.m_den = self.X_float.shape[1]
        self.n_emb = len(self.counts)

        print("Embedding features= %d, MLP features= %d" % (self.n_emb, self.m_den))
    
def collate_wrapper_criteo_offset(list_of_tuples):
    # where each tuple is (X_int, X_cat, y)
    transposed_data = list(zip(*list_of_tuples))
    X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    lS_i = [X_cat[:, i] for i in range(featureCnt)]
    lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]

    return X_int, torch.stack(lS_o), torch.stack(lS_i), T

def make_data_loaders(args):
    train_data = DLRMDataset_PandasDF(
        emb_dim = args.emb_dim,
        dataset = args.train_data,
        labels = args.labels,
        target_label = args.target_label
    )

    test_data = DLRMDataset_PandasDF(
        emb_dim = args.emb_dim,
        dataset = args.valid_data,
        labels = args.labels,
        target_label = args.target_label
    )

    collate_wrapper_criteo = collate_wrapper_criteo_offset

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,  # True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_mini_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,  # True
    )

    return train_data, train_loader, test_data, test_loader