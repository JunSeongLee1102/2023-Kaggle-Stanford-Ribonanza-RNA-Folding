import itertools
import os
import random
from argparse import Namespace
import pickle
import rapidgzip
from scipy import sparse

import lightning.pytorch as pl
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def decompress_rapidgzip(name):
    with rapidgzip.open(name, parallelization=os.cpu_count()) as f:
        data = pickle.load(f)
    return data


def load_pickle(name):
    with open(name, "rb") as f:
        data = pickle.load(f)
    return data


class RNA_DS(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sn_min: int | float = 0,
        err_max: float = 0,
        aux_loop: str | None = None,
        aug_loop: bool = False,
        p_flip: float = 0.0,
        path_mats: str | None = None,
    ):
        if sn_min > 0:
            if isinstance(sn_min, int):
                df = df.loc[(df.SN_DMS >= sn_min) & (df.SN_2A3 >= sn_min)]
            else:
                df = df.loc[df.SN_DMS + df.SN_2A3 >= sn_min * 2]
            df = df.reset_index(drop=True)
        # print(f"sn_min:{sn_min} len:{len(df)}")
        self.err_max = err_max
        self.aux_loop = "aug" if aug_loop else aux_loop
        self.p_flip = p_flip
        self.infer = sn_min < 0
        self.len_max = (df.sequence if self.infer else df.seq).apply(len).max()
        self.len_arr = np.array((df.sequence if self.infer else df.seq).apply(len))
        self.kmap_seq = {x: i for i, x in enumerate("_AUGC")}
        self.kmap_loop = {x: i for i, x in enumerate("_SMIBHEX")}
        self.df = df
        self.path_mats = path_mats

    def __len__(self):
        return len(self.df)

    def _pad(self, x: torch.Tensor):
        z = [0] * (1 + (x.ndim - 1) * 2)
        v = float("nan") if x.dtype == torch.float else 0
        return F.pad(x, z + [self.len_max - len(x)], value=v)

    def pad(self, d: dict):
        return {k: self._pad(v) for k, v in d.items()}

    def flip(self, d: dict):
        return {k: v.flip(0) for k, v in d.items()}

    def get_seq(self, r):
        seq = r.sequence if self.infer else r.seq
        seq = torch.IntTensor([self.kmap_seq[_] for _ in seq])
        mask = torch.zeros(len(seq), dtype=torch.bool)
        mask[:] = True
        return {"seq": seq, "mask": mask}

    def get_react(self, r):
        react = torch.Tensor(np.array([r.react_DMS, r.react_2A3]))
        error = torch.Tensor(np.array([r.error_DMS, r.error_2A3]))
        if self.err_max > 0:
            react[error > self.err_max] = float("nan")
        react = react.transpose(0, 1).clip(0, 1)
        return {"react": react}

    def get_mats(self, r):
        seq_id = r.sequence_id if self.infer else r.seq_id
        mat = load_pickle(f"{self.path_mats}/{seq_id}.pkl")
        bpp = np.array(mat["bpp"].todense())
        structure = np.array(mat["structure"].todense())
        len_seq = len(bpp)
        cat_mat = torch.zeros(self.len_max, self.len_max, 2)
        cat_mat[:len_seq, :len_seq, 0] = torch.from_numpy(bpp)
        cat_mat[:len_seq, :len_seq, 1] = torch.from_numpy(structure)
        return {"As": cat_mat}

    def get_loop(self, r):
        match self.aux_loop:
            case "eterna":
                loop = r.eterna_loop
            case "contra":
                loop = r.contra_loop
            case "aug":
                loop = r.eterna_loop if random.random() < 0.5 else r.contra_loop
        loop = torch.LongTensor([self.kmap_loop[_] for _ in loop])
        return {"loop": loop}

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        out = self.get_seq(r)
        mat = None
        len_seq = len(out["seq"])
        if self.infer:
            if self.path_mats is not None:
                mat = self.get_mats(r)
            if random.random() <= self.p_flip:
                out = self.flip(out)
                if mat is not None:
                    mat["As"][:len_seq, :len_seq] = torch.flip(
                        mat["As"][:len_seq, :len_seq], [0, 1]
                    )
            return self.pad(out) | mat
        out = out | self.get_react(r)
        out = out | (self.get_loop(r) if self.aux_loop else {})
        if self.path_mats is not None:
            mat = self.get_mats(r)
        if random.random() <= self.p_flip:
            out = self.flip(out)
            if mat is not None:
                mat["As"][:len_seq, :len_seq] = torch.flip(
                    mat["As"][:len_seq, :len_seq], [0, 1]
                )
        if mat is not None:
            return self.pad(out) | mat
        else:
            return self.pad(out)


def collate_fn(samples):
    list_keys = list(samples[0].keys())
    output = {}
    truncated_output = {}
    for k in list_keys:
        output[k] = torch.stack([sample[k] for sample in samples], 0)
    max_len = output["mask"].sum(-1).max()
    for k in list_keys:
        if len(output[k].shape) == 4:  # Matrix
            truncated_output[k] = output[k][:, :max_len, :max_len]
        else:
            truncated_output[k] = output[k][:, :max_len]
    return truncated_output


class SingleGPULenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):
        yielded_batches = []
        buckets = [[]] * 100
        yielded = 0
        for idx in self.sampler:
            L = self.sampler.data_source.len_arr[idx]
            L = max(1, L // 16)
            if len(buckets[L]) == 0:
                buckets[L] = []
            buckets[L].append(idx)
            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                # yield batch
                yielded_batches.append(batch)
                yielded += 1
                buckets[L] = []
        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]
        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                # yield batch
                yielded_batches.append(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            # yield batch
            yielded_batches.append(batch)
        batch_order = np.arange(0, len(yielded_batches), dtype=int)
        np.random.shuffle(batch_order)
        for idx in range(0, len(yielded_batches)):
            yield yielded_batches[batch_order[idx]]


class MultiGPULenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):
        yielded_batches = []
        buckets = [[]] * 100
        yielded = 0
        for idx in self.sampler:
            L = self.sampler.dataset.len_arr[idx]
            L = max(1, L // 16)
            if len(buckets[L]) == 0:
                buckets[L] = []
            buckets[L].append(idx)
            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                # yield batch
                yielded_batches.append(batch)
                yielded += 1
                buckets[L] = []
        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]
        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                # yield batch
                yielded_batches.append(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            # yield batch
            yielded_batches.append(batch)
        batch_order = np.arange(0, len(yielded_batches), dtype=int)
        np.random.shuffle(batch_order)
        for idx in range(0, len(yielded_batches)):
            yield yielded_batches[batch_order[idx]]


class RNA_DM(pl.LightningDataModule):
    def __init__(
        self,
        hp: Namespace,
        n_workers: int = 0,
        df_infer: pd.DataFrame | None = None,
        df_train: pd.DataFrame | None = None,
        fold_idxs: list | None = None,
        path_mat_train: str | None = None,
        path_mat_test: str | None = None,
    ):
        super().__init__()
        self.df_train = None
        self.df_val = None
        self.mat_train = None
        self.mat_test = None
        self.path_mat_train = path_mat_train
        self.path_mat_test = path_mat_test
        if fold_idxs and df_train is not None:
            self.df_train = df_train.iloc[fold_idxs[0]]
            self.df_val = df_train.iloc[fold_idxs[1]]
        self.df_infer = df_infer
        self.sn_min = getattr(hp, "sn_min", 0)
        self.err_max = getattr(hp, "err_max", 0)
        self.aux = getattr(hp, "aux_loop", None)
        self.aug = getattr(hp, "aug_loop", False)
        self.p_flip = hp.p_flip
        self.flip_val = getattr(hp, "val_flip", False)
        self.flip_tta = getattr(hp, "tta_flip", False)
        self.batch_size = hp.batch_size
        self.kwargs = {
            "num_workers": n_workers,
            "pin_memory": bool(n_workers),
        }
        self.is_multi_gpu = torch.cuda.device_count() > 1
        self.use_lenmatched_batch = hp.use_lenmatched_batch

    def train_dataloader(self):
        assert self.df_train is not None
        ds = RNA_DS(
            self.df_train,
            self.sn_min,
            self.err_max,
            self.aux,
            self.aug,
            p_flip=self.p_flip,
            path_mats=self.path_mat_train,
        )
        if self.use_lenmatched_batch:
            sampler = torch.utils.data.RandomSampler(ds)
            if self.is_multi_gpu:
                sampler = MultiGPULenMatchBatchSampler(
                    sampler, batch_size=self.batch_size, drop_last=False
                )
            else:
                sampler = SingleGPULenMatchBatchSampler(
                    sampler, batch_size=self.batch_size, drop_last=False
                )
            return DataLoader(
                ds, batch_sampler=sampler, collate_fn=collate_fn, **self.kwargs
            )
        else:
            kwargs = self.kwargs | {"shuffle": True}
            return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, **kwargs)

    def val_dataloader(self):
        assert self.df_val is not None
        dl = DataLoader(
            RNA_DS(
                self.df_val,
                1,
                -1,
                self.aux,
                False,
                p_flip=0.0,
                path_mats=self.path_mat_train,
            ),
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            **self.kwargs,
        )
        dl = (
            [
                dl,
                DataLoader(
                    RNA_DS(
                        self.df_val,
                        1,
                        -1,
                        self.aux,
                        False,
                        p_flip=1.0,
                        path_mats=self.path_mat_train,
                    ),
                    collate_fn=collate_fn,
                    batch_size=self.batch_size,
                    **self.kwargs,
                ),
            ]
            if self.flip_val
            else dl
        )
        return dl

    def predict_dataloader(self):
        assert self.df_infer is not None
        dl = DataLoader(
            RNA_DS(
                self.df_infer,
                -1,
                -1,
                self.aux,
                False,
                p_flip=0.0,
                path_mats=self.path_mat_test,
            ),
            collate_fn=collate_fn,
            **self.kwargs,
            batch_size=self.batch_size,
        )
        dl = (
            [
                dl,
                DataLoader(
                    RNA_DS(
                        self.df_infer,
                        -1,
                        -1,
                        self.aux,
                        False,
                        p_flip=1.0,
                        path_mats=self.path_mat_test,
                    ),
                    collate_fn=collate_fn,
                    **self.kwargs,
                    batch_size=self.batch_size,
                ),
            ]
            if self.flip_tta
            else dl
        )
        return dl
