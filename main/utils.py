import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedGroupKFold
from apex.normalization import FusedLayerNorm, FusedRMSNorm
from prettytable import PrettyTable


def sort_weight_decay_params(model: nn.Module) -> tuple[list, list]:
    # https://github.com/karpathy/minGPT
    whitelist = (
        nn.Linear,
        nn.MultiheadAttention,
        nn.GRU,
    )
    blacklist = (
        nn.Embedding,
        nn.LayerNorm,
        FusedLayerNorm,
        FusedRMSNorm,
    )
    decay, no_decay, leftovers = set(), set(), set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if any(s in pn for s in ("bias")):
                no_decay.add(fpn)
            elif pn in ("scale", "mem"):
                no_decay.add(fpn)
            elif "weight" in pn and isinstance(m, whitelist):
                decay.add(fpn)
            elif "weight" in pn and isinstance(m, blacklist):
                no_decay.add(fpn)
            else:
                leftovers.add(fpn)
    assert not leftovers
    pd = {pn: p for pn, p in model.named_parameters()}
    decay, no_decay = sorted(list(decay)), sorted(list(no_decay))
    decay = [pd[pn] for pn in sorted(list(decay))]
    no_decay = [pd[pn] for pn in no_decay]
    return decay, no_decay


def collate_preds(preds: list) -> dict:
    get_k, out = lambda k, ld: [p[k] for p in ld], {}
    preds = [preds] if isinstance(preds[0], dict) else preds
    for pred in preds:
        for k in pred[0]:
            v = torch.concat(get_k(k, pred))
            out[k] = out[k] + v if k in out else v
    out = {k: v / len(preds) for k, v in out.items()}
    if "loss" in out:
        out["loss"] = out["loss"].mean().item()
    return out


def epoch_preds_averaging(predss):
    n_preds = len(predss)
    avg_preds = torch.clone(predss[0]["react"])
    for i in range(1, len(predss)):
        avg_preds += predss[i]["react"]
    avg_preds /= n_preds
    avg_preds = {"react": avg_preds}
    return avg_preds


def submission(preds: torch.Tensor, fn: str) -> None:
    def mutate_map(df: pd.DataFrame, fname: str):
        id1, id2 = 269545321, 269724007
        shape, font_size = (391, 457), 6
        pred_DMS = df[id1 : id2 + 1]["reactivity_DMS_MaP"].to_numpy()
        pred_2A3 = df[id1 : id2 + 1]["reactivity_2A3_MaP"].to_numpy()
        fig = plt.figure()
        plt.subplot(121)
        plt.title(f"reactivity_DMS_MaP", fontsize=font_size)
        plt.imshow(pred_DMS.reshape(*shape), vmin=0, vmax=1, cmap="gray_r")
        plt.subplot(122)
        plt.title(f"reactivity_2A3_MaP", fontsize=font_size)
        plt.imshow(pred_2A3.reshape(*shape), vmin=0, vmax=1, cmap="gray_r")
        plt.tight_layout()
        plt.savefig(fname, dpi=500)
        plt.clf()
        plt.close()

    preds = pd.DataFrame(
        preds["react"].numpy().astype(np.float32),
        columns=["reactivity_DMS_MaP", "reactivity_2A3_MaP"],
    )
    preds.insert(0, "id", preds.index)
    if fn:
        preds.to_parquet(f"{fn}.parquet", index=False)
        mutate_map(preds, f"{fn}.png")


def grid_search(hp: dict, hp_skips: list) -> list:
    def search(hp: dict) -> list:
        kl = [k for k, v in hp.items() if type(v) == list]
        if not kl:
            args = Namespace()
            for k, v in hp.items():
                setattr(args, k, v)
            return [args]
        out = []
        for item in hp[kl[0]]:
            hp_ = hp.copy()
            hp_[kl[0]] = item
            out += search(hp_)
        return out

    def skip(hp: Namespace, hp_skips: list) -> bool:
        if not hp_skips:
            return False
        for hp_skip in hp_skips:
            for k, v in hp_skip.items():
                v = [v] if not isinstance(v, list) else v
                if not getattr(hp, k) in v:
                    match = False
                    break
                match = True
            if match:
                return True
        return False

    return [_ for _ in search(hp) if not skip(_, hp_skips)]


def get_nan_arr(L):
    nan_arr = np.zeros(L)
    nan_arr[:] = float("nan")
    return nan_arr


def merge_train_infer(df_train, df_infer, df_pseudo):
    df_infer["SN_2A3"] = 0.99
    df_infer["SN_DMS"] = 0.99
    df_infer.rename(columns={"sequence_id": "seq_id", "sequence": "seq"}, inplace=True)
    len_arr = np.array((df_infer.seq).apply(len))
    react_DMS = []
    react_2A3 = []
    for i in range(0, len(df_infer)):
        id_min = df_infer.loc[i, "id_min"]
        id_max = df_infer.loc[i, "id_max"]
        seq_len_nan_DMS = get_nan_arr(len_arr[i])
        seq_len_nan_2A3 = get_nan_arr(len_arr[i])
        reactivity_DMS_MaP_values = df_pseudo.loc[
            id_min:id_max, "reactivity_DMS_MaP"
        ].tolist()
        reactivity_2A3_MaP_values = df_pseudo.loc[
            id_min:id_max, "reactivity_2A3_MaP"
        ].tolist()
        seq_len_nan_DMS[: len(reactivity_DMS_MaP_values)] = reactivity_DMS_MaP_values
        seq_len_nan_2A3[: len(reactivity_2A3_MaP_values)] = reactivity_2A3_MaP_values
        react_DMS.append(seq_len_nan_DMS)
        react_2A3.append(seq_len_nan_2A3)
    df_infer["react_DMS"] = react_DMS
    df_infer["react_2A3"] = react_2A3
    df_train["seq_id"] = f"train_sparse_bpps/" + df_train["seq_id"]
    df_infer["seq_id"] = f"test_sparse_bpps/" + df_infer["seq_id"]
    df_train = pd.concat((df_train, df_infer))
    return df_train


def kfold(df: pd.DataFrame, n_folds: int, seed: int, cache_dir: str = "cache") -> list:
    fname = f"{cache_dir}/{n_folds}_{seed}.parquet"
    try:
        return pd.read_parquet(fname).values.tolist()
    except:
        pass
    folds = StratifiedGroupKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    folds = list(folds.split(df, df.seq.apply(len), df.seq_id))
    print(folds)
    os.makedirs(cache_dir, exist_ok=True)
    # pd.DataFrame(folds).to_parquet(fname)
    pd.DataFrame(folds, columns=["0", "1"]).to_parquet(fname)
    return folds


class ExCB(Callback):
    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            raise exception


class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step=None) -> None:
        metrics.pop("epoch", None)
        return super().log_metrics(metrics, step)

    @property
    def log_dir(self) -> str:
        version = (
            self.version
            if isinstance(self.version, str)
            else f"version_{self.version:02}"
        )
        log_dir = os.path.join(self.root_dir, version)
        if isinstance(self.sub_dir, str):
            log_dir = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
