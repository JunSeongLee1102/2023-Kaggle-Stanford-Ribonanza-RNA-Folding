#!/usr/bin/env python3
import gc
import logging
import os
import random
import sys
import time
import warnings
from glob import glob

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import rich
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
)
from rich import print

sys.path.append("../main")
from bottle import RNA_Lightning
from data import RNA_DM
from utils import (
    ExCB,
    TBLogger,
    grid_search,
    merge_train_infer,
    kfold,
    collate_preds,
    submission,
    epoch_preds_averaging,
    count_parameters,
)

if __name__ == "__main__":
    debug = 0
    n_workers = 4
    pred = True
    pred_version = None  # nth directory's checkpoints will be loaded (starting from 0), None for training
    ckpt = True
    load_ckpt_path = "PL_pretrained_checkpoint.ckpt"
    seed = 420
    early_stop = 20
    n_trials = 0
    n_folds = 5
    run_folds = [1,2]  # range(n_folds)
    log_dir = "231217_lstm_gru_pseudo_reproduce_5fold"
    pred_dir = "subs"
    pretraining = False
    metric = "loss/V"
    hp_conf = {
        "n_epochs": 200,
        "lr": 2e-3,  # 4e-4, #2e-3,
        "lr_warmup": 0.015,  # 0.1, #0.015,
        "wt_decay": 1e-1,
        "grad_clip": 5,
        "batch_size": 167,
        "n_grad_accum": 3,
        "n_mem": 0,
        "sn_min": 0.6,
        "stochastic_weight_average": False,
        "aux_loop": [None, "eterna"][0],
        "aux_struct": [None, "eterna"][0],
        "aux_scale": 0.1,
        "aux_smooth": 0.1,
        "p_flip": 0.5,
        "emb_grad_frac": 1,
        "norm_layout": "dual",
        "pos_bias_heads": 6,
        "pos_bias_params": (32, 128),
        "pos_rope": False,
        "pos_sine": False,
        "norm_rms": True,
        "norm_lax": False,
        "qkv_bias": False,
        "ffn_bias": False,
        "ffn_multi": 4,
        "n_layers": 12,
        "n_heads": 6,
        "d_heads": 48,
        "p_dropout": 0.1,
        "att_fn": ["sdpa", "xmea"][1],
        "n_folds": n_folds,
        "seed": seed,
        "note": "",
        "n_layers_rnn": 1,
        "n_heads_rnn": 2,
        "kernel_size_gc": 3,
        "use_lenmatched_batch": True,
        "val_flip": True,
        "tta_flip": True,
        "epoch_ensemble": 5,
    }
    hp_skips = []
    df_train = "../data/train_data_processed_ALL_2.parquet"
    df_infer = "../data/test_sequences_processed_ALL.parquet"
    df_pseudo = "../data/submission_for_pseudo_v2.parquet"
    df_mat_train = "../data/train_sparse_bpps"
    df_mat_test = "../data/test_sparse_bpps"
    try:
        with rich.get_console().status("Reticulating Splines"):
            if not debug:
                warnings.filterwarnings("ignore")
                for n in logging.root.manager.loggerDict:
                    logging.getLogger(n).setLevel(logging.WARN)
            torch.set_float32_matmul_precision("medium")
            torch.manual_seed(seed)
            random.seed(seed)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True)
            df_infer = pd.read_parquet(df_infer) if (pred | pretraining) else None
            df_train = pd.read_parquet(df_train)
            if pretraining:
                df_pseudo = pd.read_parquet(df_pseudo)
                df_mat_train = df_mat_train[:-18]
                df_mat_test = df_mat_test[:-17]
                df_train = merge_train_infer(df_train, df_infer, df_pseudo)
                # df_train = df_train.sample(frac = 0.1) # for debugging
            folds = kfold(df_train, n_folds, seed)
            trials = grid_search(hp_conf, hp_skips)
            n_trials = len(trials) if not n_trials else n_trials
            n_trials = len(trials) if len(trials) < n_trials else n_trials
        print(f"Log: {log_dir} | EStop: {early_stop} | Ckpt: {ckpt} | Pred: {pred}")
        for i, hp in enumerate(trials[:n_trials]):
            for j, f in enumerate(run_folds):
                print(f"Trial {i + 1}/{n_trials} Fold {j + 1}/{len(run_folds)} ({f})")
                hp.fold = f
                tbl = TBLogger(os.getcwd(), log_dir, default_hp_metric=False)
                cb = [RichProgressBar(), ExCB()]
                cb += (
                    [
                        ModelCheckpoint(
                            tbl.log_dir, None, metric, save_top_k=hp.epoch_ensemble
                        )
                    ]
                    if ckpt
                    else []
                )
                cb += [EarlyStopping(metric, 0, early_stop)] if early_stop else []
                if hp.stochastic_weight_average:
                    cb += [
                        StochasticWeightAveraging(
                            swa_lrs=1e-2, swa_epoch_start=10, annealing_epochs=10
                        )
                    ]
                dm = RNA_DM(
                    hp,
                    n_workers,
                    df_infer,
                    df_train,
                    folds[f],
                    df_mat_train,
                    df_mat_test,
                )
                model = RNA_Lightning(hp)
                if load_ckpt_path is not None:
                    model.model.load_state_dict(
                        torch.load(load_ckpt_path), strict=False
                    )
                #count_parameters(model.model)
                trainer = pl.Trainer(
                    precision="16-mixed",
                    accelerator="gpu",
                    benchmark=True,
                    max_epochs=hp.n_epochs,
                    accumulate_grad_batches=hp.n_grad_accum,
                    gradient_clip_val=hp.grad_clip,
                    fast_dev_run=debug,
                    num_sanity_val_steps=0,
                    enable_model_summary=False,
                    logger=tbl,
                    callbacks=cb,
                )
                gc.collect()
                if pred_version is None:
                    try:
                        trainer.fit(model, datamodule=dm)
                    except KeyboardInterrupt:
                        print("Fit Interrupted")
                        if i + 1 < n_trials:
                            with rich.get_console().status("Quit?") as s:
                                for k in range(3):
                                    s.update(f"Quit? {3-k}")
                                    time.sleep(1)
                        continue
                if pred:
                    try:
                        # cp = None if debug else "best"
                        # Epoch ensemble
                        predss = []
                        cur_log_dir = sorted(glob(f"{log_dir}/*"))[
                            tbl.version if pred_version is None else pred_version
                        ]  # should be checked
                        fpaths = sorted(glob(f"{cur_log_dir}/*ckpt"))
                        inds = np.zeros(hp.epoch_ensemble, int)
                        for i in range(0, hp.epoch_ensemble):
                            inds[i] = int(fpaths[i].split("/")[-1][6:].split("-")[0])
                        ind_order = np.argsort(inds)
                        for i in range(0, hp.epoch_ensemble):
                            cp = fpaths[ind_order[i]]
                            preds = trainer.predict(model, datamodule=dm, ckpt_path=cp)
                            preds = collate_preds(preds)
                            predss.append(preds)
                        avg_preds = epoch_preds_averaging(predss)
                        if "loss" in preds:
                            print(f"Loop Loss: {preds['loss']:.4f}")
                    except KeyboardInterrupt:
                        print("Prediction Interrupted")
                        continue
                    with rich.get_console().status("Processing Submission"):
                        fn = f"{pred_dir}/{log_dir}v{tbl.version:02}"
                        submission(avg_preds, fn if not debug else None)
                        del predss, preds, avg_preds
                        gc.collect()
    except KeyboardInterrupt:
        print("Goodbye")
        sys.exit()
