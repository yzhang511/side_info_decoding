import os
import sys
from datetime import date
import random
from pathlib import Path
import numpy as np

from scipy.linalg import svd
from sklearn.model_selection import StratifiedKFold

from iblatlas.atlas import AllenAtlas

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import accuracy
from torchmetrics import AUROC
import lightning as L 
from lightning.pytorch.utilities import CombinedLoader

from side_info_decoding.utils import set_seed

class SessionDataset:
    def __init__(self, dataset, roi_idx, pid_idx, **kargs):
        self.xs, self.ys = dataset
        self.n_trials, self.n_units, _ = self.xs.shape
        self.roi_idx = roi_idx
        self.pid_idx = pid_idx
        
    def __len__(self):
        return self.n_trials
    
    def __getitem__(self, index):
        return self.xs[index], self.ys[index], self.roi_idx, self.pid_idx
    
def dataloader(datasets, roi_idxs, pid_idxs, batch_size=32): 
    loaders = []
    for i, dataset in enumerate(datasets):
        sess_dataset = SessionDataset(dataset, roi_idxs[i], pid_idxs[i])
        loaders.append(DataLoader(
            sess_dataset, batch_size = batch_size
        ))
    return loaders

class Hier_Reduced_Rank_Model(nn.Module):
    def __init__(
        self, 
        n_roi,
        n_units, 
        n_t_bin, 
        rank_V,
        rank_B
    ):
        super(Hier_Reduced_Rank_Model, self).__init__()
        
        self.n_roi = n_roi
        self.n_sess = len(n_units)
        self.n_units = n_units
        self.n_t_bin = n_t_bin
        self.rank_V = rank_V
        self.rank_B = rank_B
        
        self.Us = nn.ParameterList(
            [nn.Parameter(torch.randn(self.n_units[i], self.rank_V)) for i in range(self.n_sess)]
        )
        self.A = nn.Parameter(torch.randn(self.n_roi, self.rank_V, self.rank_B)) 
        self.B = nn.Parameter(
            torch.randn(self.rank_B, self.n_t_bin)
        ) 
        self.intercepts = nn.ParameterList(
            [nn.Parameter(torch.randn(1,)) for i in range(self.n_sess)]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, datasets):
        pred_lst, gt_lst = [], []
        for dataset in datasets:
            X, Y, roi_idx, sess_idx = dataset
            roi_idx = torch.unique(roi_idx)
            sess_idx = torch.unique(sess_idx)
            n_trials, n_units, n_t_bins = X.shape
            self.Vs = torch.einsum("ijk,kt->ijt", self.A, self.B)
            Beta = torch.einsum("cr,rt->ct", self.Us[sess_idx], self.Vs[roi_idx].squeeze())
            out = torch.einsum("ct,kct->k", Beta, X)
            out += self.intercepts[sess_idx] * torch.ones(n_trials)
            out = self.sigmoid(out)
            pred_lst.append(out)
            gt_lst.append(Y)
        return pred_lst, gt_lst
    
class LitHierRRR(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch):
        losses = 0
        pred_lst, gt_lst = self.model(batch)
        for i in range(len(batch)):
            losses += nn.BCELoss()(pred_lst[i], gt_lst[i])
        loss = losses / len(batch)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch):
        accs, aucs = self._shared_eval_step(batch)
        metrics = {"val_acc": np.mean(accs), "val_auc": np.mean(aucs)}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch):
        accs, aucs = self._shared_eval_step(batch)
        for i in range(len(batch)):
            print(f"session {i} test_acc {accs[i]} test_auc {aucs[i]}")

    def _shared_eval_step(self, batch):
        pred_lst, gt_lst = self.model(batch)
        accs, aucs = [], []
        for i in range(len(batch)):
            auroc = AUROC(task="binary")
            acc = accuracy(pred_lst[i], gt_lst[i], task="binary")
            auc = auroc(pred_lst[i], gt_lst[i])
            accs.append(acc)
            aucs.append(auc)
        return accs, aucs
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-3)
        return optimizer

seed = 666
set_seed(seed)

if __name__ == "__main__":
    
    # -- args
    ap = argparse.ArgumentParser()
    g = ap.add_argument_group("Data Input/Output")
    g.add_argument("--in_path", type=str)
    g.add_argument("--out_path", type=str)
    g.add_argument("--atlas_level", type=int, default=7)
    g.add_argument("--n_rank_V", type=int, default=2)
    g.add_argument("--n_rank_B", type=int, default=5)
    g.add_argument("--n_epochs", type=int, default=1000)
    args = ap.parse_args()
    
    seed = 666
    set_seed(seed)
    
    ba = AllenAtlas()
    regions = np.unique(ba.regions.acronym[ba.regions.level == args.atlas_level])
    print("=================")
    print("Fitting model for regions: ", regions)
    
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    xval_dict = {}
    for fold_idx in range(n_folds):

        print("=================")
        print(f"Started training on {fold_idx+1} / {n_folds} folds ..")

        train_datasets, test_datasets = [], []
        lst_units, lst_regions, lst_sessions, lst_region_names, lst_pids = [], [], [], [], []

        pid_idx = 0
        for roi_idx, roi in enumerate(regions):

            f_names = os.listdir(args.in_path/roi)
            pids = [f_name.split("_")[1].split(".")[0] for f_name in f_names]

            print("=================")
            print(f"Loading {len(pids)} PIDs in region {roi}:")
            for pid in pids:
                print(pid)

            data_dict = np.load(args.in_path/roi/f"pid_{pid}.npy", allow_pickle=True).item()

            for _, pid in enumerate(pids):
                xs = data_dict["neural_contrast"]["all"]
                ys = data_dict["choice_contrast"]["all"]
                for counter, (train, test) in enumerate(skf.split(xs, ys)):
                    if counter == fold_idx:
                        train_xs, test_xs = xs[train], xs[test]
                        train_ys, test_ys = ys[train], ys[test]
                train_datasets.append((train_xs, train_ys))
                test_datasets.append((test_xs, test_ys))
                lst_units.append(data_dict["meta"]["n_units"])
                lst_regions.append(roi_idx)
                lst_region_names.append(roi)
                lst_sessions.append(pid_idx)
                lst_pids.append(pid)
                pid_idx += 1

        train_loaders = dataloader(train_datasets, lst_regions, lst_sessions, batch_size=128)
        test_loaders = dataloader(test_datasets, lst_regions, lst_sessions, batch_size=128)
        train_loaders = CombinedLoader(train_loaders, mode="min_size")
        test_loaders = CombinedLoader(test_loaders, mode="min_size")

        hier_rrr = Hier_Reduced_Rank_Model(
            n_roi = len(regions),
            n_units = lst_units, 
            n_t_bin = data_dict["meta"]["n_t_bins"], 
            rank_V = args.n_rank_V,
            rank_B = args.n_rank_B
        )

        lit_hier_rrr = LitHierRRR(hier_rrr)
        trainer = L.Trainer(max_epochs=args.n_epochs)
        trainer.fit(model=lit_hier_rrr, 
                    train_dataloaders=train_loaders)

        accs_per_batch, aucs_per_batch = [], []
        for batch in test_loaders:
            accs, aucs = [], []
            pred_lst, gt_lst = hier_rrr(batch[0])
            for i in range(len(batch[0])):
                auroc = AUROC(task="binary")
                accs.append(accuracy(pred_lst[i], gt_lst[i], task="binary").item())
                aucs.append(auroc(pred_lst[i], gt_lst[i]).item())
            accs_per_batch.append(accs)
            aucs_per_batch.append(aucs)
        test_accs = np.mean(accs_per_batch, 0)
        test_aucs = np.mean(aucs_per_batch, 0)
        print("Accuracy: ", test_accs)
        print("AUC: ", test_aucs)

        xval_dict.update({fold_idx: {}})
        xval_dict[fold_idx].update({"accs": test_accs})
        xval_dict[fold_idx].update({"aucs": test_aucs})
        xval_dict[fold_idx].update({"pid_idxs": lst_sessions})
        xval_dict[fold_idx].update({"regions_idxs": lst_regions})
        xval_dict[fold_idx].update({"region_names": lst_region_names})
        xval_dict[fold_idx].update({"pids": lst_pids})

        print("=================")
        print(f"Finished training on {fold_idx+1} / {n_folds} folds ..")

    np.save(args.out_path/f"xval_{date.today()}.npy", xval_dict)

    