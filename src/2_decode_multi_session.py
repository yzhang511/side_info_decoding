"""Example script for running multi-session reduced-rank model."""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from ray import tune
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from utils.data_loaders import MultiSessionDataModule
from models.decoders import MultiSessionReducedRankDecoder
from utils.eval import eval_multi_session_model
from utils.hyperparam_tuning import tune_decoder
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config

"""
-----------
USER INPUTS
-----------
"""
ap = argparse.ArgumentParser()
ap.add_argument(
    "--target", type=str, default="choice", 
    choices=["choice", "wheel-speed", "whisker-motion-energy", "pupil-diameter"]
)
ap.add_argument("--region", type=str, default="all")
ap.add_argument("--method", type=str, default="reduced_rank", choices=["reduced_rank"])
ap.add_argument("--n_workers", type=int, default=1)
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
args = ap.parse_args()


"""
-------
CONFIGS
-------
"""
kwargs = {"model": "include:src/configs/decoder.yaml"}

config = config_from_kwargs(kwargs)
config = update_config("src/configs/decoder.yaml", config)

if args.target in ["wheel-speed", "whisker-motion-energy", "pupil-diameter"]:
    config = update_config("src/configs/reg_trainer.yaml", config)
elif args.target in ['choice']:
    config = update_config("src/configs/clf_trainer.yaml", config)
else:
    raise NotImplementedError

if config.wandb.use:
    import wandb
    wandb.login()
    wandb.init(
        config=config,
        name="train_{}".format(args.method)
    )
set_seed(config.seed)

config["dirs"]["data_dir"] = Path(args.base_path)/config.dirs.data_dir
save_path = Path(args.base_path)/config.dirs.output_dir/args.target/('multi-sess-'+args.method)/args.region
ckpt_path = Path(args.base_path)/config.dirs.checkpoint_dir/args.target/('multi-sess-'+args.method)/args.region
os.makedirs(save_path, exist_ok=True)
os.makedirs(ckpt_path, exist_ok=True)


"""
---------
LOAD DATA
---------
"""
eids = [
    fname for fname in os.listdir(config.dirs.data_dir) if fname != "downloads"
]

print(eids)


"""
--------
DECODING
--------
"""
model_class = args.method

print('----------------------------------------------------')
print(f'Decode {args.target} in region {args.region} from {len(eids)} sessions:')
print(f'Launch multi-session {model_class} decoder:')

search_space = config.copy()
search_space['target'] = args.target
search_space['region'] = args.region if args.region != 'all' else None
search_space['training']['device'] = torch.device(
    'cuda' if np.logical_and(torch.cuda.is_available(), config.training.device == 'gpu') else 'cpu'
)

def train_func(config):
    
    configs = []
    for eid in eids:
        _config = config.copy()
        _config['eid'] = eid
        configs.append(_config)
    
    dm = MultiSessionDataModule(eids, configs)
    dm.setup()
    
    base_config = dm.configs[0].copy()
    base_config['n_units'] = [_config['n_units'] for _config in dm.configs]

    if model_class == "reduced_rank":
        model = MultiSessionReducedRankDecoder(base_config)
    else:
        raise NotImplementedError

    trainer = Trainer(
        max_epochs=config['tuner']['num_epochs'],
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=config['tuner']['enable_progress_bar'],
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)

# Hyper parameter tuning 

search_space['optimizer']['lr'] = tune.grid_search([1e-3])           # tune.grid_search([1e-2, 1e-3])
search_space['optimizer']['weight_decay'] = tune.grid_search([1e-1]) #tune.grid_search([0, 1e-1, 1e-2, 1e-3, 1e-4])

if model_class == "reduced_rank":
    search_space['temporal_rank'] = tune.grid_search([2]) #tune.grid_search([2, 5, 10, 15, 20])
    search_space['tuner']['num_epochs'] = 100 #500
    search_space['training']['num_epochs'] = 500 #800
else:
    raise NotImplementedError

results = tune_decoder(
    train_func, search_space, save_dir=ckpt_path,
    use_gpu=config.tuner.use_gpu, max_epochs=config.tuner.num_epochs, 
    num_samples=config.tuner.num_samples, num_workers=args.n_workers
)

best_result = results.get_best_result(metric=config.tuner.metric, mode=config.tuner.mode)
best_config = best_result.config['train_loop_config']

print("Best config:")
print(best_config)

# Model training 

checkpoint_callback = ModelCheckpoint(
    monitor=config.training.metric, mode=config.training.mode, dirpath=ckpt_path
)

trainer = Trainer(
    max_epochs=config.training.num_epochs, 
    callbacks=[checkpoint_callback], 
    enable_progress_bar=config.training.enable_progress_bar
)

configs = []
for eid in eids:
    config = best_config.copy()
    config['eid'] = eid
    configs.append(config)

dm = MultiSessionDataModule(eids, configs)
dm.setup()

best_config = dm.configs[0].copy()
best_config['n_units'] = [_config['n_units'] for _config in dm.configs]
best_config['eid_to_indx'] = {e: i for i,e in enumerate(eids)}
    
if model_class == "reduced_rank":
    model = MultiSessionReducedRankDecoder(best_config)
else:
    raise NotImplementedError

trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm, ckpt_path='best')


"""
----------
EVALUATION
----------
"""
metric_lst, test_pred_lst, test_y_lst = eval_multi_session_model(
    dm.train, dm.test, model, target=best_config['model']['target'], 
)

for eid_idx, eid in enumerate(eids):
    print(f'{eid} {args.target} test metric: ', metric_lst[eid_idx])
    
if config["wandb"]["use"]:
    wandb.log(
        {"eids": eids, "test_metric": metric_lst}
    )
    wandb.finish()
else:
    for eid_idx, eid in enumerate(eids):
        res_dict = {
            'test_metric': metric_lst[eid_idx], 
            'test_pred': test_pred_lst[eid_idx], 
            'test_y': test_y_lst[eid_idx]
        }
        np.save(save_path / f'{eid}.npy', res_dict)
        