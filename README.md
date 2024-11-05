Decode the full IBL reproducible ephys (RE) datasets.


### Environment setup

Create conda environment:
```
conda env create -f env.yaml
```

Activate the environment:
```
conda activate ibl_repro_ephys
```

Clone the IBL reproducible ephys repo:
```
git clone https://github.com/int-brain-lab/paper-reproducible-ephys.git
```

Install requirements and repo:
```
cd paper-reproducible-ephys
pip install -e .
```


### Datasets

Run the following to preprocess and cache [IBL](https://int-brain-lab.github.io/iblenv/index.html) datasets:
```
python src/0_data_caching.py --datasets reproducible-ephys --n_sessions 10 --base_path XXX
```

