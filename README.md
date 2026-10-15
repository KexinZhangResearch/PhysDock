# PhysDock: A Physics-Guided All-Atom Diffusion Model for Accurate Protein-Ligand Complex Prediction

---

```shell
****************************************************************************
*    ██████╗ ██╗  ██╗██╗   ██╗███████╗██████╗  ██████╗  ██████╗██╗  ██╗    *
*    ██╔══██╗██║  ██║╚██╗ ██╔╝██╔════╝██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝    *
*    ██████╔╝███████║ ╚████╔╝ ███████╗██║  ██║██║   ██║██║     █████╔╝     *
*    ██╔═══╝ ██╔══██║  ╚██╔╝  ╚════██║██║  ██║██║   ██║██║     ██╔═██╗     *
*    ██║     ██║  ██║   ██║   ███████║██████╔╝╚██████╔╝╚██████╗██║  ██╗    *
*    ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝    *
****************************************************************************
```

`PhysDock` is a **physics-guided** **all-atom** **denoising diffusion model** designed for predicting flexible protein-ligand complex structures. 

This repo is Implemented by KexinZhang (zhangkx2022@shanghaitech.edu.cn) & JialeYu (yujl2022@shanghaitech.edu.cn).

## Overview

PhysDock is a non-equivariant **physics-guided** all-atom denoising diffusion model designed for predicting flexible protein-ligand complex structures. 

![](./figs/PhysDockOverview.png)

<center>PhysDock Overview</center>

Currently, PhysDock supports **multi protein chains** and **only one small molecule ligand** input.

## Installation

* PhysDock needs to search the MSAs for the protein sequence of the receptor. Therefore, the corresponding tools need to be installed and five databases must be prepared.

```shell
# Install Bio Tools
apt install hhsuite hmmer

# Download Databases (UniRef90, MGnify, UniProt, Uniclust30, BFD) -> data/homo_data
# ~465 GB download, ~2.3 TB after extraction; parallel, resumable, checksum-verified
python scripts/download/download_homo_data.py                       # all five
python scripts/download/download_homo_data.py --only uniref90 bfd   # or a subset
```

* We provide a conda `.yaml` file to create environment to run PhysDock

```shell
# Create ENV
conda env create -f enviroment.yaml

# ENV Activation
conda activate PhysDock
```

* Download model params from Zenodo into `data/params` (`--link` makes `params/params.pt`, which the
  inference scripts load, a symlink to the download).

```shell
# Download latest EMA Params (params.pt) and the params used in the paper (params_2418.pt)
python scripts/download/download_params.py --link
```

All `scripts/download/download_*.py` share the same engine (`scripts/download/downloader.py`): every file is fetched with up to
`--workers` (default 96) parallel connections, each fetching small byte ranges, progress is checkpointed so an interrupted run resumes when
the same command is re-executed, every file is verified against the checksum published by its host (md5, or
crc32c / gzip-CRC where that is all the host provides), and archives are merged/extracted as soon as they are
verified. Source archives are kept under `data/<section>/tmp` unless `--delete-archives` is given. Use
`--dry-run` to see what would be downloaded and `python scripts/download/download_all.py` to fetch params, benchmarks and
the training dataset at once (`--all` adds the sequence databases). Everything lands in `<repo>/data`; only
`data/demo` is tracked by git.
`pip install google-crc32c` enables fast crc32c verification of the MGnify database.

## Inference

PhysDock provides convenient inference scripts for two scenarios: **redocking** and **virtual screening** (VS). Prior to redocking and VS, it is necessary to conduct corresponding preprocessing on a system. Subsequently, the cached files obtained from preprocessing are used for complex structure prediction and ranking.

### System Preparation

In PhysDock, a system encompasses one or more receptor chains along with a ligand. The input is **a `.pdb` file** of receptor and a `.sdf` file of ligand, while the output consists of the raw features (`.pkl.gz` file) for model input, typically structural-related features. In addition to structural features, PhysDock generates MSA-related features, including `msa_feature` and `uniprot_msa_feature`, by searching all sequence databases. We utilize the **MD5 encoding** of the protein sequence as a query to reuse the MSA features. Moreover, the raw feature can also encompass key residues involved in interactions processed by **PLIP**, and specify high-importance key residues with a certain probability during inference.

Every task has a Python entry point and a demo shell script in `scripts/`; the inference and training scripts read
their settings from `configs/` ([Hydra](https://hydra.cc)): every yaml value can be overridden as `key=value`
(nested keys with dots, also as extra arguments of the demo scripts) and `--cfg job` prints the resolved config.

```
scripts/
  download/   download_{all,params,benchmarks,dataset,homo_data}.py, downloader.py (shared engine)
  prepare/    prepare_system.py + .sh   system pickle from a receptor PDB and a ligand SDF
              homo_search.py + .sh      MSA search (needs the sequence databases)
  infer/      infer.py + .sh            redock one system            (configs/infer.yaml)
              redocking.py + .sh        redock a folder of systems   (configs/redocking.yaml)
              screening.py + .sh        virtual screening            (configs/screening.yaml)
              docking.sh                full pipeline: prepare -> MSA search -> redocking
  train/      train.py + .sh            training                     (configs/train.yaml)
configs/
  infer.yaml  redocking.yaml  screening.yaml  train.yaml
  model/      model sizes toy / tiny / small / medium (released) / full / medium_linear (with diffusion),
              features.yaml (featurisation), loss.yaml, data.yaml (training data)
  trainer/    default.yaml (Trainer, optimiser, EMA, checkpoints, wandb, gradient statistics), ddp.yaml
```

`scripts/prepare/*.py` take ordinary command-line options (`-h`). The demo scripts work from any directory.

System preparation of the demo input system (receptor PDB, ligand SDF and ligand CCD id in, system `.pkl.gz` and
FASTA files out), then the MSA search:

```shell
sh scripts/prepare/prepare_system.sh
sh scripts/prepare/homo_search.sh        # databases in data/homo_data (or HOMO_DATA=<dir>)
```

### Redocking

```shell
sh scripts/infer/infer.sh                # one system  -> out/5SAK_ZRY
sh scripts/infer/redocking.sh            # all demo systems (ranking=true needs scikit-learn)
```

```shell
# One system with explicit inputs
python scripts/infer/infer.py \
  input_pkl=data/demo/redocking/Posebusters_subset/5SAK_ZRY_A_1.pkl.gz \
  msa_features=data/demo/redocking/features/msa_features \
  uniprot_msa_features=data/demo/redocking/features/uniprot_msa_features \
  output_dir=out/5SAK_ZRY

# All options and their defaults
python scripts/infer/redocking.py --cfg job
```

### Virtual Screening

When screening against a pocket, the position of the ligand in the system `.pkl.gz` file (from System Preparation)
defines the pocket.

```shell
sh scripts/infer/screening.sh
```

### Full Docking Pipeline

```shell
sh scripts/infer/docking.sh
```

### Other application scenarios

Other application scenarios, such as blind docking, cross docking or standard precision (SP) flexiable docking can be implemented by specifying different parameters of python scripts. 

## Training

Training uses PyTorch Lightning (`scripts/train/train.py`, `configs/train.yaml`) with an exponential moving average of the
weights (decay 0.999; validation and checkpoints for inference use the EMA weights) and logs to
[wandb](https://wandb.ai) (`wandb login` once, or `logger.offline=true`; `logger=null` disables logging):

* every step: `train/loss`, the loss terms and `train/lr`
* every `trainer.log_every_n_steps`: `train/grad_norm`, `grad_abs_max`, `grad_abs_min`, `param_norm`, the gradient
  norm of every module and, with `grad_stats=parameter` (default), `grad/<parameter>/{norm,abs_max,abs_min}`
* every validation (`trainer.val_check_interval`): the diffusion loss and its terms (`val/loss*`) and, from structures
  sampled by the model, ligand RMSD, success rate (< 2 A), ligand MSE, smooth lDDT, ligand bond / angle error and
  inter-chain clashes, averaged over the validation systems (currently a fixed subset of the training set)

The released training set is downloaded with `python scripts/download/download_dataset.py` (`data/v1`).

```shell
python scripts/train/train.py                                   # released model size, 1 GPU
python scripts/train/train.py trainer=ddp                       # all GPUs of the node
python scripts/train/train.py compile=true                      # torch.compile the training forward pass
python scripts/train/train.py trainer.precision=bf16-mixed      # bf16 encoder (decoder, distogram, loss stay fp32)
python scripts/train/train.py init_from=data/params/params.pt   # fine-tune the released weights
python scripts/train/train.py ckpt_path=logs/train/<run>/checkpoints/last.ckpt    # resume
sh scripts/train/train.sh                                                           # quick test (toy model)
```

Precision: float32 matmuls use TF32 wherever the GPU supports it (`matmul_precision=auto`, the default of all
configs; `highest` = full float32). Under bf16 autocast the decoder, the distogram head, the loss and the coordinate
geometry (rotations, superposition) still run in float32: the generated coordinates are sensitive to rounding there.

Runs are written to `logs/train/<date>/` (checkpoints, wandb files, config). A training checkpoint can be used for
inference directly (`params=logs/train/<run>/checkpoints/last.ckpt`, the EMA weights are used).

## Available Data

The training dataset, the preprocessed benchmark dataset, and the prediction results presented in the paper can be obtained through the following scripts from zenodo.

### Benchmark Data

* PhysDock acheives **SOTA** redocking results in three benchmarks. 

![](./figs/F2.png)

* The preprocessed benchmark dataset can be download through following script.

```shell
# Download all the preprocessed benchmark dataset including Posebusters, DeepDockingDare and PhiBench.
# -> data/benchmarks/{PoseBusters,DeepDockingDare,PhiBench,features}
python scripts/download/download_benchmarks.py
```

* The training and validation dataset can be download by the following scripts

```shell
# -> data/v1/{train_val, train_val_samples_weights.json, msa_features, uniprot_msa_features}  (~145 GB)
python scripts/download/download_dataset.py
python scripts/download/download_dataset.py --only train_val weights   # or a subset
```



## Citation

If you want to cite this work, please use the following `bibtex`

```text
@article{zhang2025physdock,
  title={PhysDock: A Physics-Guided All-Atom Diffusion Model for Protein-Ligand Complex Prediction},
  author={Zhang, Kexin and Ma, Yuanyuan and Yu, Jiale and Luo, Huiting and Lin, Jinyu and Qin, Yifan and Li, Xiangcheng and Jiang, Qian and Bai, Fang and Dou, Jiayi and others},
  journal={bioRxiv},
  pages={2025--04},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

### License

[MIT License](./License)
