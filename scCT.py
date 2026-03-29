import os
today = "/resultBreast/"

import argparse
import anndata as ad
import scanpy as sc
from ALAD import process_and_save_umap_results
from visualization import visualization_full
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_CONFIGS = {
    "Breast": {
        "base_root": "./Breast2",
        "delete_list_csv": "./Breast2/Breast2_delete_celltype_list.csv",
        "prefix": "Breast2",
        "obs_mapping": {
            "train": {
                "sample_ID": "sample_id",
                "level2":    "cell_type",
            },
            "query": {
                "sample_ID": "sample_id",
                "level2":    "cell_type",
            },
        },
        "cell_type_from": None,
        "args_override": {},   
    },
    "Pancreas": {
        "base_root": "./Pancreas2",
        "delete_list_csv": "./Pancreas2/Pancreas2_delete_celltype_list.csv",
        "prefix": "Pancreas2",
        "obs_mapping": {
            "train": {
                "sample_ID": "donor",
                "level2":    "cell_subtype",
            },
            "query": {
                "sample_ID": "donor",
                "level2":    "cell_subtype",
            },
        },
        "cell_type_from": "cell_subtype",
        "args_override": {},  
    },
    "PBMC": {
        "base_root": "./PBMC2",
        "delete_list_csv": "./PBMC2/PBMC2_delete_celltype_list.csv",
        "prefix": "PBMC2",
        "obs_mapping": {
            "train": {
                "sample_ID": "sample_id",
                "level2":    "cell_type",
            },
            "query": {
                "sample_ID": "sample_id",
                "level2":    "cell_type",
            },
        },
        "cell_type_from": None,

        "args_override": {
            "lambda_adv":    1.0,
            "lambda_cycle":  0.5,
            "lambda_batch":  0.25,
            "lambda_cell":   0.25,
            "warmup_epochs": 0,
            "ramp_epochs":   10,
        },
    },
}

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def set_cwd_and_result(base_dir: str):
    ensure_dir(base_dir)
    os.chdir(base_dir)
    ensure_dir("weights")
    ensure_dir("result")

def preprocess_adata(adata: ad.AnnData):
    sc.pp.normalize_total(adata, target_sum=1e4)
    adata.X = np.log1p(adata.X)
    return adata

def build_full_latent(train_latent_path, query_latent_path,
                      train_label_path, query_label_path):
    full_latent_adata = visualization_full(
        train_latent_path, query_latent_path,
        train_label_path, query_label_path
    )
    full_latent_adata.obs.index = (
        full_latent_adata.obs.index.str.replace(r'-\d+$', '', regex=True)
    )
    return full_latent_adata

def save_umap_panels(adata, colors, save_path, fname, ncols=2):
    sc.pl.umap(adata, color=colors, frameon=False,
               wspace=0.6, ncols=ncols, show=False, save=None)
    plt.savefig(os.path.join(save_path, fname), bbox_inches="tight")
    plt.close()

def save_umap_single(adata, color, save_path, fname):
    sc.pl.umap(adata, color=color, frameon=False, show=False)
    plt.savefig(os.path.join(save_path, fname), bbox_inches="tight")
    plt.close()

def apply_obs_mapping(adata: ad.AnnData, mapping: dict, cell_type_from: str = None):

    for target_col, source_col in mapping.items():
        adata.obs[target_col] = adata.obs[source_col]
    if cell_type_from is not None:
        adata.obs['cell_type'] = adata.obs[cell_type_from]
    return adata



def parse_dataset() -> str:

    FALLBACK_DATASET = "Breast"   #  "Breast" | "Pancreas" | "PBMC"

    parser = argparse.ArgumentParser(description="scCT batch pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        default=None,
        help="dataset: Breast | Pancreas | PBMC"
    )
    args, _ = parser.parse_known_args()  

    dataset_name = args.dataset if args.dataset is not None else FALLBACK_DATASET
    assert dataset_name in DATASET_CONFIGS, (
        f"UNKNOW '{dataset_name}', Choose: {list(DATASET_CONFIGS.keys())}"
    )
    print(f"[INFO] dataset: {dataset_name}")
    return dataset_name



# python scCT.py --dataset Breast
# python scCT.py --dataset Pancreas
# python scCT.py --dataset PBMC



dataset_name = parse_dataset()
cfg = DATASET_CONFIGS[dataset_name]

delete_type_list = pd.read_csv(cfg["delete_list_csv"], index_col=0).iloc[:, 0]
newtype = delete_type_list

for itype in newtype:
    base_dir = f"{cfg['base_root']}/{cfg['prefix']}_delete_{itype}{today}"
    set_cwd_and_result(base_dir)
    result_dir = "./result"


    adata  = sc.read("../source_adata.h5ad")
    adata2 = sc.read("../query_adata.h5ad")



    apply_obs_mapping(adata,  cfg["obs_mapping"]["train"], cfg["cell_type_from"])
    apply_obs_mapping(adata2, cfg["obs_mapping"]["query"],  cfg["cell_type_from"])

    train_features_path = "../train_features.csv"
    random.seed(42)

    process_and_save_umap_results(
        adata, adata2,
        train_features_path,
        save_path=os.getcwd(),
        args_override=cfg.get("args_override", {}) 
    )

    train_latent_path = "train_latent_z.csv"
    train_label_path  = "../train_label.csv"
    query_label_path  = "../test_label.csv"
    query_latent_path = "query_latent_z.csv"

    full_latent_adata = build_full_latent(
        train_latent_path, query_latent_path,
        train_label_path,  query_label_path
    )
    full_latent_adata.write("full_latent_adata.h5ad")

    save_umap_panels(
        full_latent_adata,
        colors=["id", "deviated_state_ref", "cell_type"],
        save_path=result_dir,
        fname="umap_full_data_z.pdf",
        ncols=2
    )