import anndata as ad
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import scanpy as sc
import torch
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt
import numpy as np
import gdown
from collections import Counter
import pandas as pd
sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)
def visualization_full(train_latent_path,query_latent_path,train_label_path,query_label_path):
    latent_train = pd.read_csv(train_latent_path,index_col = 0)
    var_info = pd.DataFrame(index = latent_train.columns)
    label_train = pd.read_csv(train_label_path,index_col = 0)
    reference_adata = ad.AnnData(X=latent_train, obs=label_train.loc[latent_train.index,:], var=var_info)
    reference_adata.obs['id'] = "ref"

    latent_query = pd.read_csv(query_latent_path,index_col = 0)
    var_info = pd.DataFrame(index = latent_query.columns)
    label_query = pd.read_csv(query_label_path,index_col = 0)
    query_adata = ad.AnnData(X=latent_query, obs=label_query.loc[latent_query.index,:], var=var_info)
    query_adata.obs['id'] = "query"

    full_latent_adata = reference_adata.concatenate(query_adata)
    sc.pp.neighbors(full_latent_adata,use_rep="X")
    sc.tl.leiden(full_latent_adata)
    sc.tl.umap(full_latent_adata)
    return(full_latent_adata)
