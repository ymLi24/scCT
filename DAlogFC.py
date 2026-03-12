import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import warnings

import milopy
import milopy.plot as milopl
import pandas as pd
import scanpy as sc
from anndata import AnnData
pd.DataFrame.iteritems = pd.DataFrame.items


from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    f1_score, 
    precision_recall_curve
)
import numpy as np



def run_milo(
    adata: AnnData,
    ref_query_key: str = "ref_query",
    ref_key: str = "ref",
    query_key: str = "query",
    batch_key: str = "sample_id",
    celltype_key: str = "cell_type",
    design: str = "~is_query",
):
    milopy.core.make_nhoods(adata, prop=0.1)
    milopy.core.count_nhoods(adata, sample_col=batch_key)
    milopy.utils.annotate_nhoods(adata[adata.obs[ref_query_key] == ref_key], celltype_key)
    adata.obs["is_query"] = adata.obs[ref_query_key] == query_key
    milopy.core.DA_nhoods(adata, design=design)


def DALogFC(
    adata: AnnData,
    embedding: str = "X",
    ref_query_key: str = "ref_query",
    ref_key: str = "ref",
    query_key: str = "query",
    batch_key: str = "sample_id",
    celltype_key: str = "cell_type",
    milo_design: str = "~is_query",
    **kwargs,
):
    # Make KNN graph for Milo neigbourhoods
    n_controls = adata[adata.obs[ref_query_key] == ref_key].obs[batch_key].unique().shape[0]
    n_querys = adata[adata.obs[ref_query_key] == query_key].obs[batch_key].unique().shape[0]
    #  Set max to 200 or memory explodes for large datasets
    k = min([(n_controls + n_querys) * 5, 200])
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, use_rep=embedding, n_neighbors=k)
    run_milo(adata, ref_query_key, ref_key, query_key, batch_key, celltype_key, milo_design)

    sample_adata = adata.uns["nhood_adata"].T.copy()
    sample_adata.var["OOR_score"] = sample_adata.var["logFC"].copy()
    sample_adata.var["OOR_signif"] = (
        ((sample_adata.var["SpatialFDR"] < 0.1) & (sample_adata.var["logFC"] > 0)).astype(int).copy()
    )
    sample_adata.varm["groups"] = adata.obsm["nhoods"].T
    adata.uns["sample_adata"] = sample_adata.copy()
    return adata


def compute_identification_metrics(y_true, y_score, threshold=None):

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index] if threshold is None else threshold

    y_pred = (y_score >= best_threshold).astype(int)
    f1 = f1_score(y_true, y_pred)

    # return {
    #     "AUROC": auroc,
    #     "AUPRC": auprc,
    #     "F1@best": f1,
    #     "BestThreshold": best_threshold
    # }

    return auprc

def plot_DAlogFC(adata, output_dir=".", min_logFC=1, alpha=0.001):
    """Plot cell DAlogFC scores with check for pre-computed values"""
    if 'nhood_adata' not in adata.uns:
        print("→ Computing DAlogFC...")
        adata = DALogFC(adata)
    else:
        print("→ Using pre-computed DAlogFC from adata.uns")

    print("→ Building neighborhood graph...")
    milopy.utils.build_nhood_graph(adata)

    print("→ Plotting DAlogFC...")
    plt.rcParams["figure.figsize"] = [10, 10]
    milopl.plot_nhood_graph(
        adata,
        min_logFC=min_logFC,
        alpha=alpha,  # SpatialFDR
        min_size=1,
        show=False
    )
    plt.savefig(os.path.join(output_dir, "DAlogFC_plot.pdf"))
    print("Saved: DAlogFC_plot.pdf")

delete_type_list = pd.read_csv(
    "/PBMC2/PBMC2_delete_celltype_list.csv",
    index_col=0
).iloc[:, 0]
newtype = delete_type_list


results = []


for itype in newtype:
    target_celltype = itype

    
    delete_dir = f"/PBMC2/PBMC2_delete_{itype}"

    if not os.path.isdir(delete_dir):
        print(f"[Skip] delete dir not found: {delete_dir}")
        continue


    method_list = [
        d for d in os.listdir(delete_dir)
        if os.path.isdir(os.path.join(delete_dir, d))
    ]

    if len(method_list) == 0:
        print(f"[Skip] No method folders in: {delete_dir}")
        continue


    for method in method_list:
        method_dir = os.path.join(delete_dir, method)
        adata_path = os.path.join(method_dir, "full_latent_adata.h5ad")

        if not os.path.isfile(adata_path):
            print(f"[Skip] h5ad not found: {adata_path}")
            continue

        print(f"\n=== Processing celltype: {itype} | method: {method} ===")


        data = sc.read_h5ad(adata_path)


        params = data.uns.get("neighbors", {}).get("params", {})
        if "use_rep" not in params:
            sc.pp.neighbors(data, use_rep="X")


        try:
            result = DALogFC(
                data,
                ref_query_key="id",
                ref_key="ref",
                query_key="query",
                batch_key="sample_id",
                celltype_key="cell_type",
                milo_design="~is_query",
            )
        except Exception as e:
            print(f"[Error] DALogFC failed for {itype} | {method}: {e}")

            results.append({
                "celltype": itype,
                "method": method,
                "da_auprc": np.nan,
                "note": f"DALogFC failed: {e}"
            })
            continue

        adata = data
        celltype_key = "cell_type"


        adata.obs["is_abnormal"] = (adata.obs[celltype_key] == target_celltype).astype(int)


        if "sample_adata" in adata.uns:
            sample_adata = adata.uns["sample_adata"]
        elif "sample_adata" in result.uns:
            sample_adata = result.uns["sample_adata"]
        else:
            print(f"[Error] sample_adata not found for {itype} | {method}")
            results.append({
                "celltype": itype,
                "method": method,
                "da_auprc": np.nan,
                "note": "sample_adata missing"
            })
            continue


        groups_mat = sample_adata.varm["groups"].copy()

        n_OOR_cells = groups_mat[:, adata.obs["is_abnormal"] == 1].toarray().sum(1)


        total_cells_per_nhood = np.array(groups_mat.sum(1)).ravel()


        frac_OOR_cells = n_OOR_cells / (total_cells_per_nhood + 1e-10)


        max_frac = frac_OOR_cells.max()
        OOR_thresh = 0.2 * max_frac
        y_true = (frac_OOR_cells > OOR_thresh).astype(int)


        if "nhood_adata" in adata.uns:
            nhood_adata = adata.uns["nhood_adata"]
        elif "nhood_adata" in result.uns:
            nhood_adata = result.uns["nhood_adata"]
        else:
            print(f"[Error] nhood_adata not found for {itype} | {method}")
            results.append({
                "celltype": itype,
                "method": method,
                "da_auprc": np.nan,
                "note": "nhood_adata missing"
            })
            continue

        y_pred = nhood_adata.obs["logFC"].values


        if y_true.sum() == 0:
            print(f"Warning: No positive nhoods found for target cell type '{target_celltype}' in DAlogFC scores.")
            da_auprc = 0.0
            note = "no positive nhoods"
        else:
            try:
                da_auprc = compute_identification_metrics(y_true=y_true, y_score=y_pred)
                note = ""
            except Exception as e:
                print(f"[Error] AUPRC computation failed for {itype} | {method}: {e}")
                da_auprc = np.nan
                note = f"AUPRC failed: {e}"


        print(f"Result -> celltype: {itype}, method: {method}, da_auprc: {da_auprc}")


        results.append({
            "celltype": itype,
            "method": method,
            "da_auprc": da_auprc,
            "note": note
        })

results_df = pd.DataFrame(results)
save_path = "/PBMC2/PBMC2_DAlogFC_summary.csv"
results_df.to_csv(save_path, index=False)

print(f"\nAll done. Summary saved to: {save_path}")
print(results_df.head())