import os
import numpy as np
import torch
import pandas as pd
import scanpy as sc
import json
import umap
import matplotlib.pyplot as plt
from train import ALADTrainer
from preprocess import get_new_adata_dataloader, get_resample_combined_single_cell_dataloader
import random
import torch.nn.functional as F 

def process_and_save_umap_results(adata, adata2,
                                  train_features_path,
                                  save_path,
                                  args_override=None):  
   # seed
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)

    # gene list
    train_features = pd.read_csv(train_features_path,index_col = 0)
    gene_list = train_features.iloc[:,0].tolist()
    adata.var['highly_variable'] = adata.var_names.isin(gene_list)
    Dim = len(gene_list)

    n_cells = adata.n_obs

    class Args:
        num_epochs     = 300
        lr             = 1e-4
        latent_dim     = 50
        batch_size     = 512 if n_cells < 20000 else 4096 
        pretrained     = False
        spec_norm      = True
        dim            = Dim
        num_batches    = adata.obs['sample_ID'].value_counts().shape[0]
        num_celltypes  = adata.obs['cell_type'].value_counts().shape[0]

        # ── Warm-up / Loss 
        lambda_adv    = 1.0
        lambda_cycle  = 0.5
        lambda_batch  = 0.4
        lambda_cell   = 0.4
        warmup_epochs = 10
        ramp_epochs   = 10

    args = Args()


    if args_override:
        for key, value in args_override.items():
            setattr(args, key, value)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    groups = adata.obs.groupby('sample_ID').indices
    alldata = [adata[indices].copy() for indices in groups.values()]

    train_dataloader, label_mapping, highly_variable_genes = get_resample_combined_single_cell_dataloader(
        alldata, args, label_column='cell_type', resampling_strategy='none'
    )

    with open('highly_variable_genes.json', 'w') as file:
        json.dump(highly_variable_genes, file)

    num_datasets = 32
    test_loader, updated_label_mapping = get_new_adata_dataloader(
        adata2, highly_variable_genes, label_mapping, args, num_datasets, label_column='cell_type'
    )

    with open('label_mapping.json', 'w') as file:
        json.dump(updated_label_mapping, file)

    alad = ALADTrainer(args, train_dataloader, device)
    alad.train()


    alad.G.eval()
    alad.Dxx.eval()
    alad.E.eval()
    alad.Dxz.eval()                
    alad.BatchClassifier.eval()   
    alad.cellClassifier.eval()     

    latent_train, latent_query = [], []
    train_latent_x, query_latent_x = [], []
    train_scores, query_scores = [], []          
    train_scores_xz, query_scores_xz = [], []    
    train_scores_cell, query_scores_cell = [], []
    train_indices, query_indices = [], []
    reconstructed_labels, batch_labels = [], []

  
    train_pred_ids, query_pred_ids = [], []

    with torch.no_grad():
        # --------- train ----------
        for x, label, _, idx in train_dataloader:
            x = x.to(device)
            label = label.to(device)
            z = alad.E(x)
            x_reconstructed = alad.G(z)


            _, feature_real_xx = alad.Dxx(x, x)
            _, feature_gen_xx = alad.Dxx(x, x_reconstructed)
            score_xx = torch.sum(torch.abs(feature_real_xx - feature_gen_xx), dim=1)

            _, feature_real_xz = alad.Dxz(x, z)
            _, feature_fake_xz = alad.Dxz(x_reconstructed, z)
            score_xz = torch.sum(torch.abs(feature_real_xz - feature_fake_xz), dim=1)

            # CellClassifier 
            logits_cell = alad.cellClassifier(z)
            probs_cell = F.softmax(logits_cell, dim=1)
            p_max, _ = torch.max(probs_cell, dim=1)
            score_cell = -torch.log(p_max + 1e-12)
            pred_ids = torch.argmax(probs_cell, dim=1)

            latent_train.append(z.cpu().numpy())
            train_latent_x.append(x_reconstructed.cpu().numpy())
            train_scores.append(score_xx.cpu().numpy())
            train_scores_xz.append(score_xz.cpu().numpy())         
            train_scores_cell.append(score_cell.cpu().numpy())     
            train_indices.append(idx)
            reconstructed_labels.append(label.cpu().numpy())
            batch_labels.extend([0] * x.size(0))
   
   
            train_pred_ids.append(pred_ids.cpu().numpy())

        # --------- test ----------
        for x, label, _, idx in test_loader:
            x = x.to(device)
            label = label.to(device)
            z = alad.E(x)
            x_reconstructed = alad.G(z)

            _, feature_real_xx = alad.Dxx(x, x)
            _, feature_gen_xx = alad.Dxx(x, x_reconstructed)
            score_xx = torch.sum(torch.abs(feature_real_xx - feature_gen_xx), dim=1)

            _, feature_real_xz = alad.Dxz(x, z)
            _, feature_fake_xz = alad.Dxz(x_reconstructed, z)
            score_xz = torch.sum(torch.abs(feature_real_xz - feature_fake_xz), dim=1)

  
            logits_cell = alad.cellClassifier(z)
            probs_cell = F.softmax(logits_cell, dim=1)
            p_max, _ = torch.max(probs_cell, dim=1)
            score_cell = -torch.log(p_max + 1e-12)

            pred_ids = torch.argmax(probs_cell, dim=1)

            latent_query.append(z.cpu().numpy())
            query_latent_x.append(x_reconstructed.cpu().numpy())
            query_scores.append(score_xx.cpu().numpy())
            query_scores_xz.append(score_xz.cpu().numpy())        
            query_scores_cell.append(score_cell.cpu().numpy())    
            query_indices.append(idx)
            reconstructed_labels.append(label.cpu().numpy())
            batch_labels.extend([1] * x.size(0))

            query_pred_ids.append(pred_ids.cpu().numpy())


    latent_train = np.concatenate(latent_train, axis=0)
    latent_query = np.concatenate(latent_query, axis=0)
    # train_latent_x = np.concatenate(train_latent_x, axis=0)
    # query_latent_x = np.concatenate(query_latent_x, axis=0)
    train_indices = np.concatenate(train_indices)
    query_indices = np.concatenate(query_indices)
    train_scores = np.concatenate(train_scores, axis=0)
    query_scores = np.concatenate(query_scores, axis=0)
    train_scores_xz = np.concatenate(train_scores_xz, axis=0)         
    query_scores_xz = np.concatenate(query_scores_xz, axis=0)        
    train_scores_cell = np.concatenate(train_scores_cell, axis=0)    
    query_scores_cell = np.concatenate(query_scores_cell, axis=0)    


    train_pred_ids = np.concatenate(train_pred_ids, axis=0)
    query_pred_ids = np.concatenate(query_pred_ids, axis=0)


    save_train_latent="train_latent_x.csv"
    save_query_latent="query_latent_x.csv"
    save_train_latent_z="train_latent_z.csv"
    save_query_latent_z="query_latent_z.csv"
    save_train_score="train_scores.csv"          
    save_query_score="query_scores.csv"         
    save_train_score_xz="train_scores_xz.csv"    
    save_query_score_xz="query_scores_xz.csv"    
    save_train_score_cell="train_scores_cell.csv"
    save_query_score_cell="query_scores_cell.csv"

    pd.DataFrame(latent_train, index=train_indices).to_csv(save_path+"/"+save_train_latent_z, index=True, header=True)
    pd.DataFrame(latent_query, index=query_indices).to_csv(save_path+"/"+save_query_latent_z, index=True, header=True)
    # pd.DataFrame(train_latent_x, index=train_indices).to_csv(save_path+"/"+save_train_latent, index=True, header=True)
    # pd.DataFrame(query_latent_x, index=query_indices).to_csv(save_path+"/"+save_query_latent, index=True, header=True)
    pd.DataFrame(train_scores, index=train_indices, columns=["score"]).to_csv(save_path+"/"+save_train_score, index=True, header=True)
    pd.DataFrame(query_scores, index=query_indices, columns=["score"]).to_csv(save_path+"/"+save_query_score, index=True, header=True)


    pd.DataFrame(train_scores_xz, index=train_indices, columns=["score_xz"]).to_csv(save_path+"/"+save_train_score_xz, index=True, header=True)
    pd.DataFrame(query_scores_xz, index=query_indices, columns=["score_xz"]).to_csv(save_path+"/"+save_query_score_xz, index=True, header=True)
    pd.DataFrame(train_scores_cell, index=train_indices, columns=["score_cell"]).to_csv(save_path+"/"+save_train_score_cell, index=True, header=True)
    pd.DataFrame(query_scores_cell, index=query_indices, columns=["score_cell"]).to_csv(save_path+"/"+save_query_score_cell, index=True, header=True)


    train_id_to_name = {k: v for k, v in label_mapping.items()}
    test_id_to_name = {k: v for k, v in updated_label_mapping.items()}

    train_pred_names = np.array([train_id_to_name.get(int(i), f"UNK_{int(i)}") for i in train_pred_ids])
    query_pred_names = np.array([test_id_to_name.get(int(i), f"UNK_{int(i)}") for i in query_pred_ids])

    save_train_pred_ids = "train_pred_cell_id.csv"
    save_query_pred_ids = "query_pred_cell_id.csv"
    save_train_pred_names = "train_pred_cell_name.csv"
    save_query_pred_names = "query_pred_cell_name.csv"

    pd.DataFrame(train_pred_ids, index=train_indices, columns=["pred_cell_id"]).to_csv(save_path+"/"+save_train_pred_ids, index=True, header=True)
    pd.DataFrame(query_pred_ids, index=query_indices, columns=["pred_cell_id"]).to_csv(save_path+"/"+save_query_pred_ids, index=True, header=True)
    pd.DataFrame(train_pred_names, index=train_indices, columns=["pred_cell_name"]).to_csv(save_path+"/"+save_train_pred_names, index=True, header=True)
    pd.DataFrame(query_pred_names, index=query_indices, columns=["pred_cell_name"]).to_csv(save_path+"/"+save_query_pred_names, index=True, header=True)