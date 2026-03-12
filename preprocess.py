import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import scanpy as sc

from torch.utils.data import DataLoader
import torch
import random


def worker_init_fn(worker_id):
    seed = 42 + worker_id  
    np.random.seed(seed)
    random.seed(seed)



class SingleCellDatasetWithOneHotAndBarcodes(Dataset):
    """
    PyTorch Dataset for single-cell data, including one-hot encoding and barcodes.
    """
    def __init__(self, exp, labels, onehot, barcodes):
        """
        Args:
            exp (numpy array): Expression matrix (cells x genes).
            labels (numpy array): Cell type labels.
            onehot (Tensor): One-hot encoding for dataset sources.
            barcodes (list): List of barcodes corresponding to each cell.
        """
        self.exp = torch.tensor(exp, dtype=torch.float32)  # Convert expression data to tensor
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to tensor
        self.onehot = onehot  # One-hot encoding tensor
        self.barcodes = barcodes  # Barcodes as a list of strings

    def __getitem__(self, index):
        """
        Return a single sample from the dataset.
        """
        return self.exp[index], self.labels[index], self.onehot[index], self.barcodes[index]

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.barcodes)


def get_new_adata_dataloader(adata2, highly_variable_genes, label_mapping, args, num_datasets, label_column='level1'):
    """
    Create a DataLoader for a new single-cell dataset with one-hot encoding and barcodes.

    Args:
        adata2 (AnnData): New single-cell AnnData object.
        highly_variable_genes (list): List of highly_variable genes based on training data.
        label_mapping (dict): Mapping of training data labels to encoded labels.
        args (Args): Object containing batch_size and other parameters.
        num_datasets (int): Number of datasets for one-hot encoding dimension.
        label_column (str): Column in `adata2.obs` containing cell type labels.

    Returns:
        test_loader (DataLoader): DataLoader for the new dataset.
        updated_label_mapping (dict): Updated label mapping including new labels in adata2.
    """    
    # if adata2.X.max() > 1000:  # Check if data is already log-transformed
    #     sc.pp.normalize_total(adata2, target_sum=1e4)
    #     adata2.X = np.log1p(adata2.X)

    # Step 1: Align genes to `highly_variable_genes`
    adata2_genes = adata2.var.index.to_list()
    gene_to_index = {gene: idx for idx, gene in enumerate(adata2_genes)}

    # Create a boolean mask for `highly_variable_genes` to quickly find the matching genes
    gene_mask = np.array([gene in gene_to_index for gene in highly_variable_genes])

    # Create the expression matrix for all highly_variable_genes
    exp = np.zeros((adata2.shape[0], len(highly_variable_genes)), dtype=np.float32)
    matched_genes = [gene_to_index[gene] for gene, found in zip(highly_variable_genes, gene_mask) if found]




    # Populate the expression matrix for matched genes
    if len(matched_genes) > 0:
        # Extract the expression data for matched genes
        matched_exp = adata2[:, matched_genes].X

        # Check if the matrix is sparse and convert it to dense if necessary
        if hasattr(matched_exp, "todense"):  # If it's a sparse matrix
            matched_exp = matched_exp.todense()

        # Convert to a NumPy array to ensure compatibility
        matched_exp = np.array(matched_exp, dtype=np.float32)

        # Assign the data to the corresponding positions in the expression matrix
        exp[:, gene_mask] = matched_exp

    # Step 2: Encode labels and update label mapping
    adata2_labels = adata2.obs[label_column].values  # Extract labels
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}  # Reverse mapping
    unique_labels = np.unique(adata2_labels)

    new_label_mapping = label_mapping.copy()
    next_label_index = max(label_mapping.keys()) + 1
    for label in unique_labels:
        if label not in reverse_label_mapping:
            new_label_mapping[next_label_index] = label
            reverse_label_mapping[label] = next_label_index
            next_label_index += 1

    encoded_labels = np.vectorize(reverse_label_mapping.get)(adata2_labels)

    # Step 3: Extract barcodes
    barcodes = adata2.obs_names.tolist()

    # Step 4: Create one-hot encoding for the dataset
    dataset_onehot = torch.zeros((adata2.shape[0], num_datasets), dtype=torch.float32)

    # Step 5: Create PyTorch dataset with one-hot encoding and barcodes
    test_dataset = SingleCellDatasetWithOneHotAndBarcodes(exp, encoded_labels, dataset_onehot, barcodes)

    # Step 6: Create DataLoader
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  
        num_workers=0,
        worker_init_fn=worker_init_fn
    )
    return test_loader, new_label_mapping


def get_resample_combined_single_cell_dataloader(
    data, args, label_column='level1', random_state=42, resampling_strategy='oversample'
):
    """
    Efficiently create a dataloader for combined single-cell datasets with resampling for class imbalance.
    Includes one-hot encoding and barcodes.

    Args:
        data (list of AnnData): List of single-cell AnnData objects.
        args (Args): Arguments containing batch_size and other parameters.
        label_column (str): Column name in `adata.obs` containing cell type labels.
        random_state (int): Random seed for reproducibility.
        resampling_strategy (str): Resampling strategy ('oversample', 'undersample', or 'none').

    Returns:
        train_loader (DataLoader): DataLoader for the combined dataset after resampling.
        label_mapping (dict): Mapping of encoded labels to original labels across all datasets.
        final_genes (list): List of genes used for processing (intersection of highly_variable_genes and shared genes).
    """
    # Step 1: Find shared genes across all datasets
    all_highly_variable_genes = set.union(*[set(adata.var.index[adata.var['highly_variable']]) for adata in data])
    shared_genes = set.intersection(*[set(adata.var.index.tolist()) for adata in data])
    # final_genes = list(all_highly_variable_genes & shared_genes)
    final_genes = sorted(list(all_highly_variable_genes & shared_genes)) # Sort for consistency

    # Step 2: Prepare combined data
    label_encoder = LabelEncoder()
    combined_exp, combined_labels, dataset_indices, combined_barcodes = [], [], [], []

    for i, adata in enumerate(data):
        # Preprocessing steps
        # if adata.X.max() > 1000:  # Check if data is already log-transformed
        #     sc.pp.normalize_total(adata, target_sum=1e4)
        #     adata.X = np.log1p(adata.X)

        # Extract expression data, labels, barcodes, and dataset indices
        exp = adata[:, final_genes].X
        if hasattr(exp, "todense"):
            exp = exp.todense()
        exp = np.array(exp, dtype=np.float32)

        combined_exp.append(exp)
        combined_labels.extend(adata.obs[label_column])
        combined_barcodes.extend(adata.obs_names.tolist())
        dataset_indices.extend([i] * exp.shape[0])  # Dataset index for one-hot encoding

    # Encode labels
    combined_labels = np.array(combined_labels)
    encoded_labels = label_encoder.fit_transform(combined_labels)
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}

    # Create one-hot encoding for datasets
    dataset_onehot = np.eye(len(data), dtype=np.float32)[dataset_indices]

    # Combine all data into arrays
    combined_exp = np.vstack(combined_exp)
    combined_labels = np.array(encoded_labels, dtype=np.int64)
    dataset_onehot = torch.tensor(dataset_onehot, dtype=torch.float32)

    # Step 3: Resample data if needed
    if resampling_strategy != 'none':
        combined_exp, combined_labels, dataset_onehot, combined_barcodes = resample_data(
            combined_exp, combined_labels, dataset_onehot, combined_barcodes, resampling_strategy, random_state
        )

    # Step 4: Create PyTorch dataset
    train_dataset = SingleCellDatasetWithOneHotAndBarcodes(combined_exp, combined_labels, dataset_onehot, combined_barcodes)

    # Step 5: Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    worker_init_fn=worker_init_fn)

    return train_loader, label_mapping, final_genes


def resample_data(exp, labels, onehot, barcodes, strategy='oversample', random_state=42):
    """
    Resample the data to handle class imbalance.

    Args:
        exp (numpy array): Expression matrix.
        labels (numpy array): Label array.
        onehot (Tensor): One-hot encoding for dataset sources.
        barcodes (list): List of barcodes corresponding to each cell.
        strategy (str): Resampling strategy ('oversample', 'undersample').
        random_state (int): Random seed.

    Returns:
        resampled_exp (numpy array): Resampled expression matrix.
        resampled_labels (numpy array): Resampled label array.
        resampled_onehot (Tensor): Resampled one-hot encoding.
        resampled_barcodes (list): Resampled list of barcodes.
    """
    # Combine all data into a single array for easier manipulation
    combined_data = list(zip(exp, labels, onehot.numpy(), barcodes))

    # Split data by class
    class_indices = {label: [] for label in np.unique(labels)}
    for i, label in enumerate(labels):
        class_indices[label].append(combined_data[i])

    resampled_data = []

    # Resample each class
    max_samples = max(len(class_indices[label]) for label in class_indices)
    min_samples = min(len(class_indices[label]) for label in class_indices)

    for label, samples in class_indices.items():
        if strategy == 'oversample':
            resampled_data.extend(resample(
                samples, replace=True, n_samples=max_samples, random_state=random_state
            ))
        elif strategy == 'undersample':
            resampled_data.extend(resample(
                samples, replace=False, n_samples=min_samples, random_state=random_state
            ))
        else:
            raise ValueError("Invalid resampling strategy. Choose 'oversample' or 'undersample'.")

    # Shuffle the resampled data
    np.random.seed(random_state)
    np.random.shuffle(resampled_data)

    # Split back into exp, labels, onehot, and barcodes
    resampled_exp, resampled_labels, resampled_onehot, resampled_barcodes = zip(*resampled_data)
    resampled_exp = np.array(resampled_exp)
    resampled_labels = np.array(resampled_labels)
    resampled_onehot = torch.tensor(resampled_onehot, dtype=torch.float32)
    resampled_barcodes = list(resampled_barcodes)

    return resampled_exp, resampled_labels, resampled_onehot, resampled_barcodes