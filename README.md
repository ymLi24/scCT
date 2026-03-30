# scCT
Mapping Single-Cell Data to Human Reference Atlases to Identify Deviated Cell States


A novel reference-based single-cell Cell Type mapping method (scCT), built upon an encoder-decoder framework with cyclic adversarial training. Through training process, a robust, integrated reference atlas is constructed. For new query data, scCT performs direct mapping via a single inference step without requiring fine-tuning. In the resulting latent space, disease data that lacks a corresponding match in normal data is well-represented as isolated clusters. By acting as a diagnostic "CT scan" at the cellular level, this approach ensures accurate cell-type annotation and highlights deviated cell states, revealing alterations in diseased cells 



<img width="450" height="280" alt="figure1" src="https://github.com/user-attachments/assets/094a2599-bf61-409a-b1bf-2c17d59b1110" />

## Features
- Reference-centered cell type mapping (encoder-decoder + cyclic adversarial training)
- One-step inference for new query data (no fine-tuning)
- Latent space separates deviated/disease cell states

## Model

<img width="550" height="350" alt="figure2" src="https://github.com/user-attachments/assets/fa54d4b2-f4f7-43f2-9ee0-56dc1a386d7f" />


## Environment

- Conda
  ```
  conda create -n scct python=3.10 -y
  conda activate scct
  pip install -r requirements.txt
  pip install torch==1.11.0+cu113 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
  ```

If you also want to compute the DAlogFC metric, first install the required environments and dependencies (R, compilation tools, and prerequisite packages):

```
conda install -y r-base=4.3.1
conda install -y compilers
conda install -y -c conda-forge xz zlib
conda install -y bioconda::bioconductor-edger=4.0.16
```

Then install the local package:

```
cd milopy
pip install .
```

## Quick Start

A minimal, end-to-end runnable example from scratch to visualization.

1) Download example data  
You can download the prepared Breast dataset from the provided link.
```
https://cloud.tsinghua.edu.cn/f/9fae471421434b80bb71/?dl=1
```
Unzip it and place the folder in the same directory as the code. Keep the original folder name as Breast2 after extraction.

2) Train and infer (build the reference atlas and map the query data to the reference)
```
python scCT.py --dataset Breast
```

Expected output:  
After running, you should find a result folder under:
./scCT/Breast2/Breast2_delete_basal_cell/resultBreast/result

This folder contains UMAP visualization results.

<img width="800" height="500" alt="屏幕截图 2026-03-29 172323" src="https://github.com/user-attachments/assets/a6089cf7-f456-4b67-8b2d-b7daf121a405" />

  
3) If you also want to compute the DAlogFC metric, make sure you have installed package required.

```
python DAlogFC.py
```

Then you can see the DAlogFC AUPRC and the score of UMAP plot.

<img width="200" height="200" alt="DAlogFC03" src="https://github.com/user-attachments/assets/ad892828-e8c8-4b7a-bbac-a3fef8d7a7c8" />

