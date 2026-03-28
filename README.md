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

