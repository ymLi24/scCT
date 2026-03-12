对于这个代码。Pancreas的测试数据又会有不同，如：# 删除类型列表
delete_type_list = pd.read_csv(
    "/home/wyh/liver_atlas/Figure2_mapping_method_benchmark/Pancreas2/Pancreas2_delete_celltype_list.csv",
    index_col=0
).iloc[:, 0]
newtype = delete_type_list  # 与原代码保持一致

# 第一步：为每个删除类型跑一次预处理 + UMAP 融合，并保存 full_latent_adata 和 UMAP 面板


# 第一步：为每个删除类型跑一次预处理 + UMAP 融合，并保存 full_latent_adata 和 UMAP 面板
for itype in newtype:
    base_dir = f"/home/wyh/liver_atlas/Figure2_mapping_method_benchmark/Pancreas2/Pancreas2_delete_{itype}{today}"
    set_cwd_and_result(base_dir)
    result_dir = os.path.join(base_dir, "result")

    # 读取与预处理
    adata = sc.read("../source_adata.h5ad")
    adata2 = sc.read("../query_adata.h5ad")


    adata = sc.read("../source_adata.h5ad")
    adata.obs['sample_ID'] = adata.obs['donor']
    # adata.obs['sample_ID'] = adata.obs['donor_id']
    adata.obs['level2'] = adata.obs['cell_subtype']
    adata.obs['cell_type'] = adata.obs['cell_subtype']
    adata2 = sc.read("../query_adata.h5ad")
    adata2.obs['sample_ID'] = adata2.obs['donor']
    # adata2.obs['sample_ID'] = adata.obs['donor_id']
    adata2.obs['level2'] = adata2.obs['cell_subtype']
    adata2.obs['cell_type'] = adata2.obs['cell_subtype']

而PBMC又是# 删除类型列表
delete_type_list = pd.read_csv(
    "/home/wyh/liver_atlas/Figure2_mapping_method_benchmark/PBMC2/PBMC2_delete_celltype_list.csv",
    index_col=0
).iloc[:, 0]
newtype = delete_type_list  # 与原代码保持一致

# 第一步：为每个删除类型跑一次预处理 + UMAP 融合，并保存 full_latent_adata 和 UMAP 面板


# 第一步：为每个删除类型跑一次预处理 + UMAP 融合，并保存 full_latent_adata 和 UMAP 面板
for itype in newtype:
    base_dir = f"/home/wyh/liver_atlas/Figure2_mapping_method_benchmark/PBMC2/PBMC2_delete_{itype}{today}"
    set_cwd_and_result(base_dir)
    result_dir = os.path.join(base_dir, "result")

    # 读取与预处理
    adata = sc.read("../source_adata.h5ad")
    adata2 = sc.read("../query_adata.h5ad")


    adata = sc.read("../source_adata.h5ad")
    adata.obs['sample_ID'] = adata.obs['sample_id']
    # adata.obs['sample_ID'] = adata.obs['donor_id']
    adata.obs['level2'] = adata.obs['cell_type']
    adata2 = sc.read("../query_adata.h5ad")
    adata2.obs['sample_ID'] = adata2.obs['sample_id']
    # adata2.obs['sample_ID'] = adata.obs['donor_id']
    adata2.obs['level2'] = adata2.obs['cell_type']

请你在代码中定义一个dict，知道应该怎么读取