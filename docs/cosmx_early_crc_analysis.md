# CosMx早期结直肠癌：手动运行的结果notebook

入口：`CosMx_EarlyCRC_Communication_Niches.ipynb`。按NSCLC结果notebook的形式组织，但该数据为11份患者标本放在2张载玻片上；每份标本独立建物理图，没有虚构11张完整大切片或参考/查询划分。代码位于`cohorts/cosmx_early_crc.py`、`cohorts/early_crc_reporting.py`，R仅做数据转换。

源数据：Nature DOI `10.1038/s41586-026-10344-7`，Zenodo record20136513的`Nanostring_CosMx_exp2.zip`。已核对MD5与内部CRC。原生RNA矩阵为1,257,627细胞×6,175基因，401,694,169非零项。原作者24类注释保留；CRC、oncofetal、normal/undefined epithelium三类不合并，后者不宣称为正常或恶性真值。坐标严格取作者`x_slide_mm/y_slide_mm ×1000`，按`patient_ID`限制邻域；不能沿用NSCLC的像素转换系数或按载玻片建图。

作者QC细胞保留；默认仅排除零RNA/无效坐标。没有从原作者Seurat读取UMAP、PCA、niche/neighborhood或normalized表达。`scripts/export_cosmx_early_crc.R`以稀疏缓冲区转换RNA counts；不需要安装SeuratObject，禁止使用其归一化层代替counts。Python根据精确cell_ID重排元数据，并逐细胞核对nCount_RNA。患者临床信息只附在obs以备下游使用，程序不将其送入图、CCC筛选、编码器或checkpoint选择。

模型固定使用项目主线：原30成员、sigma20µm、有向物理细胞对、细胞–边编码器、两个直接预测MLP、多尺度联合距离loss、固定全局尺度h聚类。默认30轮上限、20轮后允许早停、patience7、训练total选checkpoint，未设置独立验证。K12，Ward合并为K3/K6；K与显示参数不影响模型缓存。不存在预测全矩阵SVD或额外映射，导出紧凑h和组成。

notebook为9个代码单元格：数据/全体细胞图；矩阵；训练并逐轮输出loss；loss图；KMeans/11标本空间图；共同h的UMAP（niche/细胞类型/患者）；粗中细UMAP及两个标本空间图；组成与有方向CCC解释；两个局部上皮–间质界面。局部选区只使用指定患者内的作者注释与几何，不使用阶段或niche挑图；没有预先给区域标上浸润/排斥病理真值。

图形契约：Python/matplotlib；细胞空间板、共同UMAP和多尺度板解释空间组织及颗粒度，loss板只展示实际训练曲线，组成/CCC双热图作训练后机制解释。统一类型与niche颜色，校准µm比例尺，PNG/PDF/SVG均保存NAS；保存图源路径和选择条件。探索性富集不报告细胞伪重复p值，也不宣称已证实同组成异通讯或优于其他方法。

所有输入转换、矩阵、模型、UMAP与输出在NAS：

- `workspace/data/cosmx_early_crc_nature2026/`
- `workspace/outputs/cosmx_early_crc_niches/`

本次只建立notebook并验证读取/实现；不执行真实数据矩阵构建、GNN训练或聚类。首次UMAP涉及全体126万细胞，可能耗时；后续读取缓存。
