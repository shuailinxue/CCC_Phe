"""Reusable tail cells; append without erasing the executed model notebook."""
import nbformat


def cells():
    result=[]
    def md(s):result.append(nbformat.v4.new_markdown_cell(s))
    def code(s):result.append(nbformat.v4.new_code_cell(s))
    md('## 8. 与空间聚类 baseline 比较\n\n使用公开的 [GraphST](https://github.com/JinmiaoChenLab/GraphST) 和 [CellCharter](https://github.com/CSOgroup/cellcharter)。代码和配置位于项目 `baselines/`，结果存入当前 NAS 实验的 `baseline_comparison/`。\n\n**比较协议：**同样的 8,000 个细胞和原始 256 个基因，新增响应/阴性对照基因不进入 baseline；同样的原始 KNN30 邻域拓扑。预处理、表示模型和聚类参数只在原 GNN 训练区域拟合，固定 K=4，不按真实标签选参数。GraphST 使用官方稀疏模型＋作者 `emb` 表示＋KMeans；CellCharter 使用 **PCA32＋一跳空间聚合＋官方 GMM**，明确是 PCA 变体，未声称使用 scVI。两者各自保留原方法的权重/归一化，并记录配置和运行时间。\n\nCCC-AE 使用预计算 CCC；细胞组成方法使用类型标签；我们的 GNN 使用 CCC 与类型分层表达监督。其他表达/空间 baseline 不得到这些监督信息，因此这是一项**同数据、不同信息条件**的比较。现有 GNN checkpoint 不重训、不改参数。')
    code("sys.path.insert(0,str(ROOT))\nfrom baselines.benchmark import run_comparison\nfrom baselines.reporting import BaselineReport\nwith (OUTPUT/'baseline_comparison_execution.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):\n    BASELINE_OUTPUT=run_comparison(OUTPUT)\n    baseline_report=BaselineReport(OUTPUT,BASELINE_OUTPUT)")
    md('### 空间聚类图\n\n第一图保留各方法的原始 cluster ID；第二图仅为看图方便，用训练中心拟合颜色映射。真实标签均在拟合结束后才用于评价，不能把配色匹配当成预测。所有图含训练区域，下方 ARI 是全组织描述性恢复指标。')
    code('show(baseline_report.spatial(align=False))\nshow(baseline_report.spatial(align=True))')
    md('### 同一批空间测试中心上的评价\n\nGraphST 的重构需要两跳图输入，因此所有方法统一使用原空间测试划分再加两跳隔离的中心集合。表中 `n`、`n_true_edge` 给出实际覆盖；未覆盖的区域不宣称已独立验证。ARI/NMI 使用原始聚类编号；F1/IoU 和 A/B accuracy 的映射只在训练中心拟合。固定 K=4 是预先定义的公平比较，不是无监督发现区域数的证据。当前只有一个模拟/训练随机种子，结果用于框架比较。')
    code("show(baseline_report.metric_plot())\ndisplay(baseline_report.metric_table().style.format({k:'{:.3f}' for k in ['overall_ARI','overall_NMI','edge_F1','edge_IoU','edge_AB_ARI','edge_AB_accuracy']}).hide(axis='index'))")
    md('### 全组织描述性指标与计算条件\n\n此表补充包括核心区在内的全组织恢复效果，含训练中心，不能解释成独立泛化评估。空间 baseline 没有 directed CCC decoder，因此不编造其 CCC 重构指标。方法原始输出、checkpoint、输入基因、严格公共测试划分、软件版本和官方源码 commit 均保存在 NAS 中。')
    code("display(baseline_report.metrics.query(\"scope == 'all_centers_descriptive'\")[['method','K','n','overall_ARI','overall_NMI','edge_F1','edge_AB_accuracy']].style.format({k:'{:.3f}' for k in ['overall_ARI','overall_NMI','edge_F1','edge_AB_accuracy']}).hide(axis='index'))")
    for c in result:c.metadata['ccc_baseline_comparison']=True
    return result
