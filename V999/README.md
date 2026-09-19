# V999：表型监督空间生态位数学核心

单尺度共享非负分解：bulk 与空间共享 HC、HI；空间独有 HO。WS/H 由 softplus 得到，WB 由统一的可微重建推断得到，gamma 可正可负。训练联合优化五项重建、按事件数平均的 Breslow Cox、可选图平滑及 L2 正则。

## 运行

在 V999 根目录使用具有 PyTorch、NumPy、SciPy、pytest 的 Python：

```bash
python -m pytest -q
python scripts/run_synthetic.py
python scripts/run_synthetic.py --device cuda --output outputs/synthetic_cuda
```

本机已验证的解释器为 `/home/xueshuailin/miniconda3/envs/cccphe/bin/python`。也可以在独立环境中执行 `python -m pip install -e '.[test]'`。脚本从自身位置定位 V999 内部源码，默认输出也固定在 V999/outputs/synthetic，不需要父项目 PYTHONPATH。

## 数据与 API

`CohortData(CB, IB, CS, OS, IS, time, event, laplacian=None)` 接收有限、非负、形状相容的张量。输入前须对齐患者、细胞类型、contact 与 directional CCC 的语义及顺序；维度校验不能替代样本和特征 ID 审计。time 必须为正，event 为 0/1。图使用对称、非正非对角且行和为零的组合 Laplacian，可用 dense 或 sparse 张量。

在数据层显式调用 `BlockScaler.fit(training_data)` 和 `transform`，保存 dataclass 参数。使用每行 RMS Frobenius 尺度；HC 对应的 CB/CS 共用尺度，HI 对应的 IB/IS 共用尺度，避免独立缩放破坏共享字典。未来留出队列只能 transform，不能参与 fit。零数据块尺度为 1，不做中心化以保持非负。

`TrainingConfig` 设置 K、设备、种子、两阶段 epoch、学习率、weight_decay、gradient_clip 和 early_stopping。`LossWeights` 默认使用真正平方 Frobenius 和；示例显式调用 `LossWeights.per_entry` 给每块设置 1/元素数权重。trainer 不进行隐式缩放。lambda_reg 作用于非负实际因素和 gamma；Adam weight_decay 作用于 raw 参数，两者含义不同。

`train(data, config, weights)` 返回 model/history。通过 `bulk_factors()`、`spatial_factors()`、`niche_dictionaries()`、`factors()` 或 `gamma` 读取结果。训练返回的模型绑定训练观测（非参数、非持久 buffer）；`bulk_factors(CB_new, IB_new)` 可推断新患者，手工创建未绑定观测的模型必须显式传入 CB/IB。`forward()` 返回五项重建、risk 和 factors，不修改模型。模型不读取文件。

early_stopping 的整数值表示 patience，依据当前阶段训练目标监测收敛并恢复最佳参数；各阶段独立重置。它不是基于留出患者的泛化选参。日志为更新前的目标值，示例另存最终模型的重新计算损失。

## 数学边界

当前模型没有 raw_WB 或患者自由参数。WB=f(CB,IB,HC,HI)，通过固定步数的可微加速投影梯度计算；Cox 可穿过该计算直接更新 HC/HI。训练、验证、测试使用同一函数与 inner 配置。gamma 从零开始时第一步 Cox 字典梯度为零，gamma 更新后出现；梯度测试同时验证非零 gamma 时的直接梯度及有限差分。数值收敛和诊断协议见 INFERRED_WB.md。

共同 W 无法吸收 HC、HI、HO 三个彼此不同的行尺度。`normalize_factors` 返回新张量，以 HC 行和 s 为共同尺度：所有 H 除以 s，WB/WS 乘以 s，gamma 除以 s。因此 HC 行和为 1，其他两个字典保持相对幅度，全部五项重建及风险严格保持。不引入用户暂不需要的 D 或额外块尺度。若只做展示，可各自将字典行转成比例，但不能再把它们直接用于原模型重建。

当前只有 WS 为队列专属活性参数，WB 每次根据输入推断；简单示例的 C-index 仍只衡量训练内排序。原简单示例不使用留出患者评估；新增的固定字典推断与困难模拟留出流程见 CHALLENGING_BENCHMARK.md，尚未进行临床泛化评估。高训练 C-index 不代表临床有效。无监督 gamma 保持零，C-index=0.5 是恒定风险基线，不是事后拟合 Cox 的性能。模拟数据刻意提供可辨识、可恢复的独立字典，不构成相同组成不同功能、真实数据或所有噪声条件下的证据。加入监督是否提升真值恢复由实际对照决定，不能保证。

## 输出

`summary.json` 保存配置、缩放、版本、Hungarian 配对后每个因素的 cosine/Pearson、匹配后的 gamma、训练内 C-index、最终损失及监督与无监督差值。`*_factors.npz` 保存规范化后的因素；`*_state.pt` 保存原始模型 state_dict；`*_history.json` 保存完整日志；`ground_truth.npz` 保存未缩放模拟真值。

没有实现 BayesPrism、COMMOT、真实 LR 数据库、预处理、多尺度或自动 K 选择。复用来源见 PROVENANCE.md。

## 困难模拟验证

新增 `python scripts/run_challenging_synthetic.py`：两种容量受限场景、严格患者留出、完整 λ 扫描、五模型种子和五次表型置换。协议、无监督 Cox 评估头及指标定义见 [CHALLENGING_BENCHMARK.md](CHALLENGING_BENCHMARK.md)。

## Inferred-WB 与失败诊断

`python scripts/run_inferred_wb.py` 运行本轮完整 80 次实验：capacity 六 λ 五种子、same_composition 完整模型 K=4/6/8 和 CCC-only K=4/6。前一轮结果保持不变，新结果位于 `outputs/inferred_wb`。详见 [INFERRED_WB.md](INFERRED_WB.md)。

## 跨视图 anti-collapse 单一增量

默认 `LossWeights.lambda_collapse=0.0`。新增项为 HC/HO/HI 三视图行余弦的乘积，在不同 niche pairs 上取均值，不改变 inferred-WB。`python scripts/run_anti_collapse.py` 执行固定网格与置换实验，协议见 [ANTI_COLLAPSE.md](ANTI_COLLAPSE.md)。
# Dual-bank residual experiment

本轮结构性 dual-bank 实验见 [DUAL_BANK.md](/home/xueshuailin/CCC_Phe/V999/DUAL_BANK.md)，完整结果见 [outputs/dual_bank/RESULTS.md](/home/xueshuailin/CCC_Phe/V999/outputs/dual_bank/RESULTS.md)。正式实验固定 lambda_collapse=0，不改变既有 single-bank、生成器或拆分。
