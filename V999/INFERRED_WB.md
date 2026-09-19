# Inferred-WB 与 same_composition 失败诊断

## 实现边界

模型参数仅保留 raw_WS、raw_HC、raw_HO、raw_HI 和 gamma。没有 raw_WB，没有任何按患者索引的可训练 bulk 参数。训练观测可作为非持久 buffer 绑定在模型上，state_dict 不包含患者观测；WB 每次 forward 根据当前输入及字典重新计算，无缓存和 forward 状态修改。新患者调用 `model.bulk_factors(CB, IB)`，全部模型因子调用 `model.factors(CB, IB)`。

独立推断接口为 `infer_bulk_activities(CB, IB, HC, HI, lambda_bc, lambda_bi, steps, learning_rate, create_graph)`。它不接收 time、event 或 gamma。create_graph=True 保留与字典相连的普通 autograd 图，False 关闭梯度跟踪但不改变数值算法；外部 no_grad 同样可用于评估。

使用固定步数的加速投影梯度（FISTA 型）：G=λ_bc HC HCᵀ+λ_bi HI HIᵀ，B=λ_bc CB HCᵀ+λ_bi IB HIᵀ；从零活性开始，按 G 的最大绝对行和作为 Lipschitz 上界更新并投影到非负正交域。有限步迭代是 NNLS 的近似解，不能宣称对任意病态字典都精确达到 argmin。没有 inner detach，没有 SciPy 训练路径，没有新的损失或正则。

训练和验证/测试使用相同实现、重建权重、步数和相对步长。该版本将推断纳入计算图，因此 Cox 可直接影响 HC/HI。gamma 初始化为零时第一步对字典的 Cox 梯度为零，这是风险头定义的结果；测试使用非零 gamma 检查通路，并另外检查实际联合训练。

既有 L2 保留，包括推断 WB 的平方和；在 CCC-only 模式中，未使用的 HC/HO 不参与 L2，也不参与重建或推断。WS、HI 与 gamma 的既有正则保持不变，未添加任何新的正则项。

## 数值检查

在训练患者及独立空间观测上构造初始化字典和由观测行组成的字典，覆盖 K=4/6/8、完整块及 CCC-only。扫描 steps={10,25,50,100} 与相对 lr={0.5,1.0}，只比较重建误差、有限性及与 NNLS 参考解的差距，不使用临床标签。

预设目标为最大 `(final-NNLS)/initial <= 1e-4`，优先选择达到目标的最少步数；若全部不满足，选择数值误差最小者并记录 tolerance_met=false。当前选择 100 步、lr=1.0；初始化压力测试最大差距约为初始重建的 0.156%，未完全满足该精度目标。所有正式训练的最终字典另行审计，将误差保存在每次实验的 inner_audit 中，不能以初始化扫描代替最终收敛检查。

`nnls_reference.py` 保留旧的 CPU NNLS，仅用于数值参考，不用于训练、验证或测试的预测机制。归一化输出仍能保持已计算的 WH 与风险不变，但有限步求解不是任意字典坐标缩放下严格不变的算法，因此新患者预测必须使用训练时的原始字典与原始 inference 配置，不能随意用展示用归一化字典替换。

## 固定实验

运行 `python scripts/run_inferred_wb.py`，默认 CPU、4 个独立进程，每个进程 PyTorch 单线程。可用 `--device cuda --workers 1 --output <新目录>` 切换设备。没有分布式训练库。

1. capacity：原数据 K_true=6、K_fit=4，原六个 λ、原五种子 31–35，各 800+800 epoch。仅用 validation C-index 均值选择正 λ，选择记录写入后评分测试患者。旧版对比只读取既存 JSON，不重跑旧模型。
2. same_composition 完整模型：K_fit=4/6/8，分别 λ=0 与 λ=0.1，每组五种子。
3. same_composition CCC-only：K_fit=4/6，分别 λ=0 与 λ=0.1，每组五种子；关闭 CB/CS/OS 的目标权重及 HC/HO 的 L2。其余训练参数一致。

诊断 λ=0.1 直接固定为前一轮 same_composition 的验证集选择，不在新 K 或新模式下调参。完整模型 K 扫描和 CCC-only 均是预设诊断，不按照测试结果选模型。训练共 30+30+20=80 次。

生成器、噪声、effect size、nuisance、survival generation 与 train/validation/test 拆分全部保持原样。启动时逐张量对比旧 ground_truth.npz，逐项对比旧 splits.json，失败即停止。模型 K 与 generator 的 K_true 分开管理，因此 K_fit=8 不需要修改 synthetic truth。

无监督基线继续使用仅训练患者拟合的冻结字典 Cox 评估头，正则固定为原值 0.01；监督模型使用联合训练 gamma。该差别明确保留，不能把原分解 gamma=0 的恒定风险作为预测基线。新版不再生成 free-WB C-index。

## Pair 指标与诊断边界

完整模型以 HC+HI 拼接匹配 A/B；CCC-only 只以 HI 匹配，避免无训练的 HC 污染指标。两类模式另外都报告 HI-only collision，便于使用共同标准对照。逐次保存 A/B 的最佳 factor、collision、HC/HI cosine、WS correlation、gamma 以及两个匹配 learned HI 行之间的 cosine。

CCC-only 不报告未训练 HC 的恢复性能，不将两种不同匹配定义下的分数直接解释成提升幅度。collision 是最佳匹配相同，不等同于已证明模型的数学不可辨识性；即使最佳匹配不同，也需结合 HI/WS 恢复强度判断是否真正分离。

真值诊断计算 WB_A/WB_B、WS_A/WS_B、WB_B/eta、WB_A/eta 及 B 与 `-log(observed time)` 的相关。后者受删失影响，仅为描述性 proxy，不能当作生存效应估计。

不能只凭有限次优化失败证明结构不可辨识；若数值误差、种子敏感性或恢复强度仍有问题，结论必须保留这些限制。

跨版本使用相同种子编号，但移除 raw_WB 改变了初始化随机数消耗顺序，因此旧／新对比不是逐参数完全相同初始化的因果实验；同一新版内各 λ 与各诊断对照保持预设种子和一致初始化流程。
