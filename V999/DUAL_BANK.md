# Dual-bank residual factorization

本轮新增 phenotype niche bank 与 background niche bank。总容量保持公平：same_composition 使用 Kp=2、Kb=4，总数6；capacity 使用 Kp=2、Kb=2，总数4。WB0 与 WBp 都由观测 bulk 输入推断，均不是模型参数；Cox 风险只读取 `WBp @ gamma`，gamma 长度为 Kp。

训练按固定三阶段协议执行。Stage 1 训练 background bank 800 epochs，不读取 survival。Stage 2 固定 background，以未截断的 signed residual 训练 phenotype bank 800 epochs。Residual-Frozen 在 Stage 2 结束后定稿；Residual-JointFT 再以0.1倍学习率联合训练200 epochs。Dual-JointScratch 从随机初始化联合训练1600 epochs，仅在 same_composition 运行。所有正式模型的 lambda_ph=0.1、lambda_collapse=0，未加入稀疏、正交或多样性约束。

same_composition 和 capacity 均使用原生成器、原 split、seeds 31–35。置换实验固定模型 seed=31、置换 seeds 101–105；五次置换精确复用同一 post-Stage-1 background state。每个 final state 文件包含 state_dict、配置、seed、stage history 与训练诊断。

bank-aware evaluation 对每个真值分别计算 phenotype/background bank 的 HC+HI 最佳匹配、HI-only 匹配、bank margin、分 bank 的 HC/HI/HO、WB/WS 恢复和交叉 bank contamination。background-only Cox probe 只在训练患者上拟合，固定后评估 validation/test，不参与训练或选择。

正式结果见 [outputs/dual_bank/RESULTS.md](/home/xueshuailin/CCC_Phe/V999/outputs/dual_bank/RESULTS.md)。
