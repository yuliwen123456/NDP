# NDP 基线冻结、差异诊断与候选模块方案

日期：2026-09-14。执行范围：只读分析历史实验，新增快照、保护工具和方案文件。未启动 V08/V09/V10，未恢复周期监控。所有指标为百分数，差值为百分点。

## NDP CURRENT PIPELINE

```text
LiDAR 输入 + 训练时 Perlin 合成 OOD
  ↓ 0.05m 稀疏体素化
Sparse UNet backbone
  ├─ 多尺度特征 → Mask4Former decoder → panoptic mask/class losses
  └─ point_features_head → 128维点特征
       ↓ Projector MLP 128→96→96→38
       38 logits = 19 ID channels + 19 auxiliary channels
       ├─ 前19 logits → ID semantic CE
       ├─ EE = logsumexp(全部38) − logsumexp(前19)
       └─ Wp投影16维 → 与可学习prior(38×16)交叉注意力 → w≥1
             ↓
       score = EE × w + training_bias
             ↓ inverse_maps 恢复原点顺序
       保存逐点原始浮点score → 官方point-level评估
```

入口 `main_panoptic.py` 使用 `trainer/pq_trainer.py`，不是同目录另一个旧 trainer 实现。源码证据均保存在 `experiments/v06_baseline/source_v06/`：`models/mask4former.py:13` 是 NDP_block，`:32` 是 Projector，`:245` 起是 EE 打分；`trainer/pq_trainer.py:94` 起是训练损失，`:145` 起是测试导出。

**NDP 的全称是 Neural Distribution Prior，并非空间邻域模块。** 当前 NDP_block 不接收点坐标或 kNN。其 prior 是随机初始化后通过反向传播更新的可学习矩阵，不是按类别数据均值构建的 centroid，也不是 EMA prototype。注意力 A=softmax(Q(e)K(prior)^T/√16)，z=A·V(prior)，w=ReLU(Linear([e,z]))+1。已有先验不能简单解释为经过监督对齐的38个语义原型。

|待查功能|源码结论|
|---|---|
|confidence/dynamic weighting|已有：逐点 logits 与 prior 注意力决定 w|
|global prior|已有共享可学习 prior；没有第二套经验类别中心|
|local spatial OOD distribution|NDP_block 没有；backbone 的空间卷积和 decoder 多尺度特征不能当作显式 OOD 一致性模块|
|class-aware threshold|当前推理和官方评估没有逐类阈值|
|entropy/energy/EE|论文提出三种静态分数；本次实际代码硬编码 EE，没有可直接切换的 static_ood_function 配置项|
|score normalization / fusion|V06/V07 均无新归一化、无模型分数融合；保存原始 EE×w+b|
|boundary suppression / EMA prototype / Mahalanobis|当前执行路径没有|
|uncertainty calibration|已有 NDP 权重及可学习 bias；不能把通用 adaptive calibration 当全新机制|

训练同时包含 ID CE、合成 OOD 的 BCEWithLogits（pos_weight=10000）、SOE 与 decoder 的 CE/mask/dice/box 及辅助层损失。SOE 对 ID 约束接近0，对 void 使用0.9软目标。不能只观察最终 OOD loss，而忽略共享 backbone 上损失比例变化。

论文也将 NDP 定义为基于可学习分布先验的自适应重加权；这是判定通用 CGAC 重复的重要依据。[NDP 原文](https://arxiv.org/html/2604.09232)

## V06 vs V07 DIFFERENCE

|项目|V06|V07|
|---|---|---|
|AP↑|74.5532068|60.5396530（−14.0135538）|
|FPR95↓|1.4472668|0.7721332（−0.6751336）|
|AUROC↑|99.3599572|99.5367129（+0.1767557）|
|GPU / precision|单RTX3090 / FP32|相同|
|physical / accumulation / effective batch|2 / 4 / 名义8，尾组实际2|相同|
|epoch / optimizer updates|10 / 1130|相同|
|LR / scheduler / seed|AdamW 2e−4 / OneCycle1130 / 2025|相同|
|初始化|相同 STU checkpoint，非上轮续训|相同|
|BN|m=1−(1−m原值)^(1/r)，尾组r=1|相同|
|sum类 criterion损失|mask/dice/box含aux乘实际组长r|不乘r|
|整个microbatch loss|再乘4/r后交给Lightning|乘1交给Lightning|
|Lightning自动累积|除以4|除以4（尾组也除4）|
|打分/归一化/融合/prior更新方法|原 EE×NDPweight+b，无新增处理|相同方法，学习后的权重不同|
|评估代码/8659预测文件覆盖|原STU评估、覆盖完整|相同|

真正差异在 V06 `runtime_v06/loss_correction.py:4`、`:12` 和 `runtime_v06/train.py:61`；V07独立脚本 `outputs/seventh-bn-only-20260914/loss_policy.py:4`、`:12` 返回原loss及scale=1。`models/criterion.py:76` 起的 mask/dice/box 是按样本求和，CE为mean，因此补偿影响不同损失的相对梯度。

完整组 r=4 时，V06使sum类梯度相对V07放大4倍；mean类全组处理相同。尾组 r=1 时，V06把整组除4造成的缩小补回，V07保留1/4。**因此V07同时改变损失相对权重和尾组强度，不能将AP下降单独归因于其中一个因素。** 两者都会影响共享特征和后续排序。第四至七次2×2对照的AP交互差为13.6930个百分点，支持“BN策略效果依赖损失处理”，但单seed不足以证明统计显著性。

Checkpoint 中 training_bias：V06 −2.72645736，V07 −2.72831202，仅差−0.00185466。对全体点添加同一个常数不会改变AP/AUROC/FPR95排序，所以这不是14点AP损失的充分解释。原先“过强校准”的描述只能作为待验证假说；没有发现V07新增了校准模块。

与论文表格NDP-EE（AP74.24/FPR1.43/AUC99.53）比，V06分别为+0.3132/+0.0173/−0.1700，V07为−13.7003/−0.6579/+0.0067。不能拼接不同checkpoint的最好指标声称一个模型同时达到目标。当前只有public validation的point-level结果，未获得hidden test或object-level成绩。

### 真实分数诊断

独立脚本 `analysis/analyze_score_distribution_v01.py` 只读已保存预测与标签，不改官方评估。固定等间隔选择每序列20帧，共380帧，经过官方范围2.5–50m、ignore0、每帧至少5个OOD点过滤后保留88帧，共8,681,992点（ID8,677,278，OOD4,714）。先定帧再读标签/分数，不用结果挑帧。初版每序列5帧只得到19个有效帧，扩展版用于增加覆盖，两个结果都保留。

下面是扩展样本的原始分数统计，不是概率：

|模型/点类型|n|mean|std|median|5%|25%|75%|95%|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|v06/ID|8677278|-2.6159|0.5542|-2.7193|-2.7261|-2.7248|-2.6808|-2.3072|
|v06/OOD|4714|21.2101|12.1249|21.2775|1.3292|11.5250|29.4025|41.0694|
|v07/ID|8677278|-2.6509|0.3951|-2.7230|-2.7281|-2.7271|-2.6989|-2.4233|
|v07/OOD|4714|9.6000|5.9241|9.2930|-0.3110|5.7330|13.3599|19.8002|

排序证据：在同一扩展样本上的precision-recall工作点：

|召回率|V06 precision %|V07 precision %|
|---|---:|---:|
|0.1|94.97|76.01|
|0.25|92.04|72.46|
|0.5|85.15|65.46|
|0.75|67.68|55.95|
|0.9|50.65|29.43|
|0.95|12.26|10.64|

V07在这些召回区间的precision均低于V06，说明高分列表中ID/OOD的相对排序变差。OOD中位数下降比ID中位数显著，但仅凭分布整体缩放不能解释排序指标，必须结合PR曲线看。

抽样AP：V06 76.5567 / V07 59.6668；抽样FPR95：0.3693 / 0.4335；抽样AUROC：99.5337 / 99.6351。**抽样FPR变化方向与全量正式结果相反**，初版小样本同样如此；说明场景差异和抽样局限，不能用这些数字替换全量成绩，也不能声称已经定位全量FPR改善来源。

按各模型已有全量95%召回阈值统计样本误报（只用于诊断，不重新选阈值）：V06/V07中raw label40点贡献88.86%/91.28%的FP，但它们同时占98.44%的ID点；“贡献多”主要受样本量影响，不能直接证明road类别特别容易误报。raw10点仅11,937个，却分别有72.38%/16.83%的误报率，显示不同标签组分布确有差异。极少样本类别（如pole仅18点）不能据此制定校准。部署时需要预测类别，现存文件只有标量score，不能把GT类别当作预测类别输入。

V07约89.52%的FP在2.5–10m，77.03%位于每1m体素≥100点的组。现有样本**不支持“主要是远距离稀疏点”**这一预设。点密度不是语义边界，未保存confidence、semantic prediction、38logits、NDPweight，不能做可信的confidence/边界分箱或分解EE与w各自作用。进一步诊断需要新的独立导出入口并经批准推理，不能虚构这些字段。

## CANDIDATE MODULE 1

名称：V08 DSDC，修订为 Spatial-context / learned-prior consistency（内部实验名，非原创声明）。

解决的问题：原NDP基于每点logit与共享prior，不直接检查同一帧的几何上下文。研究孤立的高异常分数是否与局部上下文不一致；现有统计未证明这种不一致是主要错误来源，因此先做可证伪消融。

输入：冻结V06的体素坐标、38维logits、已有prior attention A及原score。输出：每点有界残差与新score；保留原score供对照。

位置：原EE×w+b之后、inverse_map之前。新的导出/推理entrypoint读取冻结模型；不编辑历史mask4former.py或trainer。

设计：使用两个预先登记的空间尺度对A做邻域均值q_small/q_large，再计算JS(A,q)或归一化cosine距离。复用已有prior及Wp/Q/K，不建立第二套global prototype。不把A误称为真实语义概率。残差以仅ID训练参考集估计的中心和尺度标准化并裁剪，s'=s+λ·clip(z,−δ,δ)。λ=0必须逐点还原V06；稀疏无邻居/无效输入回退原score。尺度、λ、δ待使用ID协议预登记，禁止用STU OOD标签挑最优值。

是否训练：首版冻结V06，无反向传播；新增可训练参数0，存在统计缓冲与超参数。计算预计使用hash-grid pooling避免N²邻接，仍须实际测时/显存，不能现在承诺零成本。

AP预期：希望减少孤立高分ID并保留OOD排序；可能误伤小物体或边界，目标不是已验证收益。FPR预期：有望改进ID尾部，但也可能增加边界误报。判断需完整同口径评估，不能只选一个指标。

与原NDP区别：新增显式坐标邻域一致性，原模块是逐点与可学习prior的注意力。邻域辅助OOD评分在相关研究中已有先例，因此这里只提出针对当前实现的可检验扩展，不作原创性结论。

计划新文件：`experiments/v08_dual_scale/modules/dual_scale_distribution_v08.py`、`infer_v08_dual_scale.py`、新的配置修订文件；本轮仅创建方案配置与命令占位，不实现算法。

## CANDIDATE MODULE 2

通用CGAC不推荐：原NDP已根据logits和prior动态产生逐点w，重复叠加confidence gate区分度不足。替代推荐：V09 Class-Conditional ID Tail Calibration（暂定，NEEDS_MORE_TESTING）。

解决的问题：ID原始分数在标签组间有差异，但需先通过真实预测类别验证；不是看到road误报占比高就直接压低road得分。

输入：冻结V06的score、预测ID类别；来自独立training-ID参考集的各类及全局分数CDF。输出：有界、向原score收缩的逐类校准score。

位置：原score之后、导出之前，独立新entrypoint。理论形式：s'=s+λ·clip(F_global^−1(F_c(s))−s,−δ,δ)。类样本少则Fc向全局CDF收缩；极端尾部设数值保护并回退，不外推不存在的ID分位数。λ=0精确还原V06。

是否训练：不训练V06，不用OOD标签学习gate。可训练参数0；缓冲约(19+1)×Q个分位点及计数，Q待预登记。拟合参考集必须只有ID并按场景分开，排除合成OOD和void；记录模型hash、数据清单、类别覆盖。预测类别错分会带来偏差，需做稳定性检查。

AP预期：可能减轻部分高尾ID对排序的影响，但跨类变换也可能压低被错分为ID的OOD，AP可能下降。FPR预期：缓解类别间ID尾分布不齐，不保证达标。不能把ID分位数解释成OOD概率或无条件conformal保证。

与原NDP区别：冻结后的经验ID尾分布校准，原NDP是端到端学习的logit-prior attention；目标上都处理类别偏差，因此仍有重叠，属于实用对照方向，不能仅靠换名宣称创新。类条件/统计校准相关工作也需在实现前继续核对。

计划新文件：`experiments/v09_class_conditional/modules/class_conditional_v09.py`、`fit_id_statistics_v09.py`、`infer_v09_class_conditional.py`。本轮仅创建方案；未计算校准参数。

四个方向结论：修订DSDC → 推荐验证；通用CGAC → DROP（与原NDP重复）；Boundary suppression → 暂缓（缺边界证据，远距离稀疏假说未获支持）；Class-conditional → 有条件推荐（先验证预测类别与ID参考集，避免GT泄漏）。

## VERSION PLAN

V06：KEEP，主Baseline，已独立快照，历史权重保留原地。

V08：V06 + DSDC方案；feature/v08-dual-scale；未实现/未训练。

V09：V06 + class-conditional方案；feature/v09-class-conditional；替代重复的CGAC；未实现/未训练。

V10：V06 + 两个独立有效模块，暂存conditional plan。只有V08/V09各自通过完整评估后才组合；失败版本保留，不自动纳入。所有proposal指标null，不能冒充实验结果。

训练设置如将来需要训练，固定继承V06：单卡、physical2、accumulation4、FP32、LR2e−4、OneCycle1130、seed2025、10epoch、同BN与loss补偿。首版两个模块均训练free，因此训练GPU数0，推理GPU建议1，实际耗时待验证。正式比较需要使用同一checkpoint/分帧/过滤/官方评估，报告AP/FPR/AUROC全项。阶段目标AP≥74.5、FPR<1.43（理想<1.0、AUROC≥99.5）是目标，不是预测保证。

## GIT STATUS

独立研究仓库：服务器计划 `/root/autodl-tmp/NDP/research_versions`；本地 `work/ndp-version-repo`。
Remote：用户指定 https://github.com/yuliwen123456/NDP.git；GitHub已确认该仓库是公开仓库且连接器有push权限。
主分支main；Baseline commit `ca76a699cd17d85a90f71aef9c950912af5ad332`；tag `baseline-v06-ap74.5532-fpr1.4473-auroc99.3600`。
上游源码原repo `/root/autodl-tmp/NDP/code/ndp` 仍指向343gltysprk/ndp，commit f11dfbe4181db03a6a036d46b52a308f45d0e255，未改remote。未跟踪checkpoint/data保持原样，不提交。
评估源码STU commit `8f0f09c2ca4bf7b665e0ae5919b4092ddae140a2`。

**公开push尚未完成。** 命令行凭据调用失败；随后连接器公开创建README被自动审批拒绝，理由是没有明确批准这份具体研究内容公开发布。保留本地commits，不绕过该拒绝。实际最终branch/SHA、服务器同步及测试结果另见交付核验记录。

保护机制：manifest SHA256 + 提交前历史文件检查 + append-only VERSION_HISTORY + pre-commit hook。Git hook可被root绕过，不等于操作系统只读锁；每个新clone需安装hook。恢复使用新的git worktree，不reset现有工作目录。服务器checkpoint仍需独立备份；Git只保存路径、hash和源码，不能恢复已丢失的大权重。

快照config保留了历史baseline.yaml原字节，其中save_dir仍是早期默认目录；V06真正使用的save_dir由runtime_v06/plan.json的启动覆盖值决定。应联合读取，不能拿默认目录误认训练输出，也不能为了美化记录修改冻结文件。

## FILES TO BE CREATED

本轮新增并交付：`experiments/v06_baseline/`（源码、运行入口、配置、命令、指标、hash清单）、`analysis/analyze_score_distribution_v01.py`、两份score-distribution JSON、本报告、`scripts/check_version_integrity.py`、保护测试、`scripts/record_result_v01.py`、`.githooks/pre-commit`、`.gitignore`、`.gitattributes`、`AGENTS.md`、`VERSION_HISTORY.md`、根README，以及三个候选版本各自的README/config/metrics/train_command/eval_command/manifest。完整逐文件清单见交付清单。

以上全部位于新研究目录。算法文件仅列计划，必须方案获批后才新增，旧模型、loss、配置、评估源码不覆盖。每次后续结果写新时间戳文件，记录工具自动计算相对V06差值并追加history，proposal不反复覆写。
