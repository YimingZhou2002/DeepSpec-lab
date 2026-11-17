# A 项结论：特征保真度不是可用的调度信号（否命题）

**一句话**：把「token 被接受了，但 drafter 当时的隐状态是错的」这件事**测出来**是可行的、标签免费，
但它**不能用来预测下一轮会不会失败** —— 简单数据上它有效但没超过 DSpark 自带的置信度头，
困难数据上它和置信度头**双双失效**。

⇒ **A 的朴素形式（做 fidelity head 来调度）不成立。** 这是一个干净的、有把握的否定结论，
成本是一天，而不是六周。

## 假设与为什么要测

投机解码的正确性单位一直是 **token**。但现代 drafter 越来越多地在**自己预测的特征**上做条件
（DSpine 逐层注入 predecessor 的 predicted feature、DSpark 的 Markov head、DFlash 的块内并行）。
于是理论上存在一个从未被验证的失败模式：

> **token 被接受了，但 drafter 当时的"信念"（hidden state）是错的**，
> 这个错误在后续轮次里转化为拒绝，而 token 级验证永远看不见它。

## 一个先决的结构事实（读源码得到，决定了 A 的适用范围）

`deepspec/eval/dspark/evaluator.py` 的 `_update`：

```python
verified_target_hidden = extract_context_feature(
    verification.target_output.hidden_states, self.draft_model.target_layer_ids)
context.target_hidden_states = verified_target_hidden[:, :accepted + 1, :]
```

⇒ **每一轮之后，drafter 的上下文特征都被换成 target 在已验证位置上的真值。**

所以 **DFlash/DSpark 不存在跨块的特征漂移** —— 漂移只可能发生在**块内**
（块是一次并行前向，位置之间靠 Markov head / 并行注意力互相条件）。
这与 EAGLE-3 的**逐位置串行**漂移是两种不同机制，也是 Attention Drift (2605.09992)
只覆盖 EAGLE-3、明确不碰 DFlash/DSpark 的原因。
**A 的可测范围因此被限定在块内漂移上。**

## 度量为什么定义明确、且标签免费

- **对齐关系**：draft 的 `block_hidden[:, i, :]` 预测 draft token *i*；
  target 在 verify 里第 *i* 个位置的隐状态**也**预测 draft token *i*。⇒ **逐位置 i↔i 对齐**。
- **两者可比**：draft 的 `lm_head` 初始化时从 target 复制
  （`modeling.py:281` `self.lm_head.weight.copy_(lm_head.weight.detach())`），hidden_size 相同。
- **标签免费**：verify 返回的 target 隐状态**就是** drafter 预测隐状态的真值。
  **零额外前向、零额外标注。**

两个指标都在 verify 里本来就算好了：

| 指标 | 定义 | 含义 |
|---|---|---|
| `dh` | `‖h_draft_i − h_target_i‖ / ‖h_target_i‖` | 特征空间信念误差 |
| `tv` | `½·Σ_v |p_target_i(v) − p_draft_i(v)|` | 分布空间信念误差（`draft_probs` vs `target_probs`） |

对照组：**DSpark 自己的置信度头**在同一轮给出的接受概率。
**A 成立的必要条件是 fidelity 的预测力明显强于它** —— 否则这个头已经把事做完了。

## 方法

```bash
CUDA_VISIBLE_DEVICES=2 A_DATASET=gsm8k    python scripts/profiling/a_feature_fidelity.py out.json 10
CUDA_VISIBLE_DEVICES=5 A_DATASET=mt-bench python scripts/profiling/a_feature_fidelity.py out.json 12
python scripts/profiling/analyze_fidelity.py out.json
```

插桩全是 monkeypatch，**不改框架一行**：`forward_dspark_draft_block` 取块隐状态（注意
`evaluator.py` 是 `from ... import` 按名绑定，**必须同时替换 evaluator 模块命名空间里的那个名字**）、
`_propose` 取 `draft_probs` 与 `confidence_logits`、`verify_draft_tokens` 取 target 隐状态并对齐配对。

统计口径：只在**同一 prompt 内**的相邻轮之间配对；偏相关 = 先对 `k_t` 回归取残差再算 Spearman；
显著性用**置换检验**（2000 次）。共 2200 / 3960 次置换抽样，用固定种子。

同时收集**本轮接受长度 `k_t` 的裸自相关**作为上界参考：如果连"这一轮接受得多，下一轮也多"
都不成立，那任何更弱的信号都不必谈。

## 结果

### gsm8k（10 prompt，408 对相邻轮）

| 预测子 | 裸相关 | 控制 `k_t` 后偏相关 | 置换 p |
|---|---|---|---|
| `k_t`（本轮接受长度，自相关上界） | 0.307 | — | — |
| `dh`（特征信念误差） | 0.340 | **0.220** | **0.000** |
| `tv`（分布信念误差） | −0.297 | **−0.255** | **0.000** |
| `conf`（DSpark 置信度头） | 0.295 | **0.258** | **0.000** |

### mt-bench（12 prompt，593 对相邻轮）

| 预测子 | 裸相关 | 控制 `k_t` 后偏相关 | 置换 p |
|---|---|---|---|
| `k_t`（自相关上界） | 0.100 | — | — |
| `dh` | 0.015 | **−0.030** | **0.481** |
| `tv` | 0.000 | **−0.040** | **0.355** |
| `conf`（DSpark 置信度头） | 0.015 | **0.043** | **0.301** |

## 判读

1. **现象是真的。** 在 gsm8k 上，控制本轮接受长度后，信念误差仍显著预测下一轮
   （`dh` p<0.001，`tv` p<0.001）。「token 被接受但信念错了 ⇒ 之后更容易失败」这一因果链**可测且存在**。
2. **但没有增量。** `dh`(0.220) / `tv`(−0.255) 与 DSpark 自己的 `conf`(0.258)
   **在同一个量级**，`conf` 还是最强的那一个。⇒ **DSpark 的置信度头已经把这件事做完了。**
3. **困难数据上什么信号都没有。** mt-bench 上 `dh`、`tv`、`conf` **三者全部不显著**（p=0.30~0.48），
   连本轮接受长度的自相关都从 0.307 掉到 0.100。
   n=593 时 95% 置信区间约 ±0.08，**足以排除 |偏相关|>0.09 的效应**，所以这是一个**有检验力的否定**。
4. **符号不一致本身是个线索。** `dh` 在 gsm8k 上偏相关是**正的**（+0.220），与漂移假说方向相反；
   而 `tv` 是负的（−0.255），与假说一致。`dh` 的正号很可能是**混杂**：它的分母是逐位置均值，
   而特征误差随位置增大、均值又随接受长度 `k` 增长，故 `dh` 与 `k_t` 存在非线性耦合，
   线性控制 `k_t` 去不掉。**契约上应以 `tv` 为准，`dh` 的正号不应被解读为支持漂移假说。**

## 结论

> **A 的朴素形式被拒绝。** 特征保真度**不能**作为单轮调度的信号：
> 简单数据上它被 DSpark 现有的置信度头支配，困难数据上无人可用。

### 两个复用价值（本项的真实产出）

1. **否定本身是结论，且是新的。** 目前所有「自适应调度提案长度 / 推测深度」的工作
   （DSpark、AdaSpec、SpecDec++、Nightjar、ForeSpark、SpecBound…）都**隐含假设本轮状态可以预测下一轮**。
   本项在有检验力的样本上证明：**在困难数据上这个假设不成立，连它们自己的置信度头都不成立。**
   这给整个子领域划了一条边界条件。
2. **免费的真值标签是现成的。** 「target 在已验证位置的隐状态 = drafter 预测隐状态的真值」
   这个对齐关系成立且零成本，**可用于训练**（辅助损失），只是**不可用于推理期调度**。
   这条路与 KVShot 的负结果（修表示只换来 HF MAT 5.01→5.04）是同一侧，**风险已知，谨慎对待**。

### 对本项目原始设想的直接影响

最初的设想是「draft 与 verify 流水，用置信度调度前瞻」。本项说明：
**「用当前状态预测下一轮」这条路在 bs=1 的困难数据上是堵死的。**
叠加 B 项的结论（[cost-model.md](cost-model.md)：成本对 γ 与 batch 都不敏感 ⇒ 没有成本信号可优化），
**两个方向上的信号都不存在**：既没有成本可省，也没有未来可预测。

## 口径与局限

- 只测了 **DSpark + Qwen3-4B**。结论是否迁移到 DFlash / EAGLE-3 未验证。
- gsm8k 上 τ≈6.0、`k` 集中在 5-7；mt-bench 上 τ≈2.4、`k` 集中在 1-3。
  **`k` 的方差小会衰减相关系数**，这也是 mt-bench 上自相关只有 0.100 的部分原因；
  因此 mt-bench 的否定应表述为「**在这个 k 分布范围内容量很小**」，而非「绝对无效应」。
- 单 prompt 最多 320 新 token，10-12 条 prompt。**样本量对 0.1 量级的效应偏紧。**
- 置信度头在本探针里是**手动 `start()`** 的（绕过了 `evaluate()` 的数据集循环），
  其 `observe()` 的耗时被算在正常流程内；A 项只看它的**输出**，不看它的耗时。
