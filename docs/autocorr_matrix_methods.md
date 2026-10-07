# 自相关矩阵的四种用法：correlation / error / PFCG / RLS

2026-10-06。English version: [autocorr_matrix_methods_en.md](autocorr_matrix_methods_en.md)（配套英文图 autocorr_matrix_roles_en.png）。回答两个问题：（1）error 梯度是不是已经包含在 PFCG（`pfdaf_cg.py`）里了？
——**是**；（2）几种"利用自相关（Toeplitz）矩阵"的方法到底差在哪？——**差在矩阵的
职责，不在矩阵的形式**。配套图：[autocorr_matrix_roles.png](autocorr_matrix_roles.png)。
姊妹文档：[wola_vs_overlapsave.md](wola_vs_overlapsave.md)、
[fold_and_oversampling_notes.md](fold_and_oversampling_notes.md)。

---

## 1. 直接回答：error 梯度早已在 PFCG 里

`PFDAF_CG.apply()` 第 4 步（`pfdaf_cg.py:178-194`，García Morales 2006 的式 18）：

```python
gacc[iref] = γ·gacc[iref] + (1−γ)·conj(rx)·E_zh     # Φ：γ平均瞬时 [0;e] 误差梯度
R[iref]    = γ·R[iref]    + (1−γ)·conj(rx)·rxᵀ      # Gram：与梯度同一个 γ 平均
```

方向 Φ 就是**γ 平均瞬时误差梯度**——与 fb-toeplitz 的 error 模式（`φ = γφ +
(1−γ)·conj(rx)·E`）同一个无偏家族，同一出处（PAES/AES 2006），都不是 Chang &
Willson (TSP 2000) 的东西（论文的方法 = correlation 残差，见 §5 历史节）。

PFCG 与 fb-toeplitz error 模式的六个精细差别：

| 轴 | PFCG | fb-toeplitz error |
|---|---|---|
| 机器 / 准则 | OLS + **[0;e]**（时域帧误差） | WOLA + **逐 bin 标量**（结构无头） |
| 矩阵形式 | **FULL Gram** R（Q×Q，不做 Toeplitz 投影） | **Toeplitz 投影** T = toeplitz(autoR) |
| 时间常数 | 梯度与 R 用**同一个 γ**（论文的 N 块平均，同率老化） | 方向 γ=0.1，矩阵 β=0.999（**解耦**的长窗） |
| 曲率地板 | P_b = trace(R)/N_G（平均标量）+ alpha_max 限幅 | 当前帧逐 bin 功率 X2（向量，δ=1 → NLMS 最坏步） |
| 内迭代 | k_max 模型迭代 g ← g − α·R·v（流式最优仍 =1） | k_max>1 发散（冻结 T 失配） |
| 路径层 | G 投影 [I_hop,0]（每帧） | 无需（逐 bin 标量权重） |

## 2. 一页总表（图的 ① 号面板）

| 方法 | 方向谁给 | 矩阵的职责 | 形式 / 维护 | canonical ERLE |
|---|---|---|---|---|
| correlation (fb-toeplitz) | 窗口方程组残差 g = rcross − T·w | **定义目标 + 定步幅**（目标 T⁻¹rcross 每帧漂移） | Toeplitz 投影，β 长窗 | 9.35（最佳 β=1, δ=8） |
| error hybrid (fb-toeplitz) | 瞬时误差梯度 φ（真路径处期望零，无偏） | **只定步幅** ⟨v,Tv⟩ + δ·P_inst 地板 | Toeplitz 投影，β 长窗、γ 独立 | 29.78 |
| PFCG (pfdaf_cg) | γ平均瞬时 [0;e] 梯度 Φ（同无偏家族） | **只定步幅** + k_max 模型迭代 | FULL Gram 不投影，同 γ | 26.11* |
| RLS (fb-toeplitz, solver='rls') | 精确 Newton K·ξ | **全部**：P = R⁻¹ 精确逆，方向+步幅一次到位 | FULL Gram，Sherman-Morrison 秩一 | 44.68 |

*PFCG 在 OLS 512/128 几何，其余在 WOLA 1024/256——不是严格同机 A/B。

## 3. 四种方法的更新公式并排

**① correlation**（类默认；`gradient='correlation'`）

```
T  = toeplitz(autoR)                     autoR ← β·autoR + α_acc·conj(rx)Xᵀ
g  = rcross − T·w                        ← T 第 1 次出现：定义目标（会漂）
v  = g (+ β_cg·v_prev)
α  = 0.999·⟨g,v⟩ / (⟨v,T·v⟩ + δ·P̄·‖v‖²)   ← T 第 2 次；P̄ = 窗平均功率(标量)
w += α·v
```

**② error hybrid**（`gradient='error'`）

```
T  同上（β=0.999 长窗）
φ  ← γ·φ + (1−γ)·conj(rx)·E              ← T 不参与；E = D − Σ wⱼrxⱼ（无偏方向）
v  = g (+ β_cg·v_prev)
α  = 0.999·⟨g,v⟩ / (⟨v,T·v⟩ + δ·P_inst·‖v‖²)  ← T 唯一出现；P_inst = 当前帧功率(向量)
w += α·v
```

**③ PFCG**（`pfdaf_cg.py`，OLS + [0;e] + G 投影）

```
R  ← γ·R + (1−γ)·conj(rx)rxᵀ             ← FULL Gram，与梯度同一个 γ
Φ  ← γ·Φ + (1−γ)·conj(rx)·E_[0;e]        ← 方向；E_[0;e] = rfft([0; irfft 尾])
g  = Φ（每帧重启；k_max 次内迭代 g ← g − α·R·v）
α  = −⟨g,v⟩ / (⟨v,R·v⟩ + δ·P_b·‖v‖²)，限幅 alpha_max·sd   ← R 只在这里（+模型迭代）
w += α·v → G 投影 [I_hop,0]
```

**④ RLS**（`solver='rls'`；Haykin ch.9）

```
K  = P·conj(u) / (λ + uᴴPu)              ← P = R⁻¹ 由 Sherman-Morrison 秩一维护
P  ← (P − K·(uᴴP)) / λ
w += K·ξ                                  ← 方向+步幅一步精确（Newton）；ξ = 先验误差 E
```

## 4. 核心论点：矩阵的职责决定成败，形式无罪

解剖实验（FINDINGS Addendum 5）证明 Toeplitz 投影本身几乎精确：真实数据 bin 100
末窗 **‖R−T‖/‖R‖ = 0.37%**，cond(T) ≈ cond(R) ≈ 220（图 ② 两块热图肉眼难分）。
冻结精确解同一 T 系统 = **45.4 dB**——连 correlation 模式的"带噪方程组"不动点都
是极好的。所以四种方法的 35 dB 差距（9.35 → 44.68）全部来自**矩阵被赋予了什么
职责**：

- **定义目标**（correlation）：目标 = T⁻¹·rcross，rcross 每帧被 b 噪声重写，
  cond≈220 把位移放大 220 倍 → 方向指向一个漂移点。阻尼救不了（δ 1→8 只
  +0.4 dB，Addendum 8），方向本身错了。
- **只定步幅**（error / PFCG）：方向来自无偏的瞬时误差梯度（真路径处期望零），
  矩阵错了只影响迈多大步（慢/快），永不改变目的地。
- **精确逆**（RLS）：P = R⁻¹ 递归维护 + 衰减荷载 δ·λⁿ + 门控防 wind-up，
  方向步幅双精确 → 一步顶 CG 数百帧。

## 5. 数字与历史

数字（图 ③）：correlation 9.35（β=1, δ=8 最佳；语音 20 s 渐近 14.46；平稳在线
网格渐近 38.08@64s——唯一亮点）< PFCG 26.11 ≈ error 29.78 < RLS 44.68；天花板
45.4（冻结精确解）。全官套件（2026-10-06）：nlms 22.27 / fdnlms 23.77 /
cgmdf 23.77 / pfcg 26.11 / **fbtoe 44.68 冠军**。

历史线：Chang & Willson (2000) 的 correlation-CG 是出发点（Table I 逐式实现，
identity 探针验证其"特征值散布无关"结论成立）→ 流式语音上方向漂移暴露 →
PAES 2006 的瞬时误差梯度（PFCG 的 Φ，fb-toeplitz error 的 φ）接管方向、矩阵
降级管步幅 → 解剖证明矩阵形式无罪、轨迹才是全部 → RLS（Sherman-Morrison +
衰减荷载 + 防风门控）终结比赛。全帧 correlation 反例（cliff 5.94 dB）已于
2026-10-06 删除（代码在 git 2f23d9d）。

**参考**：Chang & Willson, IEEE TSP 2000（CG1/CG2/Table I）；García Morales et
al., AES 2006（PBFDAF-CG，式 8/10/18/22）；Haykin, *Adaptive Filter Theory*
ch.9（RLS）；本仓库 results/2026-10-06_conjugate-fb-toeplitz/FINDINGS.md
（Addendum 4-8 全部实验）。
