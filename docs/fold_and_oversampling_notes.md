# rig 分析端的 fold 技巧 · 三种"折叠"辨析 · 过采样口径 — 学习笔记

2026-10-06。English version: [fold_and_oversampling_notes_en.md](fold_and_oversampling_notes_en.md)。整理自本轮关于 `polyphase_dft_fb_analysis.m`（原理图
[osfb_analysis_buffer.png](osfb_analysis_buffer.png)）与本项目 WOLA 引擎
（`conjugate_fb_toeplitz.py`）的讨论。姊妹文档：
[wola_vs_overlapsave.md](wola_vs_overlapsave.md)（两台机器的系统对比，中/英）。

---

## 1. rig 在算什么，fold 出现在哪一步

rig 参数：原型滤波器 Lp = OS·nfft = 1024 抽头，nfft = 512，BLOCKSIZE = hop = 256，
OS = 2。每个 tick（每 256 样本）要算全部 257 个 bin 的带通输出：

```
y_b[t] = Σ_{d=0}^{1023} p[d]·x[t−d]·e^{−j2πk d/512}
```

naive 直接算：257 个通道 × 1024 抽头复数乘加 ≈ **2.1 Mflops/tick**。

![fold 技巧](fold_trick.png)

fold 技巧的流水线（rig 原理图第二行）：**加窗 → flip → 按 mod nfft 切段 → 逐点相加（fold）→ 一次 512-FFT**：

1. **flip 后下标 = 延迟 d**：最新样本放 index 0，"延迟差 512"的两个抽头正好相距 512；
2. **fold 的合法性**：调制波形 e^{−j2πkd/512} 对 d 以 nfft = 512 为周期，所以延迟 d
   与 d+512 的两项**调制相位完全相同**——同相位项可以先把加法做掉（分配律重组），
   变换放后面一步完成；
3. 数学恒等式（数值验证 1e-14，rig 图注 scratch_osfb_fold_proof.py）：

```
Σ_d p[d]x[t−d]·W^{kd}  =  Σ_{j=0}^{511} W^{kj} · Σ_m p[j+512m]x[t−j−512m]
                           └ 512 点 DFT ┘   └─ fold：同相位段相加 ─┘
```

一句话：**把"延迟轴"先按 mod-nfft 折叠（alias），再做一次短 FFT** —— 这就是
过采样 DFT 滤波器组的标准**多相（polyphase）分解**实现。

## 2. 技巧的优势与代价

| 优势 | 说明 |
|---|---|
| **① 滤波器长度与 FFT 成本解耦**（结构性优势） | FFT 尺寸锁死 512；原型想多长就多长（更陡阻带、更强邻带抑制），FFT 一个 flop 不涨，只有加窗乘法线性涨。OS=4 时依然一次 512-FFT。**滤波器质量几乎免费** |
| **② 下游通道减半** | 产物 257 条 bin 序列而非 513 条——存储、每 bin 自适应滤波器、综合端全部减半；AEC 的 O(nbin) 是主导项 |
| **③ 零浪费的频谱网格** | 一次 1024-FFT 会给出 513 条 fs/1024 细网格 bin，但通道间距是 fs/512——中间 256 条插值 bin 白算；fold 把同相位项预先合并，只算真实存在的通道 |
| **④ 硬件/定点友好** | 更小 FFT = 更少蝶形、更小存储（Crochiere & Rabiner 1983 时代的标准实现理由） |

| 代价 | 说明 |
|---|---|
| 失去细网格 | 只有 fs/512 间距的 257 通道；需要细 bin 的设计（如 fbtoe 靠 bin 冗余压 off-grid floor）这是损失 |
| 记账负担 | flip/fold 对齐、synthesis 端 ×OS replicate、nfft·blocksize "magic gain"（MATLAB ifft 记账常数），都是技巧带来的管道复杂度 |

成本量级（每 tick，对数量级）：直接 2.1 Mflops → 无 fold 的 1024-FFT 路线 ~51 kflops
→ **fold + 512-FFT ~23 kflops**（比 1024-FFT 路线再省 2×，比直接省 ~90×）。

## 3. 我们的滤波器组：OS=1 特例，fold 是恒等式

`conjugate_fb_toeplitz.py` 的原型窗 q 长度**恰好等于 nfft**（1024 窗配 1024 点
rfft），"按 nfft 切段"只有一段，fold 退化为空操作：

```python
X = np.fft.rfft(self.q * x_frame)   # conjugate_fb_toeplitz.py:220 — 窗长=FFT长
```

| | rig (polyphase_dft_fb_analysis) | 我们 (conjugate_fb_toeplitz) |
|---|---|---|
| 原型长 Lp | 1024 = OS·nfft | 1024 = nfft |
| FFT 点数 | 512 | 1024 |
| fold | 需要（OS=2 段相加） | **恒等（一段）** |
| 产物 | 257 条粗 bin 序列 | 513 条细 bin 序列 |

两条路线是同一枚硬币：**fold 技巧允许独立调节"原型多长"（滤波器质量）和
"bin 多密"（通道数）**。我们把便宜花在细 bin（15.6 Hz 间距，邻带冗余 +
off-grid floor 受益），rig 花在长原型粗 bin。若未来做"更陡原型 + 减半通道"
的变体，此技巧直接搬运。

## 4. 三种"折叠/重叠"辨析（最容易混的一节）

| | OLS 的信号绕回 | **rig 的分析 fold** | WOLA 的合成 OLA |
|---|---|---|---|
| 作用对象 | **信号**卷积值 | **滤波器/延迟轴** | **帧输出流** |
| 机制 | 模运算把帧尾折回帧头 | 同调制相位段相加 | 相邻帧输出逐样本相加 |
| 产生新错误值？ | **是**（线性卷积里没有的值） | 否（严格恒等，1e-14） | 否（这正是 PR 重构本身） |
| 后果 | 必须丢头 / [0;e] | 无害，纯计算加速 | Σq² = k/2 常数，`_syn = 2/k` 归一 |

判别口径：**绕回需要"两条同帧序列做循环卷积"；fold 只是"把加法搬到 FFT 之前"
（同相位才能加）；OLA 是"错开摆放后相加"（COLA 条件保证平坦）**。三者互不相同。
可视化：[wola_folding.png](wola_folding.png)（上左：OLS 绕回；上右：WOLA 逐 bin
FIR 无帧边界可绕；中：WOLA 合成 8 窗加性覆盖，Σq² = 4 ± 4e-16；底部数值证明：
零权重直通重构 1.3e-15）。

## 5. 过采样率的两种口径（2026-10-06 校正，用户定义为准）

| 口径 | 定义 | 1024/128 几何 |
|---|---|---|
| **总冗余（滤波器组标准）** | 过采样率 = 独立子带数/抽取因子 ≈ **nbin/hop**；过采样 ⇔ nbin > hop | 513/128 ≈ **4×** |
| **逐信道（STFT/WOLA 工程；本仓库旧记法 k）** | 每条复 bin 超出自身 Nyquist 的倍数 = **nfft/hop** | 8× |

对实信号两者恰差共轭因子 2（独立 bin ≈ nfft/2）。**换算：标准过采样率 =
nbin/hop = k/2 = overlap factor / 2**；重叠百分比 = (nfft−hop)/nfft = 1 − 1/k。

各几何重贴标签（物理数字不变，只换名字）：

| 几何 | nbin/hop（标准） | nfft/hop（= overlap factor） | 判定 |
|---|---|---|---|
| rig 512/256 | ≈1 | 2 | **临界采样**（总冗余口径；rig 原名 oversampled_filterbank_*.m 的 "oversampled" 一词是逐信道方言，现已更名 polyphase_dft_fb_*.m） |
| harness 官方 512/128 | ≈2 | 4 | 2× 过采样 |
| fbtoe wrapper 1024/256 | ≈2 | 4 | 2× 过采样 |
| 甜点 1024/128 | ≈4 | 8 | 4× 过采样 |
| 1024/64 | ≈8 | 16 | 8×（仍崩——崩因是 lag 共线性随 nfft/hop 增长，过采样救不了） |

## 6. 下采样因子是什么

**下采样因子（decimation factor）= hop**：每条子带输出序列的抽取步长——带通滤波器
本在 fs 下连续产出，每 hop = 128 个样本取 1 个，子带采样率 = fs/hop = 125 Hz。
代码里就是 `process()` 每帧前进的步长（`conjugate_fb_toeplitz.py:211`，调用侧
驱动循环传的块长），没有第二个参数。临界对照：M = R 才临界（MDCT 音频编码即此，
省码率）；我们故意 M/R ≈ 4 倍冗余，买三样东西：邻带泄漏的重复覆盖（off-grid floor
下降）、更细的 lag 网格（d mod hop 表达）、更高的自适应更新率（fs/hop 次更新/秒）。

## 7. 速查卡

- fold = 多相分解：延迟轴 mod nfft 折叠 + 一次 nfft-FFT，严格恒等
- fold 存在条件：Lp > nfft（原型长于 FFT）；我们 Lp = nfft，无 fold
- 三种折叠：OLS 绕回（有害，唯一）≠ rig fold（无害）≠ WOLA OLA（PR 本身）
- 过采样率标准口径 = nbin/hop（>1 才叫过采样）；k = nfft/hop 是逐信道口径 = 2× 标准
- hop = 下采样因子 = lag 网格 = 更新率的倒数刻度 = 覆盖长度的单位（n_g·hop）

**参考**：Crochiere & Rabiner, *Multirate Digital Signal Processing*, 1983
（WOLA/多相分解）；Vaidyanathan, *Multirate Systems and Filter Banks*, 1993；
本仓库：[osfb_analysis_buffer.png](osfb_analysis_buffer.png)（rig 原理图）、
[fold_trick.png](fold_trick.png)（本文新图）、
[wola_folding.png](wola_folding.png)、
[wola_vs_overlapsave.md](wola_vs_overlapsave.md) / [英文版](wola_vs_overlapsave_en.md)。
