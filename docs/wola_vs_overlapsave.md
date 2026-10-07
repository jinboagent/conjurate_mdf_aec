# WOLA（滤波器组）与 Overlap-Save：两台机器的本质区别

2026-10-06。English version: [wola_vs_overlapsave_en.md](wola_vs_overlapsave_en.md)。回答三个层层递进的问题：

1. WOLA 和 Overlap-Save 是不是两个不同的方法？（是——但比"两个方法"更尖锐：是两台不同的机器）
2. WOLA 是不是就是 Overlap-Add？（**不是——这是最大的命名陷阱**，见 §6.1）
3. "频域的滤波"是不是 Overlap-Add 和 Overlap-Save 的本质区别？（**不是**。OLA 和 OLS 之间没有任何本质区别——它们是同一台机器的两种记账方式；"频域滤波"真正分开的，是**快速卷积家族（OLA/OLS）**与**滤波器组家族（WOLA）**，见 §4）

本文以仓库代码为准：OLS 机器 = `FD_NLMS.py`（2026-10-06 前名 conjugate_mdf.py；CONJUGATE_MDF 已于 2026-10-06 作为 β 选项并入 FD_NLMS）与 `pfdaf_cg.py`；WOLA 机器 = `conjugate_fb_toeplitz.py`（NLMS 兄弟 `conjugate_full.py` 已于 2026-10-06 删除，量测存档）。所有数字为官方基准对 `test_subband_echo_cancellation.py`（canonical：延迟 800 样本 + 0.40 增益）当天实跑值。

---

## 0. 一页结论

| | 快速卷积家族 | 滤波器组家族 |
|---|---|---|
| 代表 | Overlap-Save, Overlap-Add | WOLA（加权重叠相加） |
| FFT 的角色 | **计算工具**：把时域卷积算快 | **滤波器组本体**：bin b 就是第 b 个带通滤波器的输出 |
| 滤波器住在哪里 | 时域（一条完整 FIR；频谱只是它的表示） | 子带域（每个 bin 一条复数 FIR，沿子带时间轴） |
| 频域乘法 X·W 的含义 | 一次循环卷积（= 线性卷积去掉头/补零） | 不发生（没有两条"同帧"频谱相乘这回事） |
| 循环混叠头 | 存在，必须处理 | **结构上不存在** |
| [0;e] 准则 | OLS 自适应**必需**（丢头/旧区） | **无处安放，也不需要** |
| 权重数组 | `W [nbin, N_G]`，每列是跨 bin 的一条谱（G 投影跨 bin 耦合） | `w [nbin, n_g]`，每行是 bin b 自己的抽头（bin 间零耦合） |
| 打分（准则）在哪 | 时域帧：irfft → e 帧 → [0;e] → rfft | 子带域标量：`E_b[m] = D_b[m] − Σⱼ w_b[j]X_b[m−j]` |
| 分析/合成窗 | 无（矩形帧 + 丢头记账） | sqrt-Hann 两侧加权 + PR 重构 |
| 因果延迟 | nfft − hop | nfft − hop（相同公式） |
| 官方 ERLE | NLMS 22.27 / FD_NLMS 23.77 / CGMDF 23.77 / PFCG 26.11 | WOLA-NLMS 10.66 / **FB-Toeplitz 44.68（冠军，RLS 求解器）** |

一句话：**OLA 和 OLS 的区别是"块的边界怎么记账"；快速卷积和 WOLA 的区别是"FFT 到底是什么"。** "频域滤波"（逐 bin 子带滤波）属于后者独有。

---

## 1. 快速卷积家族：OLA 和 OLS 是同一件事

### 1.1 它们加速的是同一条时域 FIR

设滤波器是时域 FIR `h[0..M−1]`（本仓库语境下就是回声路径的估计），要算 `y = x * h`（线性卷积）。直接算每输出一个样本要 M 次乘加；FFT 把它变成频域逐点乘法，代价 O(nfft·log nfft) 批量完成。

关键认识：**在 OLA 和 OLS 里，频域只是加速器，滤波语义从头到尾在时域。** 频谱 `W = FFT(h)` 不是"滤波器本身"，只是 h 的另一种写法。两个方法算的是**同一个**线性卷积，结果可证恒等（附录 A，数值 1.8e-15）。它们的全部区别只是**块的边界怎么衔接**。

### 1.2 Overlap-Add：补零 + 尾部相加

```
输入:      |——L——|——L——|——L——|        (不重叠切块)
           ↓ 每块补零到 N ≥ L+M−1
块卷积:    y_i = IFFT( FFT(x_i 补零) · FFT(h 补零) )   ← 长度 N
输出:      |——L——|
              ＼＼＼
               |——L——|                 ← 块 i 的尾巴 (M−1 个样本)
                ＼＼＼                    落进块 i+1 的输出区
                 |——L——|                  → 相加
y = y_0 + y_1 + y_2 + ...   (放在偏移 i·L 处直接累加)
```

因为补零到 N ≥ L+M−1，每块的循环卷积**本来就是**线性卷积——没有混叠，没有头要丢。块 i 的输出尾巴越过自己的 L 个样本、延伸进下一块的地盘，直接加起来恰好就是连续卷积。OLA 从头到尾无混叠，连"丢头"都不需要。

### 1.3 Overlap-Save：滑动帧 + 丢头

```
滑动帧 (每步前进 L, 帧长 N, 内含 M−1 个历史样本):
   frame:  [旧 x … 旧 x | 新 x 新 x … 新 x]
            ←– M−1 –→  ←––– L 个新样本 –––→

循环卷积:  y_circ = IFFT( FFT(frame) · FFT(h 补零) )
           y_circ[0 … M−2]   ← 头: 绕回混叠 (垃圾, 丢)
           y_circ[M−1 … N−1] ← 尾: = 线性卷积 (保留, 长度 N−M+1 = L)
```

帧**不补零**（历史样本占着位置），所以 FFT(frame)·FFT(h) 的 IFFT 是**循环**卷积：帧尾部的影响从帧头绕回来，污染前 M−1 个样本；其余样本等于线性卷积。"Save"这个名字的由来：保存帧里的历史，**丢掉被污染的头**，只留尾部 L 个干净样本。

（教科书单滤波器 OLS 的滑动步长 L = N−M+1；本仓库的 FDAF 是它的高重叠版：步长 hop = N/4，见 §3。）

### 1.4 头的数学来源（一段话）

长度分别为 A、B 的两个序列，循环卷积（长度 max(A,B)）与线性卷积（长度 A+B−1）的差 = 线性卷积尾部溢出的 B−1 个样本，被模运算**绕回**到开头。OLS 不补零（B = 帧长吃满），所以头必然存在；OLA 补零（N ≥ L+M−1），溢出区根本不存在。**头不是"误差"，是记账方式决定的数学伪影**——但它一旦被自适应准则"打分"，就变成真问题（§5）。

### 1.5 OLA vs OLS：只差记账

| | Overlap-Add | Overlap-Save |
|---|---|---|
| 输入块 | 不重叠，各补零 | 滑动重叠，不补零 |
| 循环卷积的混叠 | 无（补零消除） | 有（头 M−1 样本） |
| 边界处理 | 尾部**相加** | 头部**丢弃** |
| 输出 | 每块贡献 L 个样本（尾部额外 M−1） | 每块贡献尾部 L 个样本 |
| 结果 | 线性卷积 | 线性卷积（同一个） |

两者是同一线性卷积的两种分块方案，没有任何性能或语义差别——所以"OLA vs OLS 哪个本质更好"是个空问题。**真正的问题在下一层：这台机器被用来做自适应时，头就不再只是记账伪影了**（§5）。

---

## 2. 滤波器组家族：WOLA

### 2.1 FFT 在这里不是加速器，是滤波器组本身

WOLA 机器里没有"时域 FIR 再变换"这件事。FFT 的第 b 个 bin **就是**第 b 个带通滤波器的输出：

- 带通滤波器 b 的冲激响应 = 原型窗 q 调制到中心频率 `b·sr/nfft`；
- `X_b[m] = rfft(q · frame_m)[b]` = 该滤波器输出在时刻 m 的样本，按 hop 抽取（子带采样率 sr/hop）；
- 滤波器组的通带形状 = q 的频响 → 相邻带有重叠 → 这个重叠由**过采样因子 k = nfft/hop** 和**完美重构设计**（§2.4）吸收。

这不是类比而是恒等式：把"加窗→FFT→每 hop 步进"逐 bin 展开成时域卷积，得到的就是 513 条带通滤波器各自卷积再抽取（仓库验证：docs/osfb_bandpass_view.png——单频正弦落进对应 bin；docs/osfb_analysis_buffer.png——分析端的 flip+fold+FFT 与逐条带通滤波逐样本一致到 1e-14）。

### 2.2 三个下标：b、m、j

自适应权重 `w [nbin, n_g]` = `[513, 8]` 的含义（`conjugate_fb_toeplitz.py:195`）：

```
        频率轴 b = 0…512 (513 条互不往来的复数 FIR)
                ↓
bin b 的子带序列:  … X_b[m−3]  X_b[m−2]  X_b[m−1]  X_b[m]
                     ↕          ↕         ↕        ↕
                   w_b[3]     w_b[2]    w_b[1]   w_b[0]     ← lag 轴 j = 0…7
Ŷ_b[m] = Σⱼ w_b[j]·X_b[m−j]
```

- **b（bin）**：选哪条 FIR。每条只看见中心频率附近 ~sr/nfft 的窄带。
- **m（子带时刻/帧号）**：这条 FIR 的时钟。m 前进 1 = 全率时间前进 hop。
- **j（子带 lag/抽头号）**：抽头 j 负责 j 帧之前（≈ j·hop 个全率样本前）的路径段。

`X_b[m−j]` 取自移位寄存器 `buf_X`（`conjugate_fb_toeplitz.py:226-228`，每帧 roll 一次）——是**不同子带时刻**的真实历史样本，不是任何帧内的绕回。

### 2.3 逐 bin FIR 是"真·线性卷积"——无头的根源

`Ŷ_b[m] = Σⱼ w_b[j]·X_b[m−j]` 与时域 FIR 的唯一区别是：它作用在抽取后的子带序列上、系数是复数。这就是一条普通的卷积——**循环性没有可以发生的地方**：

- 没有两条同帧频谱相乘（乘法的两边一个是标量权重、一个是历史样本流）；
- 没有帧级 IFFT 后再取尾部的动作（误差在子带域直接成型，§5.2）；
- 各 bin 独立 → 513 个 8×8 小方程组 `T_b·w_b = rcross_b` 逐 bin 求解，bin 间零耦合。

### 2.4 路径如何被表达：hop 网格 + 复数抽头 + PR

全率路径 h[n]（例如 2048 个实抽头）投到子带域 ≈ 每 bin 一条 n_g 抽头复 FIR：

- **覆盖长度** = n_g·hop = 8×256 = 2048 样本，与全率路径对上；
- **延迟网格** = hop：真实延迟 d 先被"整块"吸收（块延迟对每条子带都一样），余量 `d mod hop` 由子带 lag 表达——所以 `d mod hop ≠ 0` 时落在网格缝隙上，靠相邻 lag 的复数内插去凑（off-grid floor，4× 过采样下 ~−2.5 dB 软坡，legacy OLS 全帧准则下是 −13 dB 悬崖）；
- **相位**：bin b 的子带信号带 `e^{j2πb·hop·m/nfft}` 型旋转，复数抽头天然吸收；
- **带间泄漏**：分析窗非理想，bin b 混进邻带成分。独立的逐 bin 权重原则上处理不了跨带耦合——这笔账由三层吸收：过采样冗余（k=4 时每个频率被 4 个 bin 覆盖）、合成窗匹配滤波（PR 设计把混叠在对消中抵消）、复数抽头的自由度。彻底的解法是 cross-band 滤波矩阵（多通道 Gram，PFCG 方向），本项目暂不需要。
- **完美重构**：分析窗 q 和合成窗 q 合计 q² = Hann 满足 COLA（重叠和 Σₘ q²[n−m·hop] = k/2 恒定）→ 无滤波时输出恒等输入。这就是 `_syn = 2/(nfft//hop)` 归一化的来历（`conjugate_fb_toeplitz.py:185`；这个常数曾经错成 1，4× 几何下输出整体翻倍——PR 探针恰好读 0.0 dB 的签名）。

### 2.5 WOLA 名字的由来

**W**eighted **O**verlap-**A**dd：合成端把每帧的 `q·irfft(E)` 放进累加器、按 hop 滑动**重叠相加**（`conjugate_fb_toeplitz.py:389-391`），"加权"指两侧的窗。它出自多速率信号处理经典（Croisier/Crochiere/Rabiner，1983）——与 §1.2 的快速卷积 OLA **同名不同物**，见 §6.1。

---

## 3. 两台机器其实惊人地同构——区别收敛为三条

把 OLS 家族按本仓库的实际形态（分区 FDAF，N_G=8）看清楚，会发现它和 WOLA 的乘法结构**同构**：

```
FD_NLMS (conjugate_mdf.py:369-378):          WOLA (conjugate_fb_toeplitz.py:226-232):
buf_Y_rx ← roll(旧帧谱), 塞入 Y_rx            buf_X ← roll(旧帧谱), 塞入 X
est_b = Σ_p W[b,p]·Y_rx[b, m−p]              Ŷ_b  = Σ_j w[b,j]·X_b[m−j]
       └── 逐 bin、跨帧 lag 的卷积 ──┘               └── 逐 bin、跨帧 lag 的卷积 ──┘
```

两边都是"每 bin 一条跨帧 lag 的 FIR"，lag 步长都是 hop。**真正的区别只剩三条**，而且每条都独立可开关：

| # | 轴 | OLS 家族 | WOLA 家族 |
|---|---|---|---|
| 1 | **误差在哪个域成型、被打分** | 拖回**时域帧**：`irfft(R)` → e 帧 → [0;e] → rfft（`conjugate_mdf.py:379-384`） | 留在**子带域**：`E_b[m] = D_b[m] − Ŷ_b[m]` 标量，直接打分（`:232`） |
| 2 | **窗** | 无（矩形帧；"加权"零次） | q 两侧加权 + PR 重构 |
| 3 | **权重的几何** | 整条谱列 `W[:,p]`：irfft 回时域是一条 hop 长滤波器，G 投影（`:419-423`）跨 bin 耦合 | 每 bin 标量行 `w[b,:]`：无任何跨 bin 操作 |

（对话里我曾把 OLS 简写成"同帧频谱相乘"——那是教科书单滤波器形态；分区形态如上，两边同构。这个修正让区别更锋利而不是更模糊：**差别不在乘法结构，在误差成型域、窗、权重几何。**）

---

## 4. 回答核心问题："频域滤波"是 OLA/OLS 的本质区别吗？

**不是。** 分两层说：

**第一层：OLA 和 OLS 之间没有本质区别。** 它们是同一台快速卷积机器的两种记账（§1.5），数值恒等（附录 A）。在它们俩里面，"频域"都只是算得快的手段——滤波器是时域 FIR，频谱乘法等于卷积，仅此而已。

**第二层："频域滤波"是快速卷积家族与滤波器组家族的分界线。** 如果"频域滤波"指的是"滤波器本身住在频域/子带域、逐带独立滤波"（WOLA 的做法），那么它恰好是 OLS/OLA **没有**的东西：

- OLS/OLA：**时域滤波器，频域加速**。把 `W = FFT(h)` irfft 回去必须还原出同一条 h——频谱是表示，不是家。
- WOLA：**滤波器就住在子带域**。`w_b[j]` 没有对应的"那条时域 FIR"的逐 bin 意义（把一行 `w[b,:]` 变换回全率时间轴得不到路径；把一列 `w[:,j]` 变换得到的是跨 bin 的时域形状——那是 G 投影管的对象，不是滤波动作本身）。信号的家也在子带域：误差、梯度、Toeplitz 方程组全部在子带域成型，时域只在入口（分析）和出口（合成）出现。

**判别试验（一招见效）**：看权重数组换个解释还成不成立。

- OLS 的 `W [257, 8]`：对某一列做 `irfft` → 得到一条 512 点时域滤波器（G 约束后支撑 ≤ hop）。"每一列是一条时域滤波器" ✓。
- WOLA 的 `w [513, 8]`：对某一行看，`w[b,:]` 是 bin b 的子带 FIR——它作用的对象是**抽取后的子带序列**，不是全率信号。没有任何一行/列能 irfft 回"那条"时域回声路径。✗

所以准确的一句话是：**"频域的滤波"（逐带子带滤波）是 WOLA 与整个快速卷积家族（OLA、OLS 都算）的本质区别；而 OLA 与 OLS 之间，连区别都只是记账。**

---

## 5. [0;e] 为什么只在一台机器里存在

### 5.1 头的产生条件：同帧相乘 + 时域打分

循环混叠头产生的**充要场景**：把同一帧的两条频谱相乘、IFFT 回时域。OLS 家族恰好每帧都这么干：

```
R = Y − Σ_p W_p ⊙ Y_rx(p帧前)          ← 频域逐点乘 (同帧谱 × 同帧谱)
e_frame = irfft(R)                      ← 时域帧
```

G 约束把每个分区滤波器的时域支撑压到 ≤ hop → 每帧循环混叠只污染帧头 hop−1 个样本；帧中段干净但属于**以前已经发过的旧数据**；只有最后 hop 是"干净且新鲜"的。于是：

- **输出侧**：emit 最后 hop（`conjugate_mdf.py:379`）——丢头，普适；
- **自适应侧**：[0;e] 把 e 帧的其余部分清零再 FFT 当梯度（`:381-384`）——混叠头是垃圾不能打分，旧区是重复计账不能打分。

**去掉 [0;e]（对全帧打分）会发生什么**：准则里混进随延迟变化的垃圾头，自适应去拟合它。实测（原 `conjugate_toeplitz.py` 引擎，2026-10-06 已删除，代码可从 git 2f23d9d 恢复）：延迟 mod hop ≠ 0 时 ERLE 崩到 **5.94 dB**（cliff）；延迟恰好对齐时反而更好（113 dB，无头可错）。这就是"准则悬崖"——完整几何见 docs/criterion_cliff.png、docs/cliff_geometry.png，帧几何见 docs/overlap_save_head.png、docs/window_placement.png。

### 5.2 WOLA：头无处安放

WOLA 的误差**从来不变成时域帧**：`E_b[m] = D_b[m] − Σⱼ w_b[j]X_b[m−j]` 是子带域的标量，直接用它打分、求梯度、解方程组。时域只在出口出现——合成端把误差谱 `irfft(E)` 加窗、重叠相加成输出流（`conjugate_fb_toeplitz.py:387-392`），那是**重构**，不是打分。

没有"同帧两条频谱相乘后 IFFT"这个动作 → 没有循环卷积 → 没有头 → **[0;e] 在这台机器里没有可以插入的位置**。这不是"我们选择不打分头部"，而是头部这个对象不存在（结构性的，不是规则性的）。

### 5.3 三层防御总表

| 层 | 管什么 | OLS 家族的做法 | WOLA 家族的做法 |
|---|---|---|---|
| **准则层**（打什么分） | 头/旧区不许进梯度 | [0;e]（显式规则） | 逐 bin 标量误差（结构无头） |
| **路径层**（权重能表示什么） | 循环权重不许污染路径 | G 投影/constraint（跨 bin） | PR 窗设计 + 复数抽头（无跨 bin 操作） |
| **求解层**（怎么解） | 步长/方向 | NLMS、CG（FD_NLMS β 选项、PFCG） | 逐 bin NLMS、子带 Toeplitz-CG（FB-Toeplitz） |

三层**互相独立**，实验证据：unconstrained + [0;e] = 26.16 dB > constrained 23.77（路径层关掉反而好——wrap 抽头给了子 hop 对齐自由度）；WOLA 两层都不需要任何规则。反例（原 `conjugate_toeplitz.py`，已删除）把准则层关掉（全帧打分）→ 5.94 dB，证明准则层是悬崖所在。

---

## 6. FAQ（最容易缠住的四个结）

### 6.1 WOLA 的 "OLA" 就是快速卷积的 Overlap-Add 吗？——不是！

| | 快速卷积 OLA（§1.2） | WOLA 的合成 OLA（§2.5） |
|---|---|---|
| 目的 | 把**时域卷积**算快 | 把**子带误差流**重构回时域 |
| 乘的什么 | FFT(x块)·FFT(h) | 无乘法（E 已经是结果谱） |
| 窗 | 无（矩形，靠补零） | 合成窗 q（靠加权） |
| 消除什么 | 无混叠可消 | 靠 PR 设计消带间混叠 |

两个 OLA 只在"重叠的输出块相加"这个动作上同名。WOLA ≠ OLA + 窗；**WOLA 根本不在快速卷积家族里**。同理，说"WOLA 需要 [0;e] 吗"这个问题本身就是错位的——[0;e] 是快速卷积家族自适应侧的补丁，WOLA 病灶都没有。

### 6.2 逐 bin 独立滤波，邻带信息丢了怎么办？

分析窗非理想 → 每个 bin 混入邻带成分 → 严格说逐 bin 独立权重是真实路径的近似。近似误差由过采样冗余（k 越大每频率被越多 bin 覆盖）+ PR 合成 + 复数抽头吸收；k=4 时官方对上 FB-Toeplitz 29.78 dB（相关 0.9995），说明近似在这个几何下足够好。要严格无近似需要 cross-band 权重矩阵（bin×bin 耦合），代价是多通道 Gram——是 PFCG 的方向，不是现在的瓶颈。

### 6.3 为什么子带内条件数随过采样变坏？

hop 是 lag 网格：k = nfft/hop 越大，相邻 lag 的子带回归子越共线（4× 时相邻 lag 相关 ρ≈0.76，白噪声真实条件数 ~4，4× 下 lag 方向条件数 ~173）。这是**表达的代价**（网格细了才准），不是病：求解层用 δ·P 曲率地板把有效条件数压到 ~1+1/δ（δ=1 时最坏退化为 NLMS 步长），实测 FB-Toeplitz 在 4× 下反而是冠军。

### 6.4 延迟、网格、覆盖怎么对账？

- 块延迟 ⌊d/hop⌋ 由"每条子带共同的块移位"吸收（harness 外部对齐或子带 lag 承担）；
- 余量 d mod hop 落在 lag 网格缝上 → off-grid floor（4× 时软坡 −2.5 dB）；
- 路径长度 ≤ n_g·hop 才能表达全（canonical 50 ms = 800 样本 << 8×256=2048 ✓）；
- 两台机器因果延迟公式相同：nfft − hop（OLS 512/128 → 384 样本 = 24 ms；WOLA 1024/256 → 768 = 48 ms）。

---

### 6.5 过采样率的两种口径（2026-10-06 用户校正）

| 口径 | 定义 | 本文档几何（1024/128） |
|---|---|---|
| **总冗余（滤波器组标准）** | 过采样率 = 独立子带数/抽取因子 ≈ **nbin/hop**；过采样 ⇔ nbin > hop | 513/128 ≈ **4×** |
| **逐信道（STFT/WOLA 工程惯例，本文沿用 rig 约定写作 k）** | 每条复 bin 超出自身 Nyquist 的倍数 = **nfft/hop** | 8× |

两者对实信号恰好差共轭因子 2（独立 bin ≈ nfft/2）。换算：**标准过采样率 = nbin/hop = k/2 = overlap factor / 2**。本文所有 "k = nfft/hop" 的说法按此换算即得标准口径；rig 512/256 在标准口径下是**临界采样**（≈1×），其原名 oversampled_filterbank_*.m 的 "oversampled" 属逐信道口径（现已更名 polyphase_dft_fb_*.m）。全部实测物理（cond、共线性、更新率、覆盖）是几何本身的函数，不受命名影响。

---

## 7. 仓库引擎全景（2026-10-06 官方实跑）（2026-10-06 官方实跑）

| 引擎 | 机器 | 准则层 | 求解层 | canonical ERLE | 判定 |
|---|---|---|---|---|---|
| NLMS（时域） | 无 FFT（金标准） | 逐样本 | NLMS | 22.27 | PASS |
| FD_NLMS | OLS | [0;e] + G 投影 | NLMS | 23.77 | PASS |
| CGMDF（FD_NLMS 的 β 选项；原 CONJUGATE_MDF 已并入 FD_NLMS 2026-10-06） | OLS | [0;e] + G 投影 | β=0 ≡ FD_NLMS（套件）；run_fdnlms β=0.3 → 24.47 | 23.77 | PASS |
| PFCG (PFDAF_CG) | OLS | [0;e]（γ 平均误差梯度） | 逐 bin Gram + CG | 26.11 | PASS |
| CONJUGATE_TOEPLITZ | OLS | **全帧（无 [0;e]）** | 长窗 Toeplitz-CG | 5.94 | FAIL（悬崖反例；**已删除 2026-10-06**） |
| WOLA-NLMS (CONJUGATE_FULL) | WOLA | 逐 bin 无头 | 逐 bin NLMS | 10.66 | FAIL（4× 几何下 NLMS 步长饥饿；**已删除 2026-10-06**——NLMS 基线由 FD_NLMS 承担） |
| **FB-Toeplitz (CONJUGATE_FB_TOEPLITZ)** | WOLA | 逐 bin 无头 | 逐 bin RLS（solver='rls'，λ=0.999，δ=0.1；次席：CG 误差梯度 29.78） | **44.68** | **PASS（冠军）** |

读法：机器（OLS/WOLA）与胜负无关，**准则层 × 求解层**的组合才决定成绩——同一台 WOLA 机器上，NLMS 求解 10.66、RLS 求解 44.68；同一族 OLS 机器上，[0;e] 23.77、全帧 5.94。CONJUGATE_TOEPLITZ 的存在价值就是证明"去掉 [0;e]"本身（而不是别的）导致悬崖（该引擎已于 2026-10-06 删除，量测存档 results/2026-10-06_conjugate-toeplitz/FINDINGS.md）。

代码地图：

- OLS 机器：`conjugate_mdf.py`（FD_NLMS：误差成型 :379-384，[0;e]，G 投影 :419-423）、`pfdaf_cg.py`
- 全帧反例：`conjugate_toeplitz.py`（**已删除 2026-10-06**；legacy 恢复版、bit 级同 2f23d9d，可从该提交恢复；`_herm_toeplitz_matvec` 已并入 conjugate_fb_toeplitz.py）
- WOLA 机器：`conjugate_fb_toeplitz.py`（分析 :218-223，移位寄存器 :226-228，子带误差 :232，求解 :266-383，合成 :387-392，_syn :185）；`conjugate_full.py`（**已删除 2026-10-06**，`_sqrt_hann` 已并入 fb_toeplitz）
- 图：docs/overlap_save_head*.png（OLS 头/丢头）、docs/window_placement.png（帧几何）、docs/criterion_cliff.png + docs/cliff_geometry.png（悬崖）、docs/osfb_analysis_buffer.png + docs/osfb_bandpass_view.png（滤波器组视角）

---

## 附录 A：OLA ≡ OLS ≡ 线性卷积（已验证）

`scratch_ols_ola_demo.py`，随机 x(64)、h(9)，N=16、L=8：

```
max |OLS - linear| = 1.78e-15
max |OLA - linear| = 1.78e-15
```

核心代码（节选）：

```python
y_lin = np.convolve(x, h)[:len(x)]            # 金标准

# OLA: 不重叠块 + 补零 + 尾部相加
for i in range(0, len(x), L):
    blk = np.zeros(N); blk[:L] = x[i:i+L]
    y_ola[i:i+N] += np.fft.irfft(np.fft.rfft(blk) * np.fft.rfft(h, n=N))[:N]

# OLS: 滑动帧(含 M−1 历史) + 整帧乘 + 丢头 M−1
for i in range(0, len(x), L):
    frame = np.zeros(N)
    tail = x[max(0, i-(M-1)):i+L]; frame[N-len(tail):] = tail
    yf = np.fft.irfft(np.fft.rfft(frame) * np.fft.rfft(h, n=N))
    y_ols[i:i+L] = yf[-L:]
```

两台"不同的机器"给出与 `np.convolve` 相同的结果——这就是 §1 的"OLA 与 OLS 是同一件事"的数值证明。WOLA 家族没有对应的恒等式可写：它的滤波器不住在时域，没有"那条 h"可以对（§4 判别试验）。

## 附录 B：参考

- Allen & Rabiner, *A Unified Approach to Short-Time Fourier Analysis and Synthesis*, Proc. IEEE 1977（STFT/WOLA 统一视角）
- Crochiere & Rabiner, *Multirate Digital Signal Processing*, 1983（多速率 WOLA 滤波器组）
- Shynk, *Frequency-Domain and Multichannel Adaptive Filtering*, IEEE SP Magazine 1992（FDAF/循环卷积与约束综述）
- Chang & Willson, *Analysis of Conjugate Gradient Algorithms for Adaptive Filtering*, IEEE Trans. SP 2000（FB-Toeplitz 求解层）
- PAES/Eneman et al. 2006（γ 平均误差梯度 φ = γφ + (1−γ)·conj(X)E）
- 本仓库：docs/DATAFLOW.md §7（准则悬崖理论）、results/2026-10-06_conjugate-fb-toeplitz/FINDINGS.md（WOLA 家族全部实验）
