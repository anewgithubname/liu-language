# 流 (Liu) — Juzhen 的实验描述语言 · 规格草案 v0.3

> 定位:**不是通用编程语言**,而是 Juzhen 后端的演示前端——一门描述机器学习
> 小实验的语言,或者更准确地说:**测度运输的可执行记法**。设计北极星四个词:
> **可沙箱、可审核、可复现、秒级反馈**。凡是伤害这四点的特性,一律不加。
>
> 名字取"流"(flow)之意——测度之流、梯度流、概率流,即这门语言的全部中轴。
> (v0.1–v0.3 讨论期间曾用名"演 (Yan)";自定名起项目仓库为 liu-lang。)

> **状态(2026-07-11)**:本文件是设计史与路线图的记录,§8 的示例保留
> 成文时语法。当前已实现表面以 `docs/liu-api.md` 为准,版本对账见
> `docs/roadmap.md`;§10 各条目的落地状态见其标题行与台账表。

文件扩展名 `.liu`。一个程序就是一次实验:自上而下执行,无用户定义函数、
无条件分支、无用户循环(迭代只存在于 `train`/`transport`/`flow`/`WGFlow`
等原语内部)。因此程序天然是一个有向无环的数据流,**所有维度、能力与出身
都可以在运行前静态检查**——错误在提交时报出,而不是训练到一半崩掉。

- v0.2 立起运输世界观:`transport`(学场)、`flow`(积分)、`#`(pushforward)、
  `inv`(pullback)、`~`(采样)、`via`(推断算法槽)。
- v0.3 把**散度(Divergence)**立为第一公民,引入测度空间的场/流对
  `WGField`/`WGFlow`(Wasserstein 梯度场及其梯度流),以及**时间反演**
  `reverse`。SVGD 与 diffusion 由此从原语降级为推论(见 §3.3、§4.6)。

---

## 1. 语法总则

- 每行一条语句;`//` 到行尾是注释;语句可用缩进续行。
- 两类语句:
  - **绑定**:`name = 表达式`
  - **动词**:`seed` / `train` / `generate` / `eval` / `plot`
- 程序第一句为 `seed n`(省略时解释器自动补 `seed 0` 并回显在结果里)。

## 2. 类型系统

核心类型(用户不写类型,由构造原语决定):

| 类型 | 是什么 | 从哪来 | 能力标记 |
|---|---|---|---|
| `Distribution` | 测度(惰性) | `gaussian`、`moons` 等原语;`T # μ` | 可精确采样?有 score? |
| `Dataset` | 有限样本 = 经验测度 | `μ ~ n`;内置 `mnist`、`text8`;`T # X` | 携带母测度(出身) |
| `Divergence` | 分布对上的泛函 D[·,·] | `reverseKL`、`W2` 等;`mirror(D)` | 所属族(f-散度/OT/…) |
| `Net` | 网络骨架 | `mlp`、`cnn`、`transformer` | |
| `Field` | 点空间 ℝᵈ 的速度场 | `transport ...`;`reverse(...)` | 可学习? |
| `Map` | 逐点映射 | `flow(v)`、`inv(T)` | 可逆? |
| `WGField` | 测度空间 P(ℝᵈ) 的梯度场 | `WGField(D, estimator=...)` | 依赖当前测度 |
| `WGFlow` | 测度演化(梯度流半群) | `WGFlow(u, steps=/time=)` | 无逐点逆 |
| `Trainable` | 判别/预测式模型 | `classifier(net)`、`lm(net)` 等 | |

另有输出类 `Figure` / `Report`;`Number` / `Vector` 为标量与字面量向量。

### 2.1 两对平行的"场/流":点空间 vs 测度空间

命名原则:**每种几何配一对场/流**(SDE/Markov 核是预留的第三对,v0.4)。

| | 点空间 ℝᵈ | 测度空间 P(ℝᵈ)(Wasserstein 几何) |
|---|---|---|
| 场 | `Field`(固定速度场) | `WGField(D, estimator=...)`(依赖当前测度) |
| 积分 | `flow(v, steps=, solver=)` | `WGFlow(u, steps=, lr=, time=)` |
| 产物 | `Map`(逐点映射;粒子独立) | `WGFlow` 对象(粒子彼此耦合) |
| `#` 语义 | 逐点映射提升到测度/点集 | 测度整体演化(交互粒子系统) |
| `inv` | 可逆 Map 合法 | **类型级非法**(`inv` 签名只收 Map) |
| 出身检查 | 软警告(允许创造性用法) | **硬错误**(起点不符则语义失效) |

三条被类型系统刻意强制的区分,也是这门语言最重要的三课:

1. **场 ≠ 映射**:中间隔一次积分(`flow` / `WGFlow`)。
2. **模型 ≠ 模型诱导的分布**:`transport` 返回场,`flow(v) # noise` 才是分布。
3. **点的流 ≠ 测度的流**:固定场下粒子各走各的;梯度流下粒子彼此耦合,
   `WGFlow # X` 是真实梯度流的 mean-field 近似,有限粒子数偏差真实存在。

## 3. 操作符与核心表达式

| 记号 | 类型签名 | 含义 |
|---|---|---|
| `μ ~ n` | Distribution → Dataset | 精确采样(要求可精确采样) |
| `target ~ n via alg(...)` | Distribution/Dataset → Dataset | 近似采样/推断(`svgd`、`langevin`、`diffusion`…);target 需求由算法声明 |
| `A \| B`,`0.3*A \| 0.7*B` | Distribution² → Distribution | 混合 |
| `T # μ` / `T # X` | Map × (Distribution\|Dataset) → 同右 | pushforward,T#μ |
| `G # X` | WGFlow × Dataset → Dataset | 测度演化;X 出身须匹配动端(硬检查) |
| `inv(T)` | Map → Map | 逆映射,免费且精确;**pullback = `inv(T) #`** |
| `flow(v, steps=, solver=)` | Field → Map | 点空间积分;数值近似槽 |
| `transport from A to B using f [for N steps] [with ...]` | → Field | 学场(flow matching);A、B 须可采样 |
| `WGField(D, estimator=...)` | Divergence → WGField | 散度的 Wasserstein 梯度场;统计近似槽 |
| `WGFlow(u, steps=, lr=, time=)` | WGField → WGFlow | 测度空间积分 |
| `reverse(G, estimator=denoiser(net))` | WGFlow → Field | **时间反演,花训练买**(DSM 拟合各时刻 score),返回 probability-flow ODE 的时变速度场——从测度空间桥回点空间 |
| `mirror(D)` | Divergence → Divergence | 镜像散度(ψ′ ≜ rφ′ − φ);f-散度族内 |
| `estimate(D)` | Divergence → Report | 散度的样本估计(f-散度:变分下界;W2:Sinkhorn) |

### 3.1 inv 与 reverse:一对刻意的对比

- `inv`:免费、精确、只属于 Map(ODE 流反向积分)。
- `reverse`:作用于 WGFlow,数学上存在(McKean 时间反演)但**要用各时刻
  边际的 score 换**——学这些 score 就是"训练一个 diffusion 模型"本身。
  ODE 实现返回 Field,进而 `flow` 出的 Map **可逆**(DDIM 反演/编码器白送);
  SDE 实现(祖先采样)返回 Markov 核,留给 v0.4 第三对。

**注记(inv 的三级价目表:免费 / 花内存 / 花训练)**:

> **`inv` 免费 ⟺ 场是自足的公式**(如训练好的网络:任何位置、时刻、
> 方向皆可求值)。场是**演化状态的读数**(梯度流的 v[q_t],由粒子系综
> 逐步估出)时,inv 不免费,但有两条付费通道:
> **花内存**——`flow(..., record=true)` 录下各步系综,NW 估计使场对
> 任意查询点逐点可重建(Φ̂_t(x) 是冻结系综的显式函数),反演 = 逐步解
> 不动点 y = x + lr·Φ̂_t(x);
> **花训练**——`reverse`(用 score 把读数场升级为自足场,再享受免费 inv)。

要点与修正记录:

1. **直接反号解出的是另一个系统**。不录史的反向模拟,场从反向演化中的
   系综估出,反号后是"从终点出发的散度梯度上升"——良定义,但不是前向
   的时间反演,不会沿原路退回。
2. **NW 平滑击破了"mean-field 不可逐点"的反对**(此前规格的论证在此
   过强,经作者挑战修正):粒子间耦合只进入**场的制造**,不进入**场的
   使用**——冻结每步系综后,Φ̂_t(·) 是诚实的单点函数,每步映射
   F_t = id + lr·Φ̂_t 是恒等的小扰动(lr·Lip < 1 时为微分同胚)。
   把粒子更新规则升级为场的函数估计,正是 NW/locallinear 一线
   (Liu et al. 2024)的贡献在语言层的回响。
3. **耗散是定量约束而非不可能性**:反演的条件数随收敛加深而恶化
   (对照倒放热方程),但梯度流近平衡时减速、收缩总量有限——demo 尺度
   上录史反演数值健康(实测 400 步深度收敛后初始云仍逐格恢复);
   诅咒在更深收敛/更细结构处才显现。对照:概率流 ODE 的场自足且
   非耗散,DDIM 反演免费且健康。

**注记(为什么 reverse 不是 `inv(flow(...))` 的糖)**:WGFlow 形态的前向
是模拟配方(注噪在期望中实现场里的 −∇log q_t 项),显式的点空间场**不经
estimator 不存在**,故 `flow(G)`/`inv(flow(G))` 无从写起。`reverse` 恰好就是
"付 score 的账、把隐式测度演化升级为显式 Field"的操作;付账之后,
`flow(reverse(G))` 与假想的"前向概率流场的反向积分"**重合**(概率流 ODE 的
双向性)。工程上标准配方固定为 reverse 方向,因为前向的噪声实现使各时刻
边际精确已知(数据 ⊛ 高斯,闭式),denoiser 在**真**边际上训练,估计误差
只在生成方向被积分一次,不复利。

### 3.2 出身一致性(provenance)

- 每个 Dataset 携带母测度(DAG 出身,结构相等可判定)。
- `WGFlow(WGField(D(p, q))) # X`:要求 X 的母测度与动端 q 结构相等
  (q 省略时以 X 的出身为准)。违例硬报错,文案讲原理
  ("起点不符,场的方向就不是该散度的下降方向")。
- **录史解除硬钉(2026-07 修订)**:`record=true` 冻结各步系综后,
  NW 读出对任意点逐点可算(如同训好的模型在训练集之外求值),回放
  在两个方向都自足——异源测度合法化,降级为**输出注记**,注记陈述
  答案的确切含义:`inv(T) # target` 读作"运回 `from=`"**当且仅当**
  前向流已收敛(descent 是初值问题,只有起点是声明边界;target 是
  t→∞ 极限而非记录终点的律)。硬钉只保护它真正的不变量:未录史的
  map 就是模拟本身,场随系综重估,异源粒子改变动力学。
- `Map # X` 出身不符仅软警告(潜空间插值等创造性用法是特性)。

### 3.3 脱糖规则

语法糖必须有精确定义。三大生成/推断范式全部展开到同一套原语:

```liu
// SVGD:顺着 reverseKL(target) 的梯度流走向目标
target ~ n via svgd(kernel=k, steps=s)
// ≡ WGFlow( WGField( reverseKL(target), estimator=nw(kernel=k) ), steps=s )
//     # (默认初始分布 ~ n)

// diffusion:沿 reverseKL(gaussian) 的梯度流把数据推平,再学会倒放
data ~ n via diffusion(net=..., time=3, steps=50)
// ≡ flow( reverse( WGFlow( WGField( reverseKL(gaussian, data) ), time=3 ),
//                  estimator=denoiser(net) ), steps=50 ) # (gaussian ~ n)

// langevin:暂留 via 糖(KL 梯度流,熵项由噪声实现);严格形式化待 v0.4 SDE 对
```

反向不留糖:`v # μ`(v 是 Field)不脱糖为 `flow(v) # μ`,必须显式 `flow`。

### 3.4 训练类表达式(train 的变体)

`transport` 与 `reverse` 本质上是 **train 的表达式形态**:三者都"用数据拟合
持久参数",只是监督信号的制造方式不同(train 用标注/自回归目标;transport
用端点配对回归速度场;reverse 用加噪轨迹回归 score)。`estimate` 作用于
f-散度族时(变分下界需拟合判别函数)同属此类。规格性后果,四条统一:

1. **从句语法统一**:训练类表达式一律接受 `for N steps` / `with lr=, batch=`;
2. **过程回报统一**:一律流式回报 `(step, loss)`,前端画实时曲线;
3. **`plot loss of X` 一律可用**(X 为 transport/reverse 的产物或 Trainable);
4. **资源核算统一**:其 steps 计入 §6.1 的总步数限额。

与之相对,`nw`/`locallinear`/`sinkhorn` 等 estimator 是**逐步重解、
无持久参数**的(非参数/逐实例);"花训练买的东西都有 loss 曲线可看,
没有 loss 曲线的东西不持有参数"——这条对偶是语言级承诺。

### 3.5 三大范式全家福(规格性注记)

| | `transport`(flow matching) | `svgd`(粒子法) | `diffusion` |
|---|---|---|---|
| 几何 | 点空间回归 | 测度空间梯度流(顺放) | 测度空间梯度流(倒放) |
| 散度 | (无,L2 回归) | reverseKL(target) | reverseKL(gaussian) |
| 已知端 | 两端样本 | 目标 score 或样本 | 目标样本 |
| 花训练买什么 | 速度场 | 不训练(估计) | 时间反演的 score |
| 可逆? | 是(Map) | 否 | ODE 实现:是 |

## 4. 原语(名词)

### 4.1 玩具分布(Distribution)

| 原语 | 参数(默认) | 说明 |
|---|---|---|
| `gaussian(mean, cov=I)` | | 各向同性写 `gaussian([2,0], 0.5)` |
| `uniform(lo, hi)` | | 盒内均匀 |
| `ring(r=1, noise=0.1)` / `spiral(turns=2, noise=0.1)` / `moons(noise=0.1)` / `checkerboard(k=4)` | | |

全部可精确采样且有解析 score。变换(惰性):`shift`、`scale`、`rotate`。

**`unnormalized(L)`**(原预留词 unnormalized_density,已实现):用坐标符号
`x1`、`x2` 书写对数密度 L(可差一个常数)。能力标记 = **有 score、不可
精确采样**——score = ∇L,配分函数在求导中消失(SVGD 用于贝叶斯推断的
根本原因);`~` 无 via 时静态拒绝:"不可精确采样,请用 via 指定推断算法"。

### 4.2 数据集(Dataset)

`mnist`、`text8`/`enwik8`、`labeled(...)`、`split(data, 0.9)`,同 v0.2。
数据只有两个来源:内置数据集和分布采样(沙箱边界,见 §6.1)。

### 4.3 网络骨架(Net)

`mlp(d0 -> ... -> dk, act=relu)`、`cnn(28x28 -> 10)`、
`transformer(dim=, blocks=, heads=, context=)`,同 v0.2。

### 4.4 判别/预测式范式(Trainable)

`classifier(net)`、`regressor(net)`、`lm(net)`、`masked_diffusion(net)`,
同 v0.2(文本生成走 `generate`)。

### 4.5 散度(Divergence)

**Divergence = 分布对上的泛函**,f-散度只是其中一族。第一槽 target(静端),
第二槽动端;**两槽各接受 Distribution 或 Dataset**("样本与分布一体"),
**第二槽可省略**(动端由 `#` 操作数的出身推断;写全则更早报错)。

| 族 | 内置 | WGField 构造法 | estimator 菜单 |
|---|---|---|---|
| f-散度 | `forwardKL`、`reverseKL`、`pearson_chi2`、`neyman_chi2` | h 变换:h(r)=r f′(r)−f(r)(Liu et al. 2024, Thm 2.1) | `nw`、`locallinear`、`ratio_then_grad` |
| OT | `W2` | 解离散 OT 取位移(Brenier 方向) | `sinkhorn(eps=)`、`hungarian` |
| (预留)核 | `mmd(kernel=)` | 核见证函数梯度 | — |

- `mirror(D)`:镜像散度(如 `mirror(forwardKL) ≡ pearson_chi2`);
  估 D 的场 = 最大化 mirror(D) 的变分下界(同文 §4)。
- `estimate(D)`:样本估计(f-散度用变分下界;W2 用 Sinkhorn)。
  分布间距离度量不另设动词,由此吸收。

### 4.6 estimator(WGField 的统计近似槽)

WGF 的场写得出、算不出(密度比未知),估计方法是一等选项,
**需求逐槽静态核对**:

| estimator | 方法 | 前提 |
|---|---|---|
| `nw(kernel=rbf)` | Nadaraya–Watson 插值,统一形式 smooth(∇log p)/Z_p − smooth(∇log q)/Z_q | target 有 score(≈SVGD,同文式 5)**或只有样本**(2026-07 修订:p 项退化为 target 样本的 KDE score,两侧同核同带宽,p̂=q̂ 处偏差精确对消;unnormalized 即 MMD 流场) |
| `locallinear(kernel=rbf)` | 局部线性(镜像散度变分目标) | **只要两端样本**(target 可为 Dataset) |
| `ratio_then_grad` | 先估比率再求导 | 只要样本;注:过拟合风险(同文图 1) |
| `denoiser(net)` | DSM 拟合时间反演所需的各时刻 score | 只要样本;专用于 `reverse` |
| `empirical` | 经验混合 score 的闭式精确计算(q_t = 数据 ⊛ 高斯是 n 分量高斯混合) | 只要样本;免训练;专用于 `reverse`。**教学件**:精确 = 完美复读训练集(记忆);denoiser 的平滑才带来泛化 |
| `sinkhorn(eps=)` / `hungarian` | 离散 OT | 只要样本;W2 族专用 |

由此立起正交分解:**`estimator=` 近似场本身(统计误差,随样本量/带宽变),
`steps=`/`solver=` 近似积分(离散化误差)**。两个槽、两种误差,
可各自控制变量做对照实验。

(v0.2 的解析场原语 `stein` 已退役——它是 `WGField(reverseKL(·), estimator=nw)`
的定理推论,见 §3.3;`langevin` 只以 via 糖形式存在。)

## 5. 动词(五个)

`seed` / `train`(仅 Trainable;默认 Adam lr=1e-4、batch=128)/
`generate`(文本条件生成)/ `eval` / `plot`(含 `plot trajectory of <事件>`、
`plot loss of X`),语义同 v0.2。`transport`、`WGField`、`WGFlow`、`reverse`
是表达式而非动词——它们出现在绑定右侧。

## 6. 静态检查(报错即教学)

| 违例 | 报错要点 |
|---|---|
| `classifier ~ 100` | 判别式模型不是分布 |
| `posterior ~ n`(不可精确采样、无 via) | 请用 via 指定推断算法 |
| `flow_dist ~ n via svgd` | svgd 需要 score,该分布只提供采样 |
| `field(descent(reverseKL(p)), nw)`(p 无 score 又无样本) | nw 需要 score 或样本;有采样器的分布请先 `p ~ n` 落成 Dataset(2026-07 起 Dataset target 合法:KDE score 差) |
| `inv(WGFlow(...))` | inv 只作用于 Map;测度演化无逐点逆(要倒放请 reverse,花训练买) |
| `WGFlow(...) # X`(X 出身 ≠ 动端) | 硬错误,见 §3.2 |
| `flow(WGField(...))` / `WGFlow(v)`(v 是 Field) | 场与积分器几何不匹配 |
| `train v on data`(v 是 Field) | train 只作用于 Trainable;学场请用 transport |
| 维度不匹配 | `expected input dim 784, got 2` |
| 资源超限 | 见 §6.1,提交时拒绝 |

### 6.1 沙箱与可复现性(语言级保证)

- **无 IO 原语**;数据只来自内置数据集与分布采样(永久排除项)。
- **不图灵完备**:程序长度有限 ⇒ 执行步数有上界。
- **资源限额**(执行前静态核算):Dataset ≤ 100k 样本;单矩阵 ≤ 10^7 元素;
  总训练/积分 steps ≤ 50k;墙钟 ≤ 60 s(超时软停)。
- **可复现**:程序文本 + seed 完全决定结果;结果页附程序与 seed,一键重跑。

## 7. 扩展机制(逃生舱)

语言不长大,宿主长大。注册对象七类:

```cpp
LIU_REGISTER_DISTRIBUTION("two_rings", ...);        // 声明能力:可采样?有 score?
LIU_REGISTER_TRAINABLE("vae", ...);
LIU_REGISTER_DIVERGENCE_FAMILY("f_div",  field_via_h,       {nw, locallinear, ratio_then_grad});
LIU_REGISTER_DIVERGENCE_FAMILY("ot",     field_via_ot_plan, {sinkhorn, hungarian});
LIU_REGISTER_DIVERGENCE("kl", family="f_div", f = ...);   // 给 f,自动导出 h
LIU_REGISTER_ESTIMATOR("locallinear", requires = samples_only, ...);
LIU_REGISTER_SOLVER("heun", ...);
```

## 8. 示例程序

### 8.1 flow matching 主示例(v0.2 保留:transport + pullback)

```liu
seed 42

noise = gaussian([0, 0], 1)
data  = moons(0.05) ~ 1000

v = transport from noise to data using mlp(2 -> 64 -> 64 -> 2)   // Field
T = flow(v, steps=50)                                             // Map

plot data, (T # noise) ~ 1000              // T#μ0 ≈ μ1 ?
plot (flow(v, steps=1) # noise) ~ 1000     // 一步采样 vs 50 步
plot (inv(T) # data) ~ 1000, noise ~ 1000  // pullback:拉回去像不像高斯
plot trajectory of (T # noise) ~ 200
```

### 8.2 WGF 主示例(target 只有样本的 domain-adaptation 情形)

```liu
seed 42

p = moons(0.05) ~ 1000                 // target:只有样本,无密度
q = gaussian([0, 0], 1)

mydiv = reverseKL(p, q)
u = WGField(mydiv, estimator=locallinear(kernel=rbf))
T = WGFlow(u, steps=10, lr=0.1)

sample_2 = T # (q ~ 500)
plot p, sample_2
plot trajectory of sample_2
estimate(mydiv) |> table               // 流完后散度剩多少
```

### 8.3 OT flow 一课(三幕;第四幕 OT 耦合的 flow matching 待 v0.4 `coupling=`)

```liu
seed 42

src = gaussian([-2, 0], 0.3)
tgt = 0.5*gaussian([2, 1], 0.3) | 0.5*gaussian([2, -1], 0.3)
X0  = src ~ 300

// 一:静态 OT —— 两团点云的运输代价
plot X0, tgt ~ 300
estimate(W2(tgt, src)) |> table

// 二:W2 的 WGF = McCann 位移插值,粒子走直线
u = WGField(W2(tgt, src), estimator=sinkhorn(eps=0.05))
plot trajectory of (WGFlow(u, steps=20) # X0)

// 三:对照 —— 独立耦合的 flow matching,轨迹是弯的
v1 = transport from src to tgt using mlp(2 -> 64 -> 64 -> 2)
plot trajectory of (flow(v1) # src) ~ 300
estimate(W2(tgt, flow(v1) # src)) |> table
```

### 8.4 diffusion:前向梯度流 + 学习的时间反演

```liu
seed 42

data  = moons(0.05) ~ 2000
noise = gaussian([0, 0], 1)

// 前向:朝高斯的 reverseKL 梯度流(熵项由注噪精确实现,前向无需估计)
fwd = WGFlow( WGField( reverseKL(noise, data) ), time=3 )

// 反演:花训练买。DSM 拟合各时刻 score → probability-flow ODE 的速度场
v_rev = reverse(fwd, estimator=denoiser(mlp(3 -> 64 -> 64 -> 2)))   // Field
T     = flow(v_rev, steps=50)                                        // Map

plot data, (T # noise) ~ 500
plot trajectory of (T # noise) ~ 200
Z = (inv(T) # data) ~ 500        // DDIM 反演/编码器:ODE 实现的 T 是真 Map,白送

// 记忆 vs 泛化,一行之差(§4.6 empirical):
v_mem = reverse(fwd, estimator=empirical)                  // 免费:完美复读训练集
plot (flow(v_mem) # noise) ~ 500, (T # noise) ~ 500        // 复读机 vs 生成模型
```

### 8.5 SVGD 双写(意图层糖 vs 机制层全展开)

```liu
seed 1

target = 0.7*gaussian([-2, 0], 0.3) | 0.3*gaussian([2, 0], 0.3)

Y = target ~ 200 via svgd(kernel=rbf, steps=500)                  // 糖
Z = WGFlow( WGField( reverseKL(target), estimator=nw(kernel=rbf) ),
            steps=500 ) # (gaussian([0,0], 1) ~ 200)              // 机制
plot trajectory of Y
```

### 8.6 MNIST 分类 / 字符 LM(同 v0.2)

```liu
seed 0
net = classifier(mlp(784 -> 128 -> 32 -> 10))
train net on mnist.train for 2000 steps with batch=64
eval net on mnist.test |> table
```

```liu
seed 7
m = lm(transformer(dim=64, blocks=2, context=64))
train m on text8 for 10000 steps
generate m from "the meaning of life is " ~ 200 chars at temperature 0.8
```

## 9. 执行模型(实现备忘)

- 递归下降 parser → 静态检查(名字、能力、出身、形状、资源)→ tree-walking
  解释器,每个原语映射到一段 Juzhen 代码。
- Field 是"(t, X) ↦ 速度矩阵"的闭包;Map 是"Field + 积分方案"的惰性组合,
  `inv` 翻转积分方向。WGField 持有散度 + estimator,每步从当前粒子重估场;
  WGFlow 持有 WGField + 步数/时间。`reverse` 触发一次训练(DSM),产出闭包
  形式的时变 Field。`#` 触发实际计算。
- WGF 每步的核心计算(核矩阵、pairwise 距离、局部线性求解)全部是
  GEMM + 逐元素运算,落在 Juzhen 甜区;SVGD 参考实现约 15 行(见讨论记录)。
- v0.3 明确不做:`coupling=`(OT-CFM/reflow,随耦合一等化留 v0.4)、
  SDE 实现与第三对场/流(Markov 核)、条件生成 `given`、密度对象与似然、
  save/load、优化器选择、自定义层。

**理论出处**:WGF 场的 h 变换(Thm 2.1)、NW 估计与 SVGD 的等价(式 5)、
镜像散度与变分下界(§4)——Liu, Yu, Simons, Yi & Beaumont,
*Minimizing f-Divergences by Interpolating Velocity Fields*, ICML 2024。
前向扩散 = KL[·‖N(0,I)] 的 Wasserstein 梯度流:Otto 演算 / JKO;
时间反演:McKean;概率流 ODE:Song et al. 2021。

---

## 10. 路线图(非规范性,记录既定方向)

以下内容已经过讨论定向,但**不属于 v0.3 规范**;列于此处防止散佚,
留待解释器原型的实现反馈校准后正式化。其中 10.2/10.2.1(概率路径与
统一提取子)、10.3.1(场代数)与 10.3.2(残差场估计器)**已在参考
解释器落地**,保留原文作为设计记录。

版本切块(2026-07 讨论定案):**v0.4 = 耦合**(纯连续,不动类型系统,
见 10.1/10.3/10.4);**v0.5 = 随机核**(SDE 与离散流一起进来,新增
Kernel 类型与时间反演机制,见 10.5)。正交小项(mnist 数据门 +
`plot images`、教学动词 `explain`)可随时插队,不占版本号。

### 10.1 受限 `for`(有界迭代;已在参考解释器落地,2026-07——含 reflow 所需的 `via=paired` 耦合)

薛定谔桥(IPF)、reflow、退火等真实算法是迭代式的。放进语言的是
迭代,不放进来的是控制流:

> `for k in 1..K` 合法,当且仅当 K 是**字面量整数**(上限如 64);
> 无 `while`、无 `break`、无条件语句。**语义定义为宏展开**:循环体复制
> K 份(k 代入常量),得到的仍是有限 DAG——静态检查器看到展开后的完整
> 程序,资源核算为 K 份之和,四条北极星原样成立。体内重绑定按 SSA 展开
> (`v = f(v)` 每轮产生新版本,出身链完整)。计算能力为有界原始递归,
> 距图灵完备甚远。

| 允许 | 禁止(永久) |
|---|---|
| `for k in 1..K`,K 字面量 | `while`、`break`、`continue` |
| 体内重绑定(SSA 展开) | 数据依赖的循环边界 |
| k 作为常量参与表达式/标签 | 条件语句、提前退出 |
| 嵌套(展开后仍受 §6.1 限额) | 递归、自定义函数 |

`for` 的价值不在省打字,在**迭代可观测**:每一轮的中间对象都是一等值,
可逐轮 plot/estimate(对比 `solver=ipf(rounds=)` 的黑盒)。报错文案:
"`while` 不存在——程序必须在运行前就知道自己会做什么;数据依赖的循环
请回到宿主层。"

### 10.2 Probability Path:概率路径作为第一公民(已定稿设计)

三大范式的公共祖先不是散度,是**曲线 + 场**。声明式曲线由**用户手写的
插值公式**给出——这是 v0.4 讨论中定稿的核心设计:

```liu
noise = gaussian([0, 0], 1)
data  = moons(0.08) ~ 1000

xt = t*data + (1-t)*noise        // 插值公式;t 是保留的时间符号
pt = prob(xt)                     // 取律:t ↦ Law(x_t),得到 ProbPath
v  = field(pt, estimator=regress(mlp(2 -> 64 -> 64 -> 2)))
T  = flow(v, steps=50)            // flow(field(prob(xt))):流、场、律,各是其名
```

**随机变量层**。插值公式是 RV 值的仿射表达式 Σ cᵢ(t)·ξᵢ(系数 cᵢ 为
t 的初等函数)。关键语义规则:

- **分布没有身份,随机变量有**。`rv(D)` 声明一次抽样;重复使用同一个
  名字 = 同一次抽样,两次 `rv(D)` = 独立两份。X−X 在变量读法下为 0,
  在独立读法下是方差加倍的卷积——两者必须可区分。
- **单次出现自动提升**:公式中只出现一次的 Distribution/Dataset 自动
  提升为匿名独立 RV(常见路径零仪式感)。
- **重复出现硬报错**(报错即教学):同一分布出现两次时拒绝,要求 `rv()`
  显式声明是同一抽样还是独立两份。
- **`prob()` 必写**:它是"随机变量世界 → 分布世界"的关卡,取律动作
  在程序里可见。
- 各 RV 默认独立;`couple(x0, x1, via=...)`(10.3)修改联合律——
  **路径 = 边际 + 耦合**,RV 的身份正是耦合得以附着的名字。

**`field(pt, estimator=)` 的数学定义**:条件期望场

  v(x, t) = E[ ẋ_t | x_t = x ],   ẋ_t = Σ cᵢ′(t)·ξᵢ

三个等价刻画:(a) ẋ_t 对 σ(x_t) 的 L² 投影;(b) 满足连续性方程
∂ₜp_t + ∇·(p_t v) = 0,故 `flow(v)` 输运出的边际恰为 p_t(管线的正当性);
(c) 回归 min_w E‖w(x_t,t) − ẋ_t‖² 的唯一最小点,故 `estimator=regress(net)`
的损失最小点就是定义本身——FM 与 DSM 是同一回归的特例,实现层合并为
一个训练例程。

**"解方程 + 选解"的表述**。连续性方程对给定路径是欠定的(解差一个对
p_t 无散度的分量),`field` 严格说做两件事:解连续性方程,再按某个
原则从解族中**选一个**——声明式路径按**耦合**选(E[ẋ|x] 是该过程诱导的
代表;换耦合,同边际,不同场)。方程只有一个,各情形变的是选择原则与
计算把手:

| 情形 | 场的表达式 | 闭式程度 | 计算把手 |
|---|---|---|---|
| 端点皆高斯 | 仿射 v(x,t) | 完全闭式 | 直接算 |
| 数据端为经验测度 | softmax 加权的训练点平均 | 闭式但只会复读(精确 = 记忆) | `regress` 是对它的平滑 |
| WGF 路径(f-散度) | ∇(h∘r_t),h 闭式、r_t 未知 | 半闭式 | `nw`/`locallinear` |
| W2 | Brenier 位移 | 离散情形精确可解 | `sinkhorn`/`hungarian` |

**退役与降级**(词变定理第四次上演):

- `interpolate(from, to, schedule=)` 构造子退役——公式比它更一般且可见;
- `transport from A to B using net` 退役为
  `flow-ready 糖 ≡ field(prob(t*B + (1-t)*A), estimator=regress(net))`;
- `reverse`/`denoiser` **降级为隐式路径专用**:声明式路径按生成方向直接
  书写(时间反演 = 公式里 t ↦ 1−t,用户自己写),其边际按构造精确可采样
  ——v0.3 的"前向边际须有闭式"特判随之消失;`reverse` 仅对无公式可改的
  隐式路径(descent/bridge)保留意义。

隐式曲线构造子不变:

| 构造子 | 曲线从哪来 | 问题类型 |
|---|---|---|
| 插值公式 + `prob()` | **声明**:用户手写 | — |
| `descent(D, from=, time=, metric=w2, family=free)` | **生成**:泛函最速下降 | 初值问题(IVP);family= 选子流形(rotation = 旋转轨道,2026-07) |
| `bridge(from, to, reference=, solver=)` | **求解**:两端钉死,变分选最优中段 | 边值问题(BVP) |

验证这层抽象的两个事实:diffusion 前向既是 KL 梯度流(隐式)也是
`sqrt(1-t*t)*e + t*x1` 型的显式公式,两种出身并存;OT 课的第二幕
(W2 的 WGF)与 OT 耦合的插值公式是同一条曲线,两个构造子在 OT 处汇合。

### 10.2.1 最终统一:一个提取子(WGField/WGFlow 的退役计划)

WGField 的参数是散度而非路径,与"field 提取路径的场"的世界观不协调。
修正:散度诱导的不是场,而是先诱导**路径**——W2 几何里朝 min 走的
最速下降曲线(注意:不是静态的 argmin,是朝它走的动力学):

```liu
qt = descent(reverseKL(p), from=q0, time=3)   // 隐式路径:最速下降的 IVP
v  = field(qt, estimator=nw(kernel=rbf))       // 同一个提取子
T  = flow(v, steps=...)                         // 同一条管线
```

于是 **`WGField(D)` ≡ `field(descent(D))`**,复合词退役(词变定理第五次)。

**命名决议:`descent`(弃 WGpath/steepestpath)**,并携带 `metric=` 槽
(缺省 `w2`):最速下降永远相对某个几何而言,这个自由度应当显式、可教学——
`metric=w2` 即文献中的 Wasserstein 梯度流;`metric=stein`(核化 Stein 几何)
下 SVGD 是**精确的**梯度流,而在 w2 读法下 nw 只是其核平滑**估计**
(同一更新式的双重身份,Q. Liu 2017 / Liu et al. 2024);
`metric=fisher_rao`(birth-death 动力学)预留。缩写词(WG*)不再进入词汇表。
两类路径的"场从哪来"恰好互为镜像:

| | 声明式路径 | WGF 路径 |
|---|---|---|
| 谁先给定 | **路径**先给(公式/耦合) | **场**先给(−∇(δD/δq),h 变换) |
| 另一半怎么来 | 场 = 解连续性方程,耦合选解 | 路径 = 场的积分(IVP) |
| 选择原则 | 耦合诱导的代表 | **最小动能/梯度形代表**(Otto 切空间正典元) |

末行是一个恰到好处的巧合:最速下降场天然是梯度形的,恰好是连续性方程
解族中范数最小的正典代表——声明式按耦合选,梯度流按正典选。

**类型系统的连锁收编**:v0.3 的测度空间对 WGField/WGFlow 在 v0.4 整体
折叠进 Path/field/flow 主管线——`descent(D, from=q0)` 把初始测度钉进
路径(出身硬检查改挂在 `from=` 上),提出的场是诚实的时变 Field,
`flow` 给出只对该出身有效的 Map。SVGD 实现中"逐步以当前粒子重估场"
的交织,是 `nw` estimator 的实现策略,藏在抽象之下。v0.3 的
"两对场/流"平行表格由此收敛为**一对 + 三种路径构造子**。

### 10.3 耦合一等化(v0.4 主轴;已在参考解释器落地,2026-07)

**统一命题:路径 = 耦合 × 条件桥。** 任何端点联合分布 q(x₀, x₁)
(耦合)加任何条件插值 p_t(x|x₀, x₁)(条件桥)都给出一条合法概率路径,
`field` 的回归定理原样适用(CFM 文献的一般形式,Tong et al. 一线)。
v0.3 的声明式路径把独立耦合硬编码成了唯一选择——单次出现自动提升
= 强制独立;v0.4 把这个槽打开。三个语言增量,从小到大:

1. **`#` 作用到 RV**(零新词)。`T # e` 对随机变量逐点推前、保持抽样
   身份。v0.3 的"重复出现 = 同一次抽样"规则由此获得表达力:同一个
   `e` 出现两次,天然就是**图耦合** (e, T#e)。Rectified flow 不需要
   任何新的耦合原语:

   ```liu
   e  = rv(noise)
   xt = t*(T1 # e) + (1-t)*e     // 耦合 = T1 的图;同一 e,同一次抽样
   ```

   与 10.1 合用即得 reflow 迭代(逐轮 `plot trajectory` 看轨迹变直,
   海报级 demo)。**建议的 v0.4 起点**:最小、免费、立即可演示。

2. **`couple(μ, ν, plan=...)`**,返回带**联合**抽样身份的 RV 对:
   `(x0, x1) = rv(couple(noise, data, plan=sinkhorn(0.1)))`。
   plan 槽即价目表:`independent`(默认,= 现状)、`ot` / `sinkhorn(eps)`
   (解一个离散 OT,minibatch 近似);图耦合由第 1 条免费给出,不进
   此处。rv() 的概念从"给分布一个身份"延伸为"给一对分布一个联合
   身份",与 v0.3 完全连续。

3. **受限 `for`**(10.1),reflow 与 IPF 的宿主。

**落地修订(2026-07)**:plan 槽定名 `via=`(与 `~ n via svgd` 的
"怎么做"槽同名同义);图耦合没有走第 1 条的 `T # e` 路线,而是落成
`couple(base, T # base, via=paired)`——理由:`T # e`(RV 逐点推前)
需要 Map 作用于 RV 世界的新公理,而 `via=paired` 把图耦合归位为
couple 的一个 plan,零新公理;reflow 由它 + 有界 `for` 完整落地
(`reflow.liu`)。另有 eager 模式:解构绑定尾随 `~ n` 冻结一次 n 对
的耦合(minibatch 偏差 vs 冻结样本偏差,一个 `~` 之差)。

**注记:耦合与分解——同一联合律的两个公理化方向(2026-07,对话中
提炼)。** `linear_gaussian`/`sine_gaussian`(10.8)与 `couple`(本节)
构造的是**同一种对象**——对上的联合律;区别只在把什么当公理:

| | 边缘 | 条件律 p(y\|x) |
|---|---|---|
| `couple(A, B, via=...)`:**边缘为公理** | 精确(就是输入)| 推论——隐含在 plan 里,只以有限样本存在 |
| `linear_gaussian` 等:**机制为公理**(disintegration) | 推论——复合分布 ∫p(y\|x)p(x)dx,无闭式 | 精确(生成机制本身) |

语言里的能力标记恰是这个对偶的投影,三条各自对应"公理可提取、
推论不可提取":

- `prob = Law(·)`(2026-07 收窄):couple 块可取边缘(公理),
  构造式联合的块报教学错(推论无闭式);
- `has_cond_sampler`:构造式联合有条件采样器(公理),couple 没有
  (推论)——10.8 的脱钩核求值因此只对构造式联合合法;
- 配对核求值只需联合样本——两族皆可(联合样本两边都是一等的)。

两个方向在极限处汇合:`via=paired` 是确定性机制(y = T(z),退化
条件律)作用于给定边缘——既是耦合又是构造式;总体极限下 `via=ot`
趋向 Monge 映射,即精确 OT 耦合渐近地就是一个确定性机制。

由此自然浮出两个未来件(未排期):从 plan 估计条件采样器(把 couple
的推论升格为公理,解锁其块的脱钩核求值);为构造式联合提供经验边缘
(`prob(y0)` 以样本作答而非报错)。

### 10.3.1 场代数与 guidance(v0.4,10.3 的姊妹条目;2026-07 定向)

Field 本是向量空间,补上它天生该有的代数:`a*v1 + b*v2` 合法,
零新词。CFG 即外插 `v = v_uncond + w*(v_cond − v_uncond)`(系数
(1+w, −w) 和为 1 但带负项——`|` 的凸组合表达不了,恰证场代数
不可替代)。类型检查三件:维度一致、时间域一致、`from=` 出身
结构相等(起点不同的组合无根基,硬报错);组合场出身标记
synthetic:`a·[v1] + b·[v2]`。

**核心定理级事实:同一个场代数,管线两侧精确性不同。**

- **descent 侧精确**:W2 梯度线性,场的线性组合 = 组合散度
  a·D1 + b·D2 的最速下降;a+b=1 时平稳分布**精确等于**几何混合
  p1^a·p2^b/Z——product-of-experts 采样的 SVGD 版。一致性检验:
  与 `unnormalized(a*L1 + b*L2)` 殊途同归,两条路线须脱糖到同一
  模拟(可执行的定理)。
- **declared 侧(CFG 所在)不精确且微妙**:高斯路径下场与 score
  逐点仿射相关,同 schedule 的仿射组合逐时刻**恰是**倾斜边际
  p_t^(1−w)·q_t^w 的概率流场;但沿组合场积分到底,终点**不是**
  p_0^(1−w)·q_0^w——倾斜与演化不交换(倾斜边际族不满足任何扩散
  的 Fokker–Planck)。"CFG 究竟采样什么"是现役研究问题
  (Bradley–Nakkiran 一线)。

价目表不变,而这正是妙处:`inv` 依然免费(ODE 可逆性不问出身;
guided flow 的 DDIM 反演实践中本就在用);10.7 的 `density=true`
依然可用——于是上述研究结论变成十行可运行的程序:对组合场的流开
密度追踪,把 CFG 实际采到的密度与 p^(1−w)q^w 并排画出。

教学件(海报级):
- w 滑块 0 → 1 → 3:样本先锐化、再过锐、最后飞出流形(CFG 经典
  artifact 的 2D 直观);旁置 descent 侧同款组合,精确落在几何
  混合上——**同一行代数,一边是定理一边是悬案**。
- 三种"平均"并排:`0.5*p or 0.5*q`(测度算术混合)、
  `unnormalized(0.5*L1+0.5*L2)`(几何混合/倾斜)、
  `flow(0.5*v1+0.5*v2) # noise`(场组合的流,declared 侧 ≠ 前两者)。

缺件:真正的 CFG 需要条件场 `field(pt | y)`——条件化机制
(label 进公式、进网络)独立成题,归 v0.5+;双模型
guidance/composition 现有词汇 + 场代数即可表达。实现代价:v0.4
中最低(Field 加两个运算符 + 出身检查)。

**存档注记(2026-07,一轮提案、一轮否决、三件归位):**

1. **路径层表述被否决,理由是契约。** 曾考虑
   `rt = unnormalized((1-w)*log(pt) + w*log(qt))`(倾斜路径),
   `field(rt)` 后台分解为"基场 + DRE 估 ratio 再求导"。否决:
   `field(path)` 的契约是返回该路径连续性方程的解,而"倾斜 score
   套 schedule 公式"**不是**倾斜路径的解(倾斜边际族一般不满足任何
   扩散的 Fokker–Planck 方程)——默认档违约的价目表是分级的谎言。
   CFG 的诚实住所就是场代数:它本是场层构造、无路径层语义,
   synthetic 标记即"对无主张的诚实主张"。
2. **`ratio_then_grad`(§4.6 预留词)归位**:制造场代数的
   **梯度场加数** ∇log(q_t/p_t)——标量势的梯度,良定义,不携带
   连续性方程主张。DRE ≡ 概率分类,故 classifier guidance =
   `field(pt) + w·b_t·grad(ratio(qt, pt))` 逐字还原 Dhariwal–Nichol
   (b_t 为 score→field 的 schedule 系数)。核模型 DRE(uLSIF/KLIEP)
   梯度有解析闭式,2D demo 不需改 Juzhen;神经 DRE 需暴露对输入的
   梯度(反向传播机制已有,改动小)。
3. **条件化定理(精确性边界的刻画)**:条件化与扩散交换
   (噪声 ⊥ y | x₀),故 p_t(·|y) ∝ p_t · p_t(y|·)——**所有条件化
   都是随演化自洽的倾斜,反之不然**。推论:classifier guidance
   w=1 精确(那就是真条件场);其 w≠1 与 CFG 的 w>1 同属倾斜悬案。
4. **倾斜路径的"真场"是另一个对象**:标价 Poisson 修正或
   Feynman–Kac 加权粒子(v0.5 Kernel)。若用户以 `field()` 索要,
   报错即教学:"你在索要倾斜路径的真场,那要解 Poisson 方程;
   若要 CFG,请写场代数——语言不假装二者相同。"

### 10.3.2 残差场估计器 `base=`(v0.4 随行,10.3.1 的估计层配套;2026-07 定向,已在参考解释器落地)

10.3.1 的双模型 guidance 目前的写法是训两个独立网络再相减:
`v_c − v_u` 是**两个独立估计的差**,各自的近似误差被 w 反向放大。
提案:给 `regress` 估计器加一个关键字,冻结一个预训练基场,只回归
修正项:

```liu
v_u = field(prob(xt_u), estimator=regress(mlp(2 -> 64 -> 64 -> 2)))
v_c = field(prob(xt_c), estimator=regress(mlp(2 -> 32 -> 32 -> 2), base=v_u))
// v_c 类型仍是 Field;内部表示为组合项 1·[v_u] + 1·[Δ],Δ 是新训的残差网络

// 默认用法(无 guidance 公式):v_c 就是条件路径的合法场,直接积分即条件采样
T = flow(v_c, steps=50)
plot p_y ~ 800, (T # noise) ~ 800

// 可选的失真旋钮(w≠1 才需要写 guidance 公式):
v = v_u + w*(v_c - v_u)                     // 按址合并后 = v_u + w·Δ;w=1 精确塌缩回 v_c
```

语义:L² 回归的最优解是逐点条件期望,故残差目标
`dx/dt − v_u(x_t, t)` 的回归极限**恰为 guidance 方向 v_c − v_u 本身**
——guidance 方向第一次成为被直接估计的对象,而不是两次估计的副产品。
`base + Δ` 是**真正的** `field(path_c)`:契约满足,非 synthetic;
synthetic 标记只在 w≠1 的缩放处出现,与 10.3.1 的诚实原则一致。

**用法分层(默认正道 vs 失真旋钮):**条件采样本身不需要 guidance
——`flow(v_c) # noise` 就是全部,类型干净,这是本条的**默认档**。
guidance 公式只为 **w≠1** 存在:实践中把 w 开到 3、5 是**故意的
失真**——越过条件场向外插值,放大条件方向,以多样性换服从度
(mode-seeking、过锐化),弥补高维下估计不完美导致的"贴条件不紧"。
残差表示让这个旋钮读起来诚实:`v_u + w·Δ` 即"基场 + w 倍修正",
旋钮只作用在修正项上,被放大的是什么一目了然。(方向提醒:推的是
源端 `noise`;`data`/`p_y` 是终点,要推终点得用 `inv(T)`,那是
编码方向。)

**修复三件:**

1. **抵消从数值近似升为表示精确。**残差场表示为组合项
   `1·[v_u] + 1·[Δ]`,场代数的按址合并(解释器已有)算出
   `v_u + w*(v_c − v_u)` 中 v_u 的净权重 = 1 + w − w = **1**——
   精确抵消;w=1 定理从"无限数据极限下成立"升为"表示层成立"。
2. **误差结构:差的估计 → 估计的差。**标准 CFG 的组合误差是
   `(1−w)·ε_u + w·ε_c`,两个独立误差被大系数反向放大(方差随 w²
   增长,即飞出流形 artifact 的统计来源之一);残差版是
   `ε_u + w·ε_Δ`:基场误差只进一次,被 w 放大的只有小得多的残差
   拟合误差——条件与无条件分布重合的区域 Δ ≈ 0,易学。
3. **出身构造保证 + 适配器经济学。**"同 schedule、同噪声端点"
   不再靠 `start_prov` 事后比对,由共享基场在构造上成立;一个基场
   摊销任意多个条件,每个条件一个小 Δ——ControlNet/adapter 的
   语言级对应。且 w 保持运行时滑块(优于 guidance distillation
   的固定 w 蒸馏)。

**修复不了的一件:**倾斜悬案与参数化无关。`v_u + w·Δ` 在 w>1 时
仍不解任何已声明路径的连续性方程;残差改进的是**估计量**(统计层),
不是**外推语义**(数学层)。10.3.1 存档注记 1 的契约论证在此同样
生效:不因构造优雅而假装契约满足。

**与注记 2 的合流:**高斯路径下 Δ ∝ b_t·∇log(q_t/p_t)——残差就是
披着回归外衣的密度比对数梯度。同一对象于是有两种估计器:
`ratio_then_grad`(判别式:DRE 再求导)与 `regress(…, base=…)`
(回归残差式:直接回归差)。一致性检验(可执行的定理):同一对
(p, q),两条路线的场应在样本充足时收敛到同一处。10.3.1 缺件里的
条件场 `field(pt | y)` 仍归 v0.5+;双模型 guidance 场景由本条用
现有词汇 + 一个关键字覆盖。

**语法归属:**`base=` 放在 `regress(...)` 内、而非 `field(...)`
顶层——它是估计策略的参数(拟合什么残差),不是路径语义的参数
(路径仍是那条路径)。置于顶层会暗示"带基场的路径"这一并不存在
的语义。

**与 10.7 补注的分工(同形不同讯号):**10.7 的
`pinned 基场 + free 修正场` 是同一个"冻结主干 + 训练修正"形状,
但由 `fit` 以**端点目标**钉住,须穿过求解器求导(adjoint,机制
未有);本条把同一形状放进**已声明路径**的 estimator 槽——训练
讯号是路径自己的回归目标,完全继承 `regress` 的现有机制,且极限
有闭式身份(恰为 v_c − v_u)。两条路在形状处合流、在讯号处分工:
路径已声明,用本条(便宜、精确);只有端点约束,才需要 10.7。

**实现备忘**(零新类型,三处改动):

1. `regress` 训练循环的回归目标由 `dx/dt` 改为
   `dx/dt − base(x_t, t)`(基场前向即可,冻结不训;base 须是
   自足场,descent 场依 10.3.1 同理拒绝);
2. 返回值直接构造为组合场 `[(1, base), (1, Δ)]`——场代数、
   w 滑块、精确抵消、`inv` 免费(Euler 积分不问场的出处)全部由
   现有机制继承;
3. provenance 记 `[v_u] + regress-residual(...)`;组合与缩放处的
   synthetic 标注沿用 10.3.1。

教学件(与 10.3.1 的 w 滑块并排):同一对 (p_all, p_y) 的三种
guidance 实现——双网络差 / 残差 `base=` / `ratio_then_grad`——
同一 w 下并排画:三者语义相同、估计器不同,artifact 的差异全部
来自估计误差的结构,一图流。

### 10.4 薛定谔桥:被 10.3 吸收为公式

布朗桥插值只是公式层的一行:

```liu
(x0, x1) = rv(couple(p0, p1, plan=sinkhorn(eps)))
xt = t*x1 + (1-t)*x0 + sqrt(t*(1-t))*sigma*z      // z 单次出现,自动提升
```

**薛定谔桥 = sinkhorn 耦合 + 布朗桥公式**——`bridge` 原语可能根本
不需要存在,词汇表不增反减。原退化谱系保留(课堂:一个 eps 滑块讲完
"生成模型一家"):

- eps → 0:桥 → OT 测地线(McCann 位移插值);
- 一端 = 参考过程的平稳分布:最优桥 = 参考过程本身,前向免学——
  **即 diffusion**(其前向"免费"的第三种解释);
- `plan=sinkhorn(eps)` 的 FM = 桥的插值子近似(SB-CFM 一线)。

动态形式(IPF 每半步 = 一次方向交替的 `reverse`)若确有需要,落位
v0.5 的随机核之上;静态形式到此已经够课堂用。

### 10.5 随机核:SDE 与离散流(v0.5 主轴)

SDE 与离散流(CTMC)是同一个断裂的两张面孔:演化随机化,`flow` 的
输出从确定性 Map 变为 **Kernel(Markov 核)**——可推(`K # μ` 惰性
成立)、**无逐点逆**。两者共享新类型与时间反演机制,故捆绑为 v0.5。

**离散流的同构表**(同一条定理,两个半群):

| 连续 | 离散(有限状态空间,如 V^L) |
|---|---|
| 路径 p_t ∈ P(ℝ^d) | 路径 p_t ∈ P(X),无障碍 |
| 连续性方程 ∂p = −∇·(vp) | 主方程 ∂p = Qₜᵀp |
| 速度场 v | 速率矩阵(生成元)Q |
| 边际场 = 条件场的条件期望 | **同一条定理**:边际速率 = 条件速率的条件期望(离散 FM 核心引理,Campbell / Gat et al.) |
| estimator=regress(L² 回归) | estimator=posterior(交叉熵回归 p(x₁\|x_t),速率由 schedule 闭式给出——denoiser 的离散同位素) |
| flow = Euler 积分 | flow = CTMC 模拟(τ-leaping) |

设计要点:

- **公式层换代数**:token 无值算术,有的是**律的凸组合**——恰好是
  已有的 `|`。逐坐标 `xt = (1-t)*x0 | t*x1` 即离散 FM 标准路径;
  `noise = delta(MASK)`(吸收态)时**即 masked diffusion**——v0.2
  遗留的 `masked_diffusion` 原语被脱糖吸收,词汇表再净减一词。
  教学点:律 mixture 路径在连续情形合法却无人用(瞬移而非运输),
  离散情形却是唯一自然选择——为什么,本身是一节课。
- **命名**:倾向 `rate`(输入是速率矩阵 Q;`transition` 易与转移
  概率矩阵 P_t 混淆——P_t 是 flow 的**输出**,混用会把生成元与半群
  混为一谈)。反方立场存档:定理只有一条,可让 `field` 做伞词按
  几何分派;但可逆性定理确实不同,词应把差异钉出来,故分名。
- **inv 价目表长出第四行**:离散流无概率流 ODE 技巧(有限集上的
  确定性映射只能整原子搬质量),逐点逆是**定理级不可能**;存在的是
  CTMC 时间反演,反向速率需要概率比 p_t(y)/p_t(x)(score 的离散
  类比)。即:*免费/花内存两档在此不可用,inv = 训练一个 ratio*
  ——新的"报错即教学"。
- SDE 侧原有条目不变:`reverse` 的 `realization=sde`(祖先采样)、
  `langevin` 的严格形式化落位于此;v_SDE = v_ODE + (σ²/2)∇log p_t
  (Fokker–Planck 保边际)是"同边际、不同轨迹"的一图流教学点。
- 实现舒适区:pmf/logits 都是矩阵;text8 小词表(V=27)或 2D 格点
  玩具(直接复用散点渲染)在 CPU 上秒级,支柱不破。

### 10.6 摊销流 `into=`:把 descent 摊销为一步生成器(drifting;2026-07 定向,已在参考解释器落地)

对 "Generative Modeling via Drifting"(Deng, Li, Li, Du & He,
arXiv:2602.04770)的充分性测验结果:**可容纳,且分解干净**——场的
部分是已有公理,新公理只有一个正交的轴。

论文机制:训练一步生成器 G(q = G#noise);drifting field
V_{p,q}(x) = V⁺_p(x) − V⁻_q(x),softmax 核加权 mean-shift 的
吸引(拉向数据 p)与排斥(推离自身 q);反对称 ⇒ q = p 时 V ≡ 0。
训练循环:x = G(z),loss = MSE(x, stopgrad(x + εV(x)));推理单次前向。

分解(落地版,2026-07):

- **场 = 已有公理**。"吸引向 p + 排斥离 q + 平衡态消失"正是散度
  最速下降场的形状;落地时不需要单设 `meanshift` estimator——它
  就是 `nw` 对 Dataset target 的 KDE 退化(统一形式
  smooth(∇log p)/Z_p − smooth(∇log q)/Z_q,本轮已落地;论文的
  softmax 归一 ≈ normalize=true 的查询侧核质量归一,另有一次
  数据侧 softmax 是工程强化,非 KDE score 语义,不收)。
- **新轴:积分的第三种解法**。语言早有正交分解"estimator= 近似场
  (统计误差),steps=/solver 近似积分(离散化误差)";`record=`
  已经证明 flow 可以换产物表示。`into=` 是同一根轴的第三档:
  **模拟(粒子)/ 录史(内存)/ 摊销(优化器)**。每步优化:
  z ~ from=,y = net(z),回归 net(z) → stopgrad(y + ε·Φ(y))——
  同一个 Euler 步,施加在生成器自己的 pushforward 上;流式 loss
  恒 = ε²·mean‖Φ‖²,即 10.8 的零流判据(平衡态自动停机)。

  ```liu
  qt = descent(reverseKL(data), from=noise)        // data:只有样本
  v  = field(qt, estimator=nw(kernel=rbf))
  G  = flow(v, into=mlp(2 -> 128 -> 128 -> 2), steps=4000, lr=0.5, batch=256)
  plot data, (G # noise) ~ 800                     // 单次前向生成(1-NFE)
  ```

- **定名修订(2026-07,落地当日)**:载体参数由暂定词 `carrier=`
  改为 **`into=`**("把场积分进一个网络";与 `from=` 成对)。理由:
  carrier 在测度论旧文献里是 support 的同义词(carrier of a measure),
  语义撞车;ML 里也无此用法。`lr=` 沿用 flow 的粒子步长语义 = 漂移
  步长 ε;网络优化器步长走 `trainlr=`(默认 1e-3);`steps=` 在
  into= 下指优化步数(默认 2000,而非模拟的 50)。net 无时间输入
  (演化在训练里,不在网络里),维度 d -> ... -> d 静态核对。
- **价目表**:G 出身**硬钉** `from=`(权重只见过 from= 的隐变量,
  异源即出分布外,且无 record= 逃生门——权重本身就是"录史",只为
  from= 作答);**不可逆**(整条 Euler 链耗散进权重,无轨迹可倒放,
  网络也不必单射),inv 教学报错指路 record=true / reverse;
  `record=true` 与 `into=` 互斥(两种买法二选一)。密度同样不可得
  (对照 10.7 的 neural ODE:恰好互补——一步推理 vs 可算密度)。
- **诚实注记(实测,2026-07)**:摊销是 mode-seeking 的。目标模态
  分离良好时(±2 双高斯),net 初始 pushforward 是一团、随参数
  连续地整体迁移,全部落入重模态;同一个场的**粒子**模拟(各粒子
  独立滚下最近的盆)双模态全覆盖。moons 这类支撑连通的目标无此问题
  (`examples/drifting.liu`)。论文在大尺度上靠特征空间核 + 大 batch
  缓解;2D 诚实版把失败暴露干净,是第四张范式卡片的一半教学价值。
- **谱系注记**:两时钟理想版(每步 argmin 拟合再走 ε)= iNGD
  Algorithm 1(Liu, Wang & Wang 2025, arXiv:2502.07650——早
  drifting 一年);drifting = 内外环合并(每步一次梯度,即"让优化器
  隐式做 NTK 投影",对应 iNGD 的 ntKiNG);KiNG(闭式核 drift、
  Fisher 预条件)是未来 `estimator=king(T=)` + `metric=ngd(T=)`
  的候选——`metric=` 槽由此获得第二个真实成员(流形引导的自然梯度),
  `T=` 是先验概率模型之门(与 10.7 的 `pretrained()` 补注碰头)。
  同一机制的第三个实例已落地(2026-07):`descent(..., family=rotation)`
  ——约束声明在**路径**上(受约束曲线是另一条曲线,与 metric= 同为
  曲线几何的正交轴:metric 选环境几何,family 选子流形),场的投影
  与 flow 的指数步都是它的定理:速度场投影到 so(d) 切空间
  (M = (1/n)Σᵢ v(zᵢ)zᵢᵀ,Ω = (M−Mᵀ)/2,z ← exp(lr·Ω)z,Lie 群
  积分器——欧拉步会因 Ωz ⊥ z 逐步吹胀点云而离开轨道),流在旋转
  轨道上走——白化后
  (`whiten(X)`,确定性 PCA 标准化)这正是 Amari 自然梯度 ICA 的
  连续时间形式,作为受约束的 W2 梯度流表达(KiNG→指数族、
  ntKiNG→NTK 函数类、rotation→SO(d):三次都是"把下降方向投影到
  模型流形切空间")。约束买回可识别性:自由的趋独立流可达任何
  独立边缘构形(Darmois 构造),旋转流只搜线性 ICA 模型类,
  非高斯源下答案唯一(至排列与符号)。实现注记:投影把 n·d 个场值
  平均成 d(d−1)/2 个数,方差便宜,故即使归一化场也用 Liu & Wang
  尖锐带宽——宽带宽把 KDE score 高斯化成线性场,而线性场的反对称
  矩为零(旋转信号全在高阶累积量里;Stein 恒等式下 q 项反对称部分
  精确为零)。`record=`/`into=` 教学报错:整张映射是各步旋转之积,
  一个 d×d 正交阵,无史可录、无网可摊(其逆免费 = 转置,留待用例)。
  (`examples/sica_rotation.liu`:MCC 0.77 → 1.00,8 轮;对照
  `examples/sica.liu` 自由流。)
- 早前把 consistency/蒸馏类("直接学 Map")列为片段边界疑难;
  drifting 给出 Map 进管线的正确位置——不是新的 estimator 目标,
  而是 flow 的第三种解法。蒸馏一个**已训练**的 flow(consistency
  models)是另一件事,into= 有意不做(教学报错说明)。

### 10.7 自由场:neural ODE / CNF(讨论中,2026-07)

路径的第四种来源(前三种:声明、descent、solved/bridge):**场本身
自由参数化,内部完全无约束,只用端点边际目标钉住**。

```liu
v  = field(parameter=mlp(2 -> 64 -> 64 -> 2))  // 自由场:path 槽空缺,parameter 槽宣告角色
T  = flow(v, steps=50, density=true)            // Map 携带 log-det(迹积分)
v* = fit(v, minimizing=kl(data, T # noise))     // 穿过求解器优化自由参数(adjoint)
```

语法定案(2026-07,修正自 `field(mlp(...))`):`field` 的位置参数
类型恒为 Path;路径缺席是有语义的缺席(内部无约束),缺席就让槽空着。
由此显式化一条一直默守的**设计法则:Net 只走具名槽,槽名宣告角色**
——`estimator=regress(net)`(回归器)、`into=net`(流的载体,10.6)、
`parameter=net`(纯参数化,此处)。裸 net 不指称任何测度论对象,
被槽命名后才获得语义。`field(parameter=...)` 返回带 **free(未钉)**
标记的 Field:`flow`/`#` 合法(随机初始化的流也是流),`fit` 之后
标记翻为 pinned;未 fit 即当生成模型用,检查器提醒。

三个规格要点:

- **散度方向**:MLE = min_θ KL(p_data ‖ T_θ#noise)——forward KL,
  因为 data 只有样本、模型侧必须有密度。反向不可行(data 无密度)。
- **可行性的全部定价在"模型有密度"**:瞬时变量替换
  d log p_t/dt = −tr(∇v),沿逆轨迹积分得 log-likelihood。由此
  **密度价目表**成为新的能力标记:高斯族免费;`flow` 的 Map 花一次
  迹积分(`density=true`;概率流 ODE 同样适用——diffusion 的
  likelihood 评估顺带解锁);drifting 的一步 G 与任意 net 不可得。
  10.6 与 10.7 恰好互补:一步推理 vs 可算密度,鱼与熊掌。
- **内部无约束的代价即教学点**:CNF 轨迹出名地乱;加动能正则
  ∫|v|² dt 即回到 OT 测地线(RNODE/OT-Flow 一线)——正则化槽
  `fit(..., regularize=kinetic)` 与 §8.3 OT 课呼应。

实现注:`fit` 要求穿过 ODE 求解器求导(adjoint 或直接展开),是
解释器目前没有的机制;2D 玩具可用固定小步数直接反传,秒级可行。

**补注(2026-07):free/pinned 是区分两种用场的唯一轴。**
guidance 用的已训场与 neural ODE 的待训场是**同一构造子**
`field(parameter=net)` 的两种状态:权重已定 = pinned(冻结使用),
随机初始化 = free(`fit` 翻转状态)。"已训好 vs 待训"是 Net 的
状态,不是新词。外部预训练权重经"**模型即数据**"之门进入:
`pretrained("name")`——具名工件、随解释器分发或 sha256 钉死、
权重即常量,与 `mnist` 同地位,I/O 禁令与逐位确定性均不破。

派生推论:**场代数 × free/pinned 混合 = 残差/适配器微调**。
`v = field(parameter=pretrained("base")) + field(parameter=mlp(...))`
——前者 pinned、后者 free,`fit(v, ...)` 只动 free 部分(LoRA 式
冻结主干 + 训练修正场),零新词。静态检查随之:纯 pinned 组合可
直接 `flow`(guidance);含 free 的组合须先 `fit`,否则当生成模型
用时检查器提醒(10.7 原规则的自然延伸)。

---

### 10.8 条件核与 `|`(zero-flow 条件运输;2026-07 定向,已在参考解释器落地)

**定名修订(2026-07,落地当日)**:条件算子由暂定词 `given` 改为
**`|`**——条件概率的标准记法,直接照搬;混合让位改用 **`or`**
(混合本就是概率式"或":以 w₁ 抽 A 或 w₂ 抽 B)。优先级令 `|` 紧于
`#`,于是 `t*y1|c1` 与 `T # y0|c0` 按数学直觉解析,无需括号。
同日再修订:**x1/x2 不再保留**——标识符查找先查用户绑定,坐标符号
仅在名字未绑定时生效(条件核程序因此可用 x0/x1 作变量名;遮蔽导致
unnormalized 类型错误时报错附提示)。唯一保留符号只剩 `t`。

条件分布进入语言的方式:**核(Markov kernel)成为一等对象**,
`|` 一符双职——**取律处附加**条件指标(law gate),求值时**填充**
开放的条件槽。(2026-07 再修订:条件标记从公式内迁到 `prob` 门——
公式是抽取层的插值,条件化是律层的运算,两层不混写。原行内拼法
`t*(y1|x1) + ...` 降为教学报错;`prob(yt | x1, x0)` 按**项序**声明
指标,槽序 = 项序,乱序与"指标不与任何项同抽"皆教学报错;链式
`(yt | x1) | x0` 等价。降解不变——标记仍落在项上,canon 与 RNG
流逐位不动,黄金输出零重生成。)训练机制来自 Wang, Wang, Liu & Suzuki 2026(Zero-Flow
Encoders)式 (6)(7) 与定理 3.3:独立耦合下只插值 y 块,速度网络
同时吃**两端各自的条件值**,其闭式最优
v_t(y_t; η, ξ) = E[Y₁−Y₀ | y_t, X₁=η, X₀=ξ] 的 ODE 把
p₀(·|ξ) 运到 p₁(·|η)。

```liu
(y0, x0) = rv(linear_gaussian)      // 一次联合抽样,解构成命名坐标块
(y1, x1) = rv(sine_gaussian)
yt = t*y1 + (1-t)*y0                          // 纯抽取插值,裸块
v  = field(prob(yt | x1, x0), estimator=regress(mlp(3 -> 64 -> 64 -> 1)))
Tk = flow(v, steps=40) # (y0|x0)             // 运输后的核:条件槽仍开放
plot (Tk | x0) ~ 1000                         // 配对实例化(同名 = 同一次抽样)
plot (Tk | [1.2]) ~ 300                       // 单点切片:p̂₁(·|x=1.2)
```

**类型故事**:`#` 把核映成核,`|` 把核映成分布。plot/采样一个
未实例化的核是教学错误("核是一族分布——用 | 填上条件槽")。

**配对 vs 脱钩求值(2026-07 讨论定案)**:曾误以为一切 `| W`
求值都需要条件采样器,被否决——**配对情形只要联合样本**:配对样本
里的 y⁽ⁱ⁾ 本来就是 p₀(·|z⁽ⁱ⁾) 在它自己的 z⁽ⁱ⁾ 处的合法样本,逐对
过 T(·; z, z) 即逐切片条件运输,聚合输出 = 运输后的条件分布 ⊗ 条件
边际(定理 3.3 的结构)。**只有脱钩求值**(网格、固定点
`| [1.2]`、独立分布)需要源联合的条件采样器——手里没有"在 z 处
的 y"。判别器正是既有的 **draw identity**:`|` 后是同一次抽样
的坐标块 → 配对、零要求;独立名字/新抽样 → 检查条件采样能力
(构造式玩具联合有闭式;仅有样本的联合没有,报错指回配对路线)。

规格要点:

- **联合玩具**携带坐标块结构 (dy, dx) 与"条件采样器"能力标记:
  `linear_gaussian(slope=0.5, noise=0.3)`(x~N(0,1), y=slope·x+noise·ε)、
  `sine_gaussian(freq=2, noise=0.2)`(y=sin(freq·x)+noise·ε);
- **解构绑定** `(y, x) = rv(D)`:一次联合抽样的两个命名坐标块,
  同 src——draw identity 在此第三次复用(前两次:公式重复出现、
  残差 base 出身);
- **律门层**(2026-07):指标只允许绑定与某项同一次抽样的坐标块
  (违例教学报错);带条件项与无条件项可混——`tht = t*th1 + (1-t)*noise`
  加 `prob(tht | d1)` 即标准 conditional flow matching(`|` 比 `+` 结合
  更紧,行内写法需整体括号,绑名后无此虑),10.3.1 缺件清单中的
  条件场由此闭环,无需新的 field 语法;
- **槽填充**:v1 中单参数 `| W` 同值填入全部条件槽(条件转移
  用例);按槽填充(zero-flow encoder 用例:两槽分别为 f(Y) 与 Y)
  留待后续;
- **零流判据**(定理 3.1/3.4):v_{t=0.5} 的范数是"两条件分布相等/
  条件独立"的检验统计量——未来的诊断动词候选,本条不含。

### 10.9 约束的语义家与流形之路(2026-07 定向;三家中两家已落地)

ICA 线(旋转流形)把一个一般问题逼到了台面上:**"把流/扩散约束到
流形上"这句话,约束该住在语言的哪个词上?** 答案不唯一——因为它
在数学上就不是一个对象。本节记录三个(加一个被否决的)语义家、
它们的判据,以及由此定下的流形扩散路线。

**判据(一句话):约束住在"选择真正改变对象"的地方。**

| 家 | 语法 | 对象 | 状态 |
|---|---|---|---|
| **构造约束** | `descent(D, from=, family=rotation[(block=L)])` | 曲线本身变了:IVP 落在子流形上,场 = 切空间投影、flow = 指数步皆为**路径的定理** | 已落地(2026-07) |
| **投影约束** | `project(pt, family=rotation)`(拟) | 声明桥到流形的**整体投影**:逐时刻正交 Procrustes,R_t = polar(E[x_t x₀ᵀ]),闭式;因各 t 目标独立,逐 t 最小化 = 全局最近旋转桥(pathwise L²);场自动反对称线性(v = ṘRᵀy)——**不需要 estimator**;任意 t 边际 = R_t # p₀,免费可采样(descent 路径做不到) | 数学已验证(Procrustes 迭代 0.77→1.000,~24 轮,与 regress(rotation) 同终点),动词待落地 |
| **估计约束** | `field(pt, estimator=regress(rotation))` | 曲线不动(Law 是公式说的那样),**实现它的假设类**受限:反对称线性场的最小二乘逐时间片闭式(特征基 Ω′ᵢⱼ=(M′ᵢⱼ−M′ⱼᵢ)/(λᵢ+λⱼ)),无网无 SGD;贪心速度投影 ≠ 整体最近曲线(投影与积分不交换) | 已落地(2026-07) |
| ~~读取约束~~ | ~~`prob(xt, projection=)`~~ | **被否决的形态**:`prob` 是零成本的静态门(Law(·),不采样、不耗 RNG、无超参);共享群元的投影桥需要矩估计,是有成本的构造——放进 prob 会让静态门长出 estimator 槽。该直觉指向的对象真实存在,归宿是上面的 `project` 动词 | 否决记录 |

三个合法家两两数学不同(受约束梯度流 ≠ 投影桥 ≠ 贪心场投影),
全部良定义、全部有用;`family=`/`regress(rotation)` 的不对称是
**有意义的**,不是不一致——descent 是构造动词(约束改变被构造物,
诚实),prob 是读取动词(读取不能带修饰)。

**两种"流形"问题,勿混:**

- **(a) 数据活在流形上**(球面/环面/SO(3) 帧——蛋白质、机器人):
  文献里的 manifold diffusion 指这个;路径 = **逐样本**测地/投影插值。
- **(b) 映射被约束在流形上**(数据在 ℝᵈ,运输限于 SO(d)):ICA 线
  做的这个;路径 = **全系综共享**的群元曲线。
  两者共享 family 词汇,对象不同。

**流形扩散路线(裁决,2026-07;依据为文献三大代表作的一致分解):**
Riemannian Flow Matching(Chen & Lipman,arXiv:2302.03660——测地
插值 x_t = exp_{x₀}(t·log_{x₀}x₁),premetric 框架明确允许投影弦等
非测地条件路径)、Riemannian Score-Based(De Bortoli et al.,
arXiv:2202.02763——流形布朗运动加噪、测地随机游走采样)、SE(3)
蛋白质扩散(Yim et al.,arXiv:2302.02277——IGSO(3) 内在热核噪声、
李代数上的 score)。三家同构:**约束进在路径级**(内在构造的噪声/
插值),切空间参数化与 exp 积分作为随之而来的定理;"欧氏路径 +
只在回归器投影"是外在(extrinsic)退路,插值离开流形、目标是弦、
中间边际不在 M 上,文献明确视为次选。

因此 v0.5+ 若做流形扩散:

1. **约束进路径级**,与本节判据严丝合缝;
2. (b) 用 `project(pt, family=)`(共享群元,Procrustes 闭式,上表);
   (a) 用**逐样本**插值原语——公式级 `geodesic(x0, x1)` 或
   `prob(xt, geometry=sphere)`:逐点确定性变换(如 x/‖x‖)零成本、
   不耗 RNG,**不破坏 prob 的静态门性质**——(a) 形态下 prob 带
   geometry 是合法的,被否决的只是 (b) 形态(共享群元要矩估计);
3. v0.5 的 SDE 轴与此汇合:流形噪声(IGSO(3) 式热核)同样是
   路径级构造,`realization=sde` 的流形版不需要新的约束家。

**2×2 闭合**(2026-07):{descent, transport} × {边缘, 过程} 四格全部
带旋转约束落地——`sica_rotation` / `sica_rf_rotation`(边缘,均 1.000)、
`sica_process` / `sica_rf_process`(过程,高斯墙 0.81/0.72 → 1.000)。
过程级传输的闭式:矩滞后合并到通道级(仅同滞后乘积进 C̃),同一特征基
公式解 C×C;λ 加权精确,无白化假设——descent 侧 block 投影手工近似的
一步,最小二乘自动完成。

**实验台账**(全部 quant 断言在案):descent+rotation 靠 bootstrap
扩散逃鞍(`sica_rotation` 0.77→1.000);regress(rotation) 靠 OT 耦合
的一致有符号信号确定性爬坡(`sica_rf_rotation` 0.766→1.000,48 轮,
逆映射往返实测精确 0);Procrustes 投影桥(python 验证)0.77→1.000,
~24 轮;白化法则:约束流要白化(不白化卡 0.77,旋转动不了协方差谱),
自由传输要留相关(白化后塌到 0.74,`sica_rf` 注释)。

### 10.10 可编程参数化核:逐抽取层(2026-07 定向,v0.5 先遣)

SBI 线撞到了词汇表的边界:`sbi_npse`/`sbi_svgd` 能跑,是因为玩具联合
(sine_gaussian 等)都是**出厂预制**的 C++ 采样器。流形识别论文
(Khoo, Liu & Beaumont 2026)的模型 Y = G_θ(w₀+Wx) + Π⊥ε 预制不了——
它的模拟器引用**程序运行到一半才训练出来的 G_θ**。结论:用户必须能把
一段程序注册成 Markov 核 p(y|w)。这不是加函数,是加一个语义层。

**两级语义(本条的全部内容)**:

- **测度层**(既有语言):名字指测度/映射/场/核,算术是 t-公式代数,
  Dataset 算术禁止——"测度是名词"。
- **逐抽取层**(核体内部):`kernel(w) { ... }` 的体是**对单次抽取
  逐点定义**的直线程序,按批向量化执行(名字指 d×n 的抽取块)。
  Markov 核在数学上本就是"对条件变量逐点定义"的对象——体内语言的
  存在理由。边界由类型检查器守死:体内的惯用法(抽取相乘、逐元素
  函数)一步都不许泄漏到测度层,反之亦然。

**语法与语义**:

```liu
K = kernel(th) {
    y = sin(2*th) + 0.2*gaussian([0], 1)
}
```

- 体 = 绑定序列,**末句的值是核的输出 y**;参数 `th` 在实例化时绑定
  条件块(维度届时检查)。
- 体内合法:先前绑定与参数(抽取块);数值字面量(标量);`+ − * /`
  (标量×块、同形逐元素、单行块对多行块的行广播——W·x 的形状);
  `sin/cos/exp/log/sqrt` 逐元素;分布值**自动提升为新抽取**(与
  t-公式的自动提升同一规则;同一 canon 出现两次 = 教学报错,指路
  "先绑定再复用");`T # z` 逐抽取应用训练态映射(运行时对象进入
  模拟器——本条的动机)。
- 体内非法(各配教学报错):`~`(体是"每次一抽",没有 n)、`\|`/`or`
  (体内没有测度)、descent 映射的 `#`(WG 映射是再模拟,不是函数)、
  数据依赖分支(留在未排期)。
- **非图灵完备原样保持**:体是直线程序,无递归、无条件分支。
- **体内有界 for(2026-07 补,乘积似然的需要)**:在逐抽取层,
  `for i in 1..K { ... }` 是**表达式**,其值 = 各轮末绑定的**行堆叠**
  (⊕ᵢyᵢ——N 个 iid 观测的块本来就是这个形状;外层 for 是语句,同一
  个词在两层各有层内含义,先例是 `\|` 与 `#`)。语义 = 宏展开:体复制
  K 次、`i` 为每份拷贝的标量常数、K 字面量、上限 64;分布的提升作用域
  **按轮重置**(每轮的 gaussian 是新抽取——iid 的字面意思;同轮内重复
  canon 仍报错);体内 for 不嵌套(教学报错);canon 打印循环形式而非
  展开。动机:N 观测的乘积似然 p(Y₁..N\|w) 只是"观测块从 d 行变 Nd
  行"——钉块即条件化的定理逐字照用。**实测更正(同月)**:堆叠路线
  在 KDE 估计下撞上可交换性错配(似然对槽位置换不变、逐槽距离比较
  吃 1/N! 惩罚,实测门纹丝不动),真正开门的是下文的 **KL 分解**;
  体内 for 保留为逐抽取层的堆叠原语(`kernel_probes` 探针 4)。

**实例化 = §10.8 的既有表面,零新语法**:`(K \| [w]) ~ n`(定参跑
模拟器——序贯 SBI 缺口 (b) 就地消灭)、`(K \| prior) ~ n`(联合
(y; w),y 块在前)、`(K \| X0) ~ n`(参数取自 Dataset)。产物是普通
联合 Dataset,下游(条件 descent、公式、解构)零改动消费。paired
模式(`\| 同抽块`)对可编程核无意义(它的联合本来就由实例化构造)——
教学报错。**核联合的块解构(2026-07)**:`(y, w) = (K \| z) ~ n` 冻结
**一张** n 次联合抽取的表,两个名字成为它的 (y; w) 行块视图(Dataset
携带行分界标记;`~ n` 必写——实例化是律,不是表)。这补齐了序贯候选
清单里的"块解构"小件。

**多观测条件化:KL 分解(2026-07,用户提议的定理进语言)**:
`descent(reverseKL(P), from=(q \| Obs))`——`q` 是参数空间的普通
Dataset(整体移动),`Obs` 的**列**是同一参数的 N 个观测(Dataset 或
核联合的解构块)。贝叶斯 + 对数似然可加性把目标泛函精确拆开:

```
KL[q ‖ p(·|Y₁..N)] = Σᵢ KL[q ‖ p(·|Yᵢ)] − (N−1)·KL[q ‖ prior] + 常数
```

(熵项系数 N − (N−1) = 1——恰好一份。)**装配位置是本条的全部教训**,
三次实测失败(高斯位置玩具,后验精确已知)按顺序:(一)按场求和
Σᵢvᵢ − (N−1)v₀ 携带 N+1 份 KDE 排斥项,带宽互不相消——净反扩散 +
反先验漂移,云跑到 ±12(真后验 ±1);(二)单排斥但各项用联合空间
带宽(系统性偏宽:中位数吃进 y 向散布),远场斜率不消,尾部仍炸;
(三)单排斥单带宽,不动点要乘 N 个**各自平滑过的**似然因子,要求
q̂ 方差低于核底 h/2——云塌缩(实测 sd 0.09,真值 0.24)。存活的
装配:可加性落在**对数权重**里——`log Wⱼ = Σᵢ log L̂ᵢ(wⱼ)`,每个
单观测似然是 y-核对参数行的 NW 回归(`L̂ᵢ(wⱼ) = Σₗ k_y(Yᵢ,yₗ)
k_w(wⱼ,wₗ) / Σₗ k_w`——库抽取要靠参数空间的**邻居**解释每个观测,
堆叠路线的"一次抽取解释全部"从构造上不会出现),之后每步是对
**似然加权先验 KDE** 的普通 descent:一份吸引、一份排斥、一个共享
带宽,两侧同核各平滑一次,不动点 = 加权后验本身,远场精确中和。

**似然角色的带宽自成规则**(与插值角色的 Liu & Wang / Gretton 规则
并列):库的**分辨率**定尺度,不是库的散布——(i) `k_w` 回归窗 =
k-NN 分位(k = max(8, M/100),窗内 ~k 个抽取供噪声边缘化);(ii)
`k_y` = 模拟器**自身的条件噪声尺度**(Gamma/最近邻估计:W-最近库
抽取对的 y 距离平方中位数)再乘 KDE 最优收缩率 kq^(−2/(4+d_y))
(裸 Gamma 尺度与 p(y\|w) 本身同宽,会抹平似然必须分辨的结构;
距离分位规则在稠密低维库上塌到噪声地板以下,权重退化为单抽取)。
全部确定性,每次 apply 预计算一次。守卫:reverseKL 独有(拆分是
对数的性质)、目标必须是联合样本库(y 块在前,维度核对)、无
rotation/record/into/inv,`# (q \| Obs)` 硬钉在自家云与自家观测集。
验收:`sbi_nobs.liu`(粒子云对精确高斯后验,均值/带宽两条 quant)、
`manifold_local.liu` 的**门**(32 个邻域观测识别植入切向:\|cos\|
0.815 对先验 0.637,弧速跟踪植入值;单观测墙原样保留)。

**铺开钉块 = 条件族传输,排斥项条件化(2026-07,用户"NW 场也是摊销
场"讨论收口)**:`from=(y|x)` 的钉边缘取**非退化**分布时,每个粒子钉
在**自己的**观测上——一次运行完成条件分布传输(Y-边缘逐位 = 冻结钉
抽取,逐切片 = 目标条件;`#` 返回配对云 (y, x∞))。此时吸引已逐粒子
条件化而排斥仍是混池 KDE——q 的 X-**边缘**不是切片密度,闭式对账
(linear_gaussian(0.8,0.3),切片真值 N(1.096y, 0.351²))双败:切片
内密度低估 → 塌缩(sd 0.16),中央体积把外侧切片外推 → 响应过冲
(1.89)。修复 = 用户的恒等式 ∇_x log q(x|y) = ∇_x log q(x,y) 的
估计器读法:钉行本来就在云里,联合 KDE 取自由行梯度——排斥核乘
钉通道核 k_hy(y_j, y_c)(hy = 库的似然角色尺度,两项同分辨率条件化,
远场逐切片相消)。修后斜率 1.12、切片 sd 0.35–0.38,切片均值与闭式
差 ≤0.03(`cond_family.liu`,quant 双面:界同时排除塌缩 0.16 与未
条件化边缘 0.87)。点质量钉块走原混池路径(单切片混池即条件;此门
同时保全部既有 golden 逐位不变)。摊销边界注记:NW 场逐快照可在任意
(x, y) 求值(record= 依赖的事实),但排斥含 q_t——时间轴仍由模拟
生成,族的"任意成员"仍要逐 y 付费;整族一次付清的是这里的**云形式**
(切片靠原子或窗),函数形式仍属 2a。

**复现契约**:体内抽取按语句序×出现序消耗全局 RNG;canon = 体的
AST 规范打印,捕获的外部对象以其既有 canon 入串(训练映射 = desc)。
无密度无 score:samples-only,score-needing 路径给指路报错。

**分期**:期 1 = 本条(体语言 + 实例化 + SBI 用例)——已落地;
期 2 —— **已落地(同日)**:`dot(a,b)`(逐抽取内积,投影的原料)、
`jvp(T,z,v)`(训练映射的方向导数,中心差分过两次映射应用,h=1e-3,
与路径系数导数同一纪律;线性映射下精确到舍入)、单列 Dataset 作体内
常量块(w0 = inv(G) # Y0 进模拟器的通道)——流形论文的 Def. 1 逐字
成体(`manifold_local.liu`:quant 断言两门一墙——Def.1 的贴环、
Assumption 1 的谱各向异性(径向探针的像被压缩 ~9×)、以及**单观测墙**
(一个观测几乎不识别切向,后验滞留先验 2/π 附近——论文邻域设计的
反面印证;开门的构造 = 下文 KL 分解,逐块定尺度随之落地));`kernel_probes.liu` = 语言内自检套件(dot 恒等
精确为零、jvp 与前向差分一致、用户模拟器复现解析矩,全部 quant 断言);
期 2.5 —— **已落地(2026-07,矩阵值参数)**:`rows(a, lo, hi)` 体内行
切片(0 起、半开、**字面**界——检查器静态看见每个块维度,体内 for 的
纪律)。d×m 矩阵参数以 dm-向量行走,列由 rows 切出,W·x 逐列用既有
单行广播写出:`z = w0 + rows(w,0,3)*rows(x,0,1) + rows(w,3,6)*rows(x,1,2)`
即 Def. 1 的 m=2(`manifold_torus.liu`)。**随之实测出 m=2 的分辨率墙**
(有推导):固定比例 k-NN 窗的半径 r 要 M ≥ k_q/r^d 个库抽取,d=6、
r=0.2 时 M ~ 10⁵+(M² 预计算不可行)——单位先验下门全关;先验 0.35
下部分信号(法向泄漏 0.53→0.44)但张成塌向主方向(coverage
0.31→0.24,植入 0.89);序贯轮(库从当前云重模拟,3×16 观测)只买回
边际(泄漏 0.45)——**轮数买不回窗分辨率**。quant 双面断言(信号在、
墙在)。核回归似然在此让位给训练式条件密度估计(sbi_npse 路线)或
论文的 DSM 点估计——engine 的边界有了定量坐标。
期 3 = v0.5 原列表(SDE/CTMC/时间反演)以本条的核类型为宿主。

## 附 A:EBNF 速览

```
program   = { statement } ;
statement = seed | binding | verb ;
seed      = "seed" NUMBER ;
binding   = IDENT "=" expr ;
verb      = train | generate | eval | plot ;

train     = "train" IDENT "on" expr [ "for" NUMBER "steps" ] [ "with" kwargs ] ;
generate  = "generate" IDENT "from" STRING "~" NUMBER "chars"
            [ "at" "temperature" NUMBER ] ;
eval      = "eval" IDENT "on" expr [ "|>" "table" ] ;
plot      = "plot" plottarget { "," plottarget } ;
plottarget= expr | "trajectory" "of" expr | "loss" "of" IDENT ;

expr      = mixture ;
mixture   = weighted { "|" weighted } ;
weighted  = [ NUMBER "*" ] push ;
push      = postfix { "#" postfix } ;                (* 右结合:T # (S # μ) *)
postfix   = primary { "~" NUMBER [ "via" call ] | "." IDENT } ;
primary   = transport | call | IDENT | NUMBER | vector | "(" expr ")" ;
transport = "transport" "from" expr "to" expr "using" expr
            [ "for" NUMBER "steps" ] [ "with" kwargs ] ;
call      = IDENT "(" [ arglist ] ")" ;
            (* 含 flow, inv, WGField, WGFlow, reverse, mirror, estimate, ... *)
arglist   = arg { "," arg } ;  arg = [ IDENT ":" ] expr | dims | kwarg ;
dims      = NUMBER "->" NUMBER { "->" NUMBER } ;
kwargs    = kwarg { "," kwarg } ;  kwarg = IDENT "=" expr ;
comment   = "//" 至行尾 ;
```

## 附 B:v0.1 → v0.2 变更记录

1. 注释 `#` → `//`;`#` 改为 pushforward 操作符(黑板记号 T#μ 直接可执行)。
2. 新类型 `Field`、`Map`;新原语 `flow`、`inv`、`stein`、`langevin`。
   场/映射分离,求解器参数移居 `flow`。
3. 新表达式 `transport from ... to ... using ...`,吸收全部生成式训练;
   `rectified_flow` 一词退役。
4. `~` 增加 `via` 从句;删除 `sample` 动词,动词六减一为五。
5. `train` 作用域收窄为判别/预测式 Trainable。
6. 能力标记(可精确采样/有 score/可逆)与按能力的静态检查。
7. 逃生舱注册对象由两类扩为四类。

## 附 C:v0.2 → v0.3 变更记录

1. 新类型 `Divergence`:分布对上的泛函(不限 f-散度),按**族**注册
   (f-散度族、OT 族;核族预留)。双槽,第二槽可省;槽接受
   Distribution/Dataset。内置 `forwardKL`、`reverseKL`、`pearson_chi2`、
   `neyman_chi2`、`W2`;新增 `mirror(D)`、`estimate(D)`
   (分布距离度量议题由 estimate 吸收)。
2. 新对象对 `WGField`/`WGFlow`(测度空间的场/流),与点空间的
   `Field`/`flow`/`Map` 平行;确立"每种几何一对场/流"的命名原则
   (SDE/Markov 核为预留的第三对)。
3. `WGField` 引入 `estimator=` 槽,需求逐槽静态核对;确立
   **统计近似(estimator)与数值近似(steps/solver)的正交分离**。
4. 出身一致性检查:WGFlow 硬、Map 软;v0.2 讨论中的 ensemble-dependent
   能力标记方案废除,改由类型承担。
5. 新增 `reverse(WGFlow, estimator=denoiser(net)) → Field`:学习的时间反演,
   与免费精确的 `inv` 构成刻意对比;diffusion 由此纳入体系
   (probability-flow ODE 落在现有 Field/Map 机器,DDIM 反演白送),
   `WGFlow` 增加 `time=` 参数;SDE 实现留 v0.4。
6. `stein` 原语退役(降级为 `WGField(reverseKL(·), estimator=nw)` 的推论);
   `svgd`、`diffusion` 均为有精确脱糖的 via 糖;`langevin` 暂留 via 糖。
7. 决议:`coupling=`(OT-CFM/reflow)不进 v0.3,随耦合一等化留 v0.4。
8. 逃生舱注册对象扩为七类(+ DIVERGENCE_FAMILY、DIVERGENCE、ESTIMATOR)。
9. 确立**训练类表达式**范畴(§3.4):`transport`、`reverse`、f-散度族的
   `estimate` 是 train 的表达式形态——从句语法、loss 流、`plot loss of`、
   步数核算四者统一;与逐步重解、无持久参数的 estimator 相对。
10. 新增 `estimator=empirical`(经验混合 score 闭式计算,免训练;教学件:
    精确反演 = 记忆训练集,denoiser 的平滑才泛化)及 §3.1 注记
    (reverse 付账后与前向概率流场的反向积分重合;显式场不经 estimator
    不存在,且标准配方让学习只发生在噪声实现的真边际上)。

## 附 D:收编清单(充分性测验记录,随每次测验更新)

充分性猜想:文献中每个 flow 类生成模型都是本语言的派生项——
一个乘积结构上的点(**路径来源 × 耦合 × estimator × carrier × 实现
几何**),不需要新原语。测验记录:

### 已实现(参考解释器可运行)

| 模型 | 路径来源 | 耦合 | estimator | 实现 |
|---|---|---|---|---|
| Flow Matching | 声明式公式 | 独立(自动提升) | `regress(net)` | ODE → Map(可逆) |
| Diffusion(VP,概率流) | 声明式(一行系数之差) | 独立 | `regress(net)`(旧径 `denoiser`) | ODE → Map(`inv` 免费) |
| SVGD / W-梯度流 | `descent(reverseKL)` | — | `nw(kernel)` | 交错 ODE → 粒子;`record=true` 可逆 |
| 贝叶斯采样(unnormalized) | `descent` | — | `nw` + 数值 score | 同上 |
| 摊销蒸馏 | descent 后回归 | — | — | carrier 雏形 |
| CFG / guidance(双模型) | 声明式 ×2,同 noise | 独立 | `regress` ×2 + **场代数** `v_u + w*(v_c − v_u)`(synthetic 出身,inv 免费) | ODE → Map |

### 规格定稿,解释器未实现

| 模型 | 归入方式 | 落位 |
|---|---|---|
| Rectified flow / reflow | RV 上的 `#`(图耦合 = 重复出现规则)+ 有界 `for` | §10.3,v0.4 |
| OT-CFM | `(x0,x1) = rv(couple(A, B, via=ot/sinkhorn(eps)))`:batch 内配对,对等块共享 draw identity;`~ n` 冻结一次耦合 | §10.3,已落地(2026-07) |
| 薛定谔桥(静态) | sinkhorn 耦合 + 布朗桥公式(`bridge` 免设) | §10.4,v0.4 |
| Drifting(一步生成) | descent + samples-only `nw` + `flow(v, into=net)`:回归 net(z) → stopgrad(y+εΦ(y)),loss = 零流判据;1-NFE | §10.6,已落地(2026-07) |
| DDPM 祖先采样 / Langevin | 同场,SDE 实现 → Kernel | §10.5,v0.5 |
| 离散 FM(CTMC) | 律 mixture 路径(`\|`)+ `rate` + `estimator=posterior` | §10.5,v0.5 |
| Masked diffusion | 上行特例:`noise = delta(MASK)` 吸收态 | §10.5,v0.5 |
| Neural ODE / CNF | `field(parameter=net)` + `fit`(forward KL;density 价目表) | §10.7 |
| OT-Flow / RNODE | 同上 + `regularize=kinetic` → OT 测地线 | §10.7 内 |
| Product of experts(SVGD 版) | descent 场线性组合 = 组合散度最速下降,平稳分布 = 几何混合(精确) | §10.3.1,v0.4 |
| 插值速度场(NW 归一化) | `nw(kernel=rbf, normalize=true)` / `via svgd(..., normalize=true)`:SVGD 更新恰为 NW 插值的**分子**,除以核质量 (1/n)Σⱼk(xⱼ,y) 得 W2 速度 ∇log(p/q) 的一致估计(Liu, Yu, Simons, Yi & Beaumont 2024 式 5;分母随查询点变化,不能并入学习率) | 已落地(2026-07) |
| 条件运输(zero-flow) | `(y,x)=rv(joint)` + `|` 双职 + 核一等化:独立耦合下只插值 y 块、两端条件同进网络(Wang, Wang, Liu & Suzuki 2026 式 6/定理 3.3);配对求值只需联合样本,脱钩求值需条件采样器 | §10.8,已落地(2026-07) |
| 散度×几何分派 | `normalize=` 删除(2026-07):reverseKL+metric=stein = 精确 SVGD(未归一化),reverseKL+w2 = 归一化 NW(式 5 一致估计),`mmd(data)` = witness 梯度,`w2(data, eps=)` = Sinkhorn 重心位移;`via svgd` 脱糖 stein/mmd;metric= 槽自此语义真实 | 已落地(2026-07) |
| samples-only 的 nw(统一形式) | nw 统一为平滑 score 差 smooth(∇log p)/Z_p − smooth(∇log q)/Z_q:有 score 时 p 项为系综点上真实 score 的核平滑(不变),Dataset target 时退化为样本 KDE score(同核同带宽,p̂=q̂ 处零点精确;unnormalized = MMD 流,Arbel et al. 2019;带宽在系综∪样本合并集上取 median);`via svgd` 同步接受 Dataset target | 已落地(2026-07) |
| 约束传输(regress(rotation)) | FM 假设类限制到反对称线性场 → 逐片闭式回归(无网);exp 步积分,映射恰为旋转,inv = 转置免费;白化+约束 = 精确认源(MCC 1.000),自由传输 0.984——约束买回可识别性在传输引擎上的对应版 | 台账,已落地(2026-07) |
| SICA-RF(传输引擎,零新词) | 每轮:`couple(Z, decouple(Z), via=ot)` + 声明路径 + regress + `T # Z` = 迭代传输降 KL(MIRI 机制);与 WGF 版同目标异列(声明×regress×ODE vs descent×nw×粒子);保证独立、近似认源;不可先白化(实测) | 台账,已落地(2026-07) |
| 过程独立 ICA(SOBI 象限) | `window(X,L)` 延迟嵌入 + `decouple(block=L)` 排列积 + `rotation(block=L)` = R⊗I_L(滞后平均反对称矩投影):高斯 AR 源瞬时法必卡(`mixed_ar` 实测 MCC 0.809)、窗口版满分;窗口云独立 = 过程独立(至视界 L) | 台账,已落地(2026-07) |
| 自然梯度 ICA(旋转流形流) | `whiten(X)` + `descent(reverseKL(decouple(Z)~n), from=Z, family=rotation)`:约束在路径上,场 = so(d) 投影、flow = 指数步皆为其定理;z ← exp(lr·Ω)z = Amari 1998 的连续时间形式;约束买回线性 ICA 可识别性(自由流受 Darmois 构造之害);尖锐带宽(投影后方差便宜);`field(..., family=)` 教学报错指路 descent | §10.6 谱系注记,已落地(2026-07) |
| SBI 条件 descent(conditional SVGD) | `from=(TH0 \| D0)`:条件化沿用唯一语法 `\|`(新关键词方案被否),系综 = 冻结联合、y-块动、x-块钉在观测;钉住即条件化是 log 的定理(∇_y log p(y\|x) = ∇_y log p(y,x)),自由行降条件 KL 精确;目标始终是联合模拟抽样(likelihood-free),核权重的钉住行距离 = ABC 式窗口;仅 reverseKL(mmd/w2 不分解);与 sbi_npse 构成 {descent, transport} SBI 对,FSM-MLE(Khoo et al. 2025)= 同格 Dirac 载体/频率对偶 | 台账,已落地(2026-07) |
| Classifier guidance | `field(pt) + w·b_t·grad(ratio(qt,pt))`;`ratio_then_grad` 归位;w=1 精确(条件化定理) | §10.3.1 注记,v0.4 |
| CFG(残差参数化) | `estimator=regress(net, base=v_u)`:冻结基场直接回归 guidance 方向,组合项表示 → 基场权重精确抵消 | §10.3.2,v0.4 |
| 残差/适配器微调(LoRA 式) | 场代数 × free/pinned:pinned 主干 + free 修正场,`fit` 只动 free | §10.7 补注 |

### 有意留在片段外(排除理由即资产)

- **GAN**:目标是 minimax,内层含另一个优化——直线程序语义容不下;
  且无路径无场。
- **离散层 normalizing flow**(RealNVP 一族):直接参数化可逆 Map,
  绕过 path/field 轴;`fit` + "架构保证 log-det" 或可收编,悬而未决。
- **Consistency / 教师蒸馏**:carrier 回答一半(学到的是 Map),
  "目标来自教师流"尚无槽位。开放疑难。

统计(2026-07):已实现 6 族,定稿待实现 12 族,共 18 族,共享六个词
(`prob / field / flow / # / descent / couple`)、三个具名槽
(`estimator= / carrier= / parameter=`)与场上的线性代数(§10.3.1)。
