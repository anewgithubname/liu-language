# 流 (Liu) v0.3 — 参考解释器原型

规格见 `docs/liu-spec.md`。单文件 tree-walking 解释器(`liu.cpp`),
后端为 Juzhen CPU(BLAS)。

## 构建

```bash
sudo apt install g++ libopenblas-dev liblapack-dev libboost-dev   # Linux
# macOS: brew install openblas
./interpreter/build.sh          # 自动 init submodule;产出 build_liu/liu
./interpreter/test.sh           # 三层回归(改解释器后必跑,见 test/README.md):
                                # 退出码 + golden 输出(可复现契约;有意变更用
                                # REGEN=1 换金样)+ 教学报错子串 + 定量断言
```

## 运行示例

```bash
./build_liu/liu examples/hello.liu          # hello world:名词、采样、plot、flow matching 四行(约 3 秒)

# 概率路径(spec §10.2 定稿设计):插值公式 → prob 取律 → field 回归 → flow
./build_liu/liu examples/path_fm.liu        # xt = t*data + (1-t)*noise
./build_liu/liu examples/path_diffusion.liu # xt = sqrt(1-t*t)*e + t*data
                                                        # 一行系数之差,FM 变 diffusion

./build_liu/liu examples/svgd.liu           # SVGD:via 糖 + descent 机制层
./build_liu/liu examples/svgd_inverse.liu   # 录史反演:record=true 后 inv 合法
./build_liu/liu examples/svgd_bayes.liu     # 未归一化密度(贝叶斯):score 免配分函数
./build_liu/liu examples/svgd_normalized.liu # 一个散度、两种几何:metric=stein(精确 SVGD)vs w2(归一化 NW,Liu et al. 2024)
./build_liu/liu examples/svgd_data.liu      # samples-only:mmd(data) = witness 梯度(MMD 流)vs reverseKL(data) = KDE score 差
./build_liu/liu examples/w2_descent.liu     # Wasserstein descent:Sinkhorn 重心位移场(无核;对经验目标记忆化)
./build_liu/liu examples/drifting.liu       # 摊销流 flow(v, into=net):drifting 一步生成器,1-NFE 推理(spec 10.6)
./build_liu/liu examples/ot_cfm.liu         # OT-CFM:couple(noise, data, via=ot),batch 内配对,轨迹拉直(spec 10.3)
./build_liu/liu examples/reflow.liu         # rectified flow:有界 for + via=paired,一轮拉直、一步采样(spec 10.1)
./build_liu/liu examples/schrodinger_bridge.liu # 薛定谔桥 = sinkhorn 耦合 + 布朗桥公式,零新词(spec 10.4)
./build_liu/liu examples/cond_transfer.liu  # 条件核:y|x 构造/求值,zero-flow 条件运输(spec 10.8)
./build_liu/liu examples/guidance_cfg.liu   # CFG guidance:场代数 v_u + w*(v_c - v_u)(spec 10.3.1)
./build_liu/liu examples/guidance_residual.liu # 残差估计器 regress(net, base=v_u):条件采样免公式,基场权重精确抵消(spec 10.3.2)
./build_liu/liu examples/flow_matching.liu  # transport 旧糖(内部脱糖为路径回归)
./build_liu/liu examples/diffusion.liu      # reverse+denoiser(隐式路径专用,遗留)
./build_liu/liu examples/err_no_score.liu   # 报错即教学(×7)
./build_liu/liu examples/err_inv_wgflow.liu
./build_liu/liu examples/err_provenance.liu
./build_liu/liu examples/err_repeated_dist.liu  # 分布无抽样身份 → 要求 rv()
./build_liu/liu examples/err_residual_base.liu  # base= 出身不符:异源基场的残差不是本路径的场
```

### 概率路径(v0.4 §10.2 设计的先行实现)

- 保留时间符号 `t`;算术 `+ - * /` 与 `sqrt/exp/log/sin/cos` 可构造系数;
- 插值公式 = 仿射 RV 表达式 Σ cᵢ(t)·ξᵢ:**单次出现的分布/数据集自动提升**为
  独立随机变量,**重复出现硬报错**(分布没有抽样身份),`rv(D)` 显式声明
  身份(同名复用 = 同一抽样,系数自动合并);
- `prob(xt)` 取律得 ProbPath;`field(pt, estimator=regress(net))` 训练
  条件期望场 E[ẋ_t|x_t](一个 L2 回归,FM 与 DSM 皆其特例);
- `transport` 保留为糖,内部即脱糖为上述回归;声明式路径不需要 `reverse`
  (时间反演 = 公式里 t ↦ 1-t)。

### §10.2.1 统一(一个提取子、一条管线)

- 散度诱导路径而非场:`descent(D, from=q0, time=, metric=w2)` 是最速下降的初值问题;
- `field` 双接口:声明式路径 → `regress`(回归条件期望场),descent →
  `nw`(场由 h 变换给定,模拟时逐步重估——estimator 的实现策略);
- `flow(v, steps=, lr=)` 吞并原 WGFlow 的模拟;WG 场积出的 Map 默认不可逆,
  **`record=true` 录史后可逆**(NW 场对任意点逐点重建,反演 = 不动点求解;
  "inv 免费/花内存/花训练"三级价目表见 spec §3.1);出身硬检查挂 `from=`,
  **仅限未录史的 map**——录史冻结后回放自足,异源测度合法、降级为输出
  注记(spec §3.2 修订);
- `reverse` 改收 WGpath;`WGField`/`WGFlow` 退役为指路的教学报错;
- svgd 糖脱糖为 `flow(field(descent(reverseKL(target), from=N(0,I), metric=stein), nw)) # ...`(Dataset 目标则为 `descent(mmd(target))`)。

## 交互 Playground(方案 A)

```bash
./interpreter/build.sh
python3 web/server.py --port 8080     # 零依赖(Python 标准库)
# 打开 http://localhost:8080 :编辑程序 → Run → 逐语句 loss 曲线(按行分组、
# 实时流式)+ 图卡动画(边跑边出、标注来源行)+ 报错行在编辑器内高亮
```

直线程序让每个运行期事件都能静态归因到源码行——loss 事件、图数据、
报错都带 `line` 与 `iter` 栈(有界 `for` 的迭代索引,§10.1;循环外为
`[]`)。playground 按 (line, iter) 分组 loss 曲线与图卡,逐轮可视。

服务器为语言沙箱补上进程级护栏:90s 墙钟超时杀、2GB 内存上限、程序
≤16KB、并发 ≤4;确定性使整轮缓存可靠(同文本+seed 重跑即时返回)。
静态展示版(无服务器)见 `web/demo.html`。

## v0.3 规格覆盖

| 已实现 | 说明 |
|---|---|
| `seed`、绑定、`//` 注释 | |
| `gaussian/uniform/moons/ring/spiral`、加权混合 `or` | 高斯混合带解析 score |
| `~ n`(Distribution/pushforward/Dataset 子采样) | Dataset 携带出身 |
| `unnormalized(L)`(坐标符号 x1/x2 书写对数密度) | 有 score(数值梯度)、不可精确采样——`~` 必须走 via;SVGD 的贝叶斯本行 |
| `~ n via svgd(kernel=rbf, steps=, lr=)` | 脱糖为 `descent(reverseKL(target), metric=stein)`(score 目标;Stein 几何下精确)或 `descent(mmd(target))`(Dataset 目标);`normalize=` 已删除——形式由散度×几何决定 |
| `mlp(a -> b -> ... -> z)` | 隐层 ReLU,末层线性,输入自动 +1 时间维 |
| `transport from A to B using net [for N steps] [with lr=, batch=]` | flow matching,流式 loss |
| `flow(v, steps=)` / `inv(T)` / `T # μ` / `T # X` | Euler ODE;pushforward 惰性 |
| `reverseKL(p[, q])` / `mmd(data)` / `w2(data, eps=)`、`descent(D, from=q0, time=, metric=w2/stein)` | 隐式路径(最速下降 IVP,§10.2.1);**散度×几何决定场的形式**:reverseKL+stein = 精确 SVGD、reverseKL+w2 = 归一化 NW(一致 W2 速度)、mmd = witness 梯度(MMD 流)、w2 = Sinkhorn 重心位移(无核;对经验目标记忆化;暂不支持 record=) |
| `field(qt, estimator=nw(kernel=rbf))` + `flow(v, steps=, lr=)` | 梯度流与声明式路径共用一条管线;WG 场的 Map 不可逆、出身硬检查挂 `from=`(未录史;录史后异源合法 + 注记);`WGField`/`WGFlow` 已退役为教学报错 |
| `flow(v, into=mlp(d->...->d), steps=, lr=, batch=, trainlr=)` | 摊销流(§10.6,drifting):优化器代替粒子模拟器,每步回归 net(z) → y+ε·Φ(y);1-NFE 推理;loss = 零流判据;硬钉 from=、拒绝 inv、与 record= 互斥 |
| `reverse(G, estimator=denoiser(net), steps=, lr=)` | DSM(OU 前向,闭式边际),返回概率流 ODE 的 Field |
| `regress(net, base=v)`(残差估计器,§10.3.2) | 冻结基场、回归目标减 base 前向 → 极限恰为 guidance 方向;返回 `1*[base]+1*[Δ]` 组合场,场代数下基场权重精确抵消 |
| 条件核(§10.8):`(y,x)=rv(joint)`、`y\|x`、`linear_gaussian`/`sine_gaussian`;混合改用 `or` | 公式里 `y\|x` 构造条件端点(两端条件都进网络,Zero-Flow Encoders 式 6);条件 map 推核、`\|` 填槽(配对只需联合样本;脱钩需条件采样器) |
| `couple(A, B, via=independent/ot/sinkhorn(eps)/paired)`(§10.3) | 对上的联合律:batch 内配对(Hungarian / 熵正则 plan);`(x0,x1)=rv(couple(...))` 对等块共享 draw identity 进公式 → OT-CFM;`~ n` 冻结一次耦合;`via=paired` 配对 (z, T#z)(reflow);sinkhorn 耦合 + `sqrt(t*(1-t))*sigma*z` = 薛定谔桥(§10.4) |
| `decouple(X)` / `mixed_sources(m)` / `whiten(X)` / `family=rotation`(ICA 线) | decouple = couple 的对偶(边缘乘积,逐坐标 bootstrap);mixed_sources 为 (x,s) 对偶块联合(真源随抽取同行,恢复质量可检验);whiten 确定性 PCA 标准化(无 RNG);`descent(..., family=rotation)` 把路径约束到旋转轨道——场 = so(d) 投影、flow = 指数步 z←exp(lr·Ω)z 皆为其定理(Amari 自然梯度 ICA;须先白化;record=/into= 教学报错,field 上写 family= 报错指路 descent)——`sica.liu` / `sica_rotation.liu` |
| 有界 `for k in 1..K { ... }`(§10.1) | K 字面量、上限 64,宏展开语义;体内重绑定、k 作常量进表达式;事件带 `iter=[k]`,playground 逐轮分组;`while`/`break`/`if` 永久教学报错 |
| `plot a, b` / `plot trajectory of x` | ASCII 散点,轨迹三帧 |
| `plot_signal x, y` | 信号/波形视图:列按索引序、每坐标行一条线;裸分布抽 500;终端 min–max 带宽降采样,web 折线(`signals.liu`) |
| `mixed_signals(m)` 真时序玩具 | 一次抽取 = 一条轨迹(sin + 锯齿,无理周期比→源云遍历独立;随机性 = 每 draw 两相位);端到端 BSS 见 `sica_signals.liu`(波形进→旋转流→波形出,MCC 1.000) |
| 过程独立四件套:`window(X,L)`/`unwindow(Z,L)` / `decouple(block=L)` / `rotation(block=L)` + `mixed_ar` | 延迟嵌入→排列积→R⊗I_L 约束流;高斯 AR 源瞬时法必卡(实测 0.809)、窗口版满分(`sica_process.liu`,quant 双侧断言) |
| SICA-RF 传输引擎(零新词) | 每轮声明路径 → regress → `T # Z`,迭代传输降 KL(MIRI 机制);OT 耦合减扭曲;保证独立、近似认源;不可先白化——`sica_rf.liu` |
| `estimator=regress(rotation)` 约束传输 | FM 假设类 = 反对称线性场,逐片闭式(无网无 SGD);映射恰为旋转、inv 免费;白化+约束 = MCC 1.000——`sica_rf_rotation.liu` |
| `regress(rotation(block=L))` 过程级 | 滞后合并矩 + 同一闭式解 C×C;高斯墙对边缘传输同样成立(0.72)、块类开门(1.000)——`sica_rf_process.liu`,2×2 矩阵闭合 |
| 静态检查 | score 缺失 / inv(WGFlow) / 几何不匹配 / `v # μ` 不留糖 / 资源上限 |

| 未实现(报错提示) | 归属 |
|---|---|
| `locallinear` / `ratio_then_grad` / `empirical` estimator | v0.3 完整版 |
| `estimate` / `mirror` / `W2`(sinkhorn) | v0.3 完整版 |
| `train`(classifier/lm)、`generate`、`eval`、mnist/text8 | v0.3 完整版 |
| 核按槽填充、`locallinear`、零流判据动词 | v0.4 余项(spec §10.8;对账见 docs/roadmap.md;有界 for、couple、薛定谔桥、reflow 已落地) |
| SDE/Markov 核对、离散流(CTMC)、零流判据动词 | v0.5(spec §10.5) |

已知偏差(原型简化,非规格变更):静态检查在逐句求值前执行而非独立
pass;`plot` 对 pushforward 默认取 500 样本;退出时的 profiler 输出来自
Juzhen 全局计时器。
