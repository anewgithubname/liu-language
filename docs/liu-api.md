# Liu (流) — API Reference

*Exhaustive usage reference for every function, operator, and verb in the
v0.3+ prototype, checked line-by-line against `interpreter/liu.cpp`. For
the narrative semantics see `docs/liu-reference.md`; for a 3-minute
overview see `docs/liu-cheatsheet.md`.*

---

## 1. Program form

- One statement per line. `//` starts a comment (to end of line).
- Two statement kinds: **bindings** `name = expression` and **verbs**
  (`seed`, `plot`).
- The first statement should be `seed n` (integer). If omitted, the
  interpreter inserts `seed 0` and says so in the output.
- **Reserved symbol** (cannot be bound): `t` (time, used in path
  formulas). `x1`/`x2` — the coordinate symbols of `unnormalized(...)` —
  are not reserved: user bindings shadow them; the coordinate reading
  applies only while the names are unbound.

## 2. Operator precedence

Loosest to tightest; higher rows are split first:

| level | operators | meaning |
|---|---|---|
| 1 | `or` | mixture of distributions |
| 2 | `+` `-` | addition / subtraction |
| 3 | `*` `/` | multiplication / division |
| 4 | `#` | pushforward |
| 4.5 | `\|` | conditioning — binds *tighter* than `#` |
| 5 | `~ n [via alg(...)]` | sampling (postfix) |
| 6 | unary `-`, calls `f(...)`, vectors `[a, b]`, parens `(...)`, the `transport` sentence | primaries |

Consequences worth knowing: `T # src ~ 100` parses as `T # (src ~ 100)`
(sample first, then push); `T # y0|c0` parses as `T # (y0|c0)` — the
math reading, no parens needed; `prob(xt | d)` is `prob((xt | d))` for
the same reason; `->` is not a general operator — it appears only
inside `mlp(...)`.

## 3. Operators

### `=` — binding

`name = expression`. Straight-line, no rebinding of reserved symbols.
Reusing a name shadows the previous binding.

### `~` — sampling (postfix)

`e ~ n` where **n is a literal integer**, `1 ≤ n ≤ 100 000`
(an expression for n is a parse error). By the type of `e`:

| left operand | result |
|---|---|
| `Distribution` | draw n i.i.d. points → `Dataset`. Requires an exact sampler; `unnormalized(...)` targets must use `via`. |
| `Dataset` | subsample n points **with replacement** → `Dataset` (carries the parent's provenance and trajectory). |
| lazy pushforward `T # μ` | draw n from μ, transport through T → `Dataset` (records the trajectory). |
| `Field`, `RV`, `Path` | error with a pointer to the right construction. |

### `via` — inference-algorithm slot

`target ~ n via svgd(kernel=rbf, steps=500, lr=0.5)` — the only
algorithm in the prototype. `target` is either a Distribution with a
score (Gaussian, Gaussian mixture, or `unnormalized`) — desugaring to
`flow(field(descent(reverseKL(target), from=N(0,I), metric=stein), nw), steps=) # (N(0,I) ~ n)`
(svgd IS the Stein-geometry descent; the update is exact there) — or a
**Dataset**, desugaring to the `mmd` descent (the witness-gradient
flow). Initialization is `N(0, I)` in the target's dimension; ~12
trajectory frames are recorded. `normalize=` is gone (2026-07): the
form follows from the divergence and the metric.

### `#` — pushforward

| form | semantics |
|---|---|
| `Map # Distribution` | **lazy** — returns a pushforward Distribution; each later `~ n` draws fresh base points and transports them. |
| `Map # Dataset` | applies the map **now** → `Dataset` with recorded trajectory. |
| `Field # μ` | error by design: integrate first (`flow(v) # μ`). |
| `Path # μ` | error: extract the field first. |

Provenance rules for gradient-flow (descent) maps:
- **unrecorded** map: hard-pinned to its path's `from=` measure — the
  map is the simulation itself, so foreign particles change the
  dynamics (hard error, names the `record=true` escape);
- **recorded** map (`record=true`, after one forward application): the
  frozen replay is self-contained — any measure of matching dimension
  is legal in both directions; a mismatch prints a once-per-program
  note stating what the answer means.

### `|` — conditional kernels (spec §10.8)

Two jobs, one word:

- **Law-gate conditioning** (2026-07) — `prob(xt | d)`: the path formula
  is built from **bare** blocks (`xt = t*th1 + (1-t)*noise`, pure draw
  interpolation), and `|` attaches the conditioning **where the law is
  taken**. Each index must be a coordinate block of the same joint draw
  as the term it conditions (`(d, th) = rv(joint)`; draw identity is
  enforced — an index coupled to no term is a teaching error, since
  conditioning on it could not bite). Several indices: `prob(yt | x1, x0)`
  or chained `(yt | x1) | x0` — **declared in term order** (condition
  slots are assembled in term order end to end; out of order is a
  teaching error). Training regresses the conditional velocity
  `v(y_t; conditions, t) `— all declared conditions enter the net (Wang,
  Wang, Liu & Suzuki 2026, eq. 6). The retired inline spelling
  `t*(y1|x1) + ...` is a teaching error: the formula interpolates draws,
  conditioning is an operation on laws.
- **Table application** (2026-07, SICA line) — a conditional map pushed
  onto a **data-backed** conditioned block, `T # (E | C)` with `(E, C)`
  block views of one table (e.g. `lagsplit` windows): eager, per-column
  conditional ODE integration — each column's free block moves under
  its **own** pinned context, and when the map was trained with several
  condition slots the one context **tiles all of them** (SICA Theorem
  3.3's substitution `C̃′ := C̃`; also the kernel surface's "all slots
  share one value" rule). Returns the moved table in the input's
  `[free; pinned]` layout, blk-marked (destructures). No provenance
  pin: a trained conditional net is a total function. Verified
  contracts: exact identity at `lr = 0`; near-identity between equal
  conditional laws (`sica_alg1.liu`, 0.898 over two full refinements).
- **Kernel evaluation** — a conditional map pushes a kernel,
  `Tk = T # (y0|x0)`, whose condition slots stay **open**; `|`
  fills them: `Tk | x0` (paired — same draw, needs only joint
  samples), `Tk | [1.2]` (a point), `Tk | μ` / `Tk | X`
  (a measure / dataset of condition values — these decoupled forms need
  the source joint's conditional sampler). The result samples like a
  distribution (`~ n`) and yields the paired dataset `(y′, x)`.
  Sampling or plotting an **un-instantiated** kernel is an error.
  All condition slots share one value in this version.

### `(y, x) = e` — destructuring bind

Destructures **one draw** of a joint distribution into two named
coordinate blocks (same draw identity). `e` must be a joint
(`linear_gaussian`, `sine_gaussian`, `couple(...)`) or `rv(joint)`.
For `couple` only, an optional `~ n` **freezes one coupling**:
`(x0, x1) = rv(couple(A, B, via=ot)) ~ 500` computes the matching once
over 500 pairs and binds block views of the frozen joint Dataset
(training then subsamples columns, pairs intact). Without `~`, the
coupling is re-drawn and re-matched inside every training batch
(minibatch coupling — the OT-CFM convention).

### `for k in 1..K { ... }` — bounded iteration (spec §10.1)

The ONLY loop. `K` (and the lower bound) must be **literal integers**;
at most 64 iterations; nesting allowed. Semantics is macro expansion:
the body is K copies with `k` a constant, so the checker still sees a
finite straight-line program. Rebinding inside the body is ordinary
rebinding (`T = flow(...) # T` iterates a transport). Every event a body
statement emits (loss, plot, error) carries `iter=[.., k]` — the
playground groups loss curves and plot cards per round. `while`,
`break`, `continue`, `if` do not exist (permanent teaching error);
data-dependent bounds are rejected at parse time. The loop index cannot
be `t` and disappears after the loop.

### `or` — mixture

`w1*A or w2*B or ...` — components must be Distributions; unweighted
components get weight 1; weights are normalized by their sum (they need
not sum to 1). The mixture has a score iff every component is Gaussian.
Intended for the samplable toy distributions.

### Arithmetic `+ - * /` and unary `-`

Dispatch by operand types, in this order:

| operands | meaning |
|---|---|
| number ∘ number | plain arithmetic |
| t-expression ∘ t-expression/number | coefficient algebra, e.g. `sqrt(1 - t*t)` |
| x1/x2-expression ∘ x1/x2-expression/number | log-density algebra for `unnormalized` |
| number `*` Distribution | mixture weight (only meaningful before `or`) |
| coefficient(t) `*` Distribution/Dataset/RV | a term of a path formula (auto-lifts the measure to a fresh RV) |
| RV `±` RV | path-formula sum; a measure occurring twice is an error — disambiguate with `rv(...)` |
| number `*` Field, Field `/` number, Field `±` Field, `-`Field | **field algebra** (self-contained fields only; descent fields are rejected — combine at the divergence level). Repeated leaves merge; zero-net combinations are an error; summands must share dimension and (when known) starting measure. |

### `->` — layer chain

Only inside `mlp(...)`: `mlp(2 -> 64 -> 64 -> 2)`.

### `[a, b, ...]` — vector literal

Numbers only (means, box corners).

## 4. Distribution constructors

| constructor | defaults | dim | sampler | score |
|---|---|---|---|---|
| `gaussian(mean, std=1)` | isotropic; `mean` a vector | len(mean) | ✓ | ✓ |
| `uniform(lo, hi)` | box; **half-width taken from the first coordinate** (prototype: use equal side lengths) | len(lo) | ✓ | ✗ |
| `moons(noise=0.1)` | two interleaved half-circles | 2 | ✓ | ✗ |
| `ring(r=2, noise=0.1)` | circle radius r, radial noise | 2 | ✓ | ✗ |
| `spiral(turns=2, noise=0.1)` | | 2 | ✓ | ✗ |
| `torus(R=1, r=0.4, noise=0.05)` | ring's big sibling — a genuine **2-D manifold in R³** (major radius R, minor r, isotropic noise). Uniform in the two ANGLES like ring (a data toy, not the area-uniform law: the inner rim is denser). RNG per sample: θ, φ, then 3 normals. Plots of 3-row datasets render in 3-D (see §15). `examples/manifold_torus.liu` | 3 | ✓ | ✗ |
| `unnormalized(L)` | L written in `x1`, `x2`, up to an additive constant | 2 | ✗ (`~` needs `via`) | ✓ (numeric ∇L, central differences, h=1e-3) |
| `linear_gaussian(slope=0.5, noise=0.3)` | **joint** over (y, x): x ~ N(0,1), y = slope·x + noise·ε; conditional sampler ✓ | 2 (dy=1, dx=1) | ✓ | ✗ |
| `sine_gaussian(freq=2, noise=0.2)` | **joint**: y = sin(freq·x) + noise·ε | 2 (dy=1, dx=1) | ✓ | ✗ |
| `decouple(X[, block=L])` | the **dual of couple**, any dimension: the product of a Dataset's empirical **block** marginals. `block=1` (default) bootstraps each coordinate independently (d·n picks) — all dependence forgotten. `block=L` bootstraps each block of L rows **jointly** (independent picks across blocks): on `window()`ed data this is the paper's **permuted product** — each channel keeps its own within-window dynamics, only cross-channel dependence is forgotten (**process independence**). L must divide the row count. Argument must be a Dataset; a Distribution errors with a pointer to `~ n`. Core of SICA/MIRI-style iterations. | d | ✓ | ✗ |
| `mixed_sources(mix=0.7)` | ICA toy, a **pair-joint over (x, s)**: independent uniform × bimodal sources `s`, observed through the linear mixture `x = [[1,m],[m,1]]·s` — one draw carries both, rows [0,2) = x, rows [2,4) = s. Destructure to keep the ground truth: `(X, S) = rv(mixed_sources(0.7)) ~ n` freezes one draw (block views; `prob(X)` extracts the observation Dataset). The demo task is still to recover the product structure from `X` alone — `S` exists so recovery is *checkable* (`examples/sica.liu`, `examples/sica_rotation.liu`). Unfrozen block marginals have no closed form (teaching error). | 4 (dy=2, dx=2) | ✓ | ✗ |
| `mixed_signals(mix=0.7)` | ICA toy, **time-series** version — same pair-joint shape as `mixed_sources`, but one draw of n columns is **one trajectory**: `s1 = sin` (arcsine marginal), `s2` = sawtooth (uniform marginal), periods ~39.6/~46.8 **samples** with an irrational, well-separated ratio, so the phase pair fills the torus and the source *cloud* is genuinely independent (ergodicity — a rational ratio draws a closed Lissajous curve and independence-seeking flows rightly refuse to converge). Randomness = two fresh phases per draw (2 uniforms, independent of n); the waveforms are deterministic. `plot_signal` shows the channels; the full unmixing demo is `examples/sica_signals.liu` (MCC 1.000). | 4 (dy=2, dx=2) | ✓ | ✗ |
| `mixed_ar(mix=0.7)` | ICA toy, **Gaussian AR** version — the identifiability wall made runnable: two stationary unit-variance Gaussian AR(1) sources (ρ = 0.9 / −0.6, very different spectra), same pair-joint layout and mixing as `mixed_signals`. Every time-marginal is Gaussian, so **instantaneous** independence-seeking provably cannot identify the rotation; the signal lives in the lagged structure — `window` + `decouple(block=)` + `rotation(block=)` recover it (`examples/sica_process.liu`: instantaneous stuck at MCC ≈ 0.8, windowed 1.000). RNG: two normals per time step, interleaved. | 4 (dy=2, dx=2) | ✓ | ✗ |
| `mixed_nl(gain=0.4)` | ICA toy, **nonlinear-mixing** version — the same AR pair as `mixed_ar` observed through an **instantaneous invertible nonlinear warp**: four residual GELU layers `x ← x + a·gelu(W_j·x)` (fixed `W`s, gain `a` the knob; each layer a residual contraction, so the mixture stays in the identifiability theorem's class; GELU is non-odd, so the warp has an even component no whitening + rotation can linearize). Depth is load-bearing (measured): at two layers a rotation still correlates ~0.97 with the sources; at four the linear class genuinely sticks (~0.90). The working demo is the coarse-to-fine composition — rotation stage to the linear ceiling, then `project=instant` peels the nonlinear residual (`examples/sica_nl.liu`: 0.73 → 0.90 → 0.97). RNG identical to `mixed_ar`. | 4 (dy=2, dx=2) | ✓ | ✗ |
| `couple(A, B, via=independent)` | **joint over pairs** (spec §10.3): marginals A, B (Distribution or Dataset each); a draw samples n from each side and pairs **within the batch**. `via=independent` keeps draw order; `via=ot` matches exactly (Hungarian, deterministic given the draws); `via=sinkhorn(eps=0.1)` samples partners from the entropic plan (cost normalized by its mean; n extra uniforms of RNG). `via=paired` pairs each z with its own image: the second slot must be a pushforward of the first marginal — `couple(base, T # base, via=paired)` draws (z, T(z)) (reflow). `ot`/`sinkhorn` require equal dimensions. The two blocks are **peer coordinates**: both may appear as formula terms of one draw — `xt = t*x1 + (1-t)*x0` trains on genuinely paired endpoints. `rv()` wraps the couple, never its marginals (the pair identity lives on the couple). | dA+dB | ✓ | ✗ |

### `window(X, L)` — delay embedding

`window(X, L)` reshapes a **trajectory** Dataset (d channels × n time
points, columns in time order) into the cloud of its sliding windows:
`(d·L) × (n−L+1)`, **channel-major** rows (row `c·L + lag` = channel c
at t+lag), so `decouple(block=L)` and `family=rotation(block=L)` see
one block per channel. Deterministic, no RNG. This is the paper's
one-line reduction: *process-level statements about a time series are
cloud-level statements about its window embedding* — cross-channel
independence of windows IS process independence (up to horizon L).
A Distribution argument is a teaching error (sample the trajectory
first); L must lie in [1, n]. The inverse is **`unwindow(Z, L)`**: take
the lag-0 row of each channel block, `(d·L) × m → d × m` — read the
channel trajectories back off a window cloud (after a window-space
flow, whose `rotation(block=L)` maps keep windows consistent windows
of one rotated series). Plotting the raw cloud draws L shifted copies
of every channel; `plot_signal unwindow(Z, L)` draws each channel
once. Deterministic, no RNG; L must match the `window` that built the
cloud. **`lagsplit(W, L)`** (2026-07, SICA line) reorders a
channel-major window cloud into `[elements; contexts]`: the lag-0 row
of each channel becomes the element block (rows `[0, d)`), the
remaining lags stack lag-major below; the result carries the block
marker, so it **destructures** — `(E, C) = lagsplit(W, L)` gives the
element and context as formula-ready block views of one table (the
formula moves E, the law gate conditions on C — Algorithm-1-style
per-element refinement), while `decouple(block=L)` keeps operating on
the channel-major original. Pure row permutation, no RNG; L ≥ 2.

### `whiten(X)` — Dataset standardization

`whiten(X)` takes a **Dataset** and returns the PCA-whitened Dataset:
`Z = Λ^{-1/2} Uᵀ (X − mean)`, so `cov(Z) = I` exactly. Deterministic —
sample statistics only, no RNG consumed; eigenvector signs are
canonicalized (largest-magnitude entry positive), so the output is a
pure function of the input. A Distribution argument is a teaching error
(mean/covariance are sample statistics — sample first); a degenerate
covariance direction is an error. The companion of `family=rotation`:
rotations preserve the covariance, so whitening first puts every
candidate linear unmixing on the single orbit `O(d)·Z`
(`examples/sica_rotation.liu`).

## 5. Random variables and declared paths

- **`rv(D)`** — one named draw of a Distribution or Dataset. Reusing the
  bound name means *the same draw* (coefficients merge); distinct `rv`
  calls are independent. `rv` of something that is **already** a random
  variable is a teaching error: identity is minted exactly once (at
  `rv()` for measures, at the destructuring bind for joints) —
  re-wrapping would either do nothing or silently break a coupling.
- **Auto-lifting** — a Distribution/Dataset occurring **once** in a
  formula is lifted to a fresh independent RV; occurring **twice** is a
  hard error (same draw or independent copies? say which, with `rv`).
- **Path formulas** — affine combinations `Σ cᵢ(t)·ξᵢ` with
  t-dependent coefficients built from arithmetic and the unary math
  functions. All terms must share one dimension.
- **`prob(xt)`** — the law of an RV expression. `prob(xt | d0[, d1…])`
  attaches conditioning indices at the gate (law-gate conditioning,
  §3 `|`) — the result is the curve of **conditional** laws. For a
  **t-dependent formula** → `ProbPath` (a curve of measures; consumed by `field`, not
  samplable — a curve needs a time). For a **t-free single draw** (an
  unscaled, unconditioned RV: `rv(D)` or a destructured block) the law
  is a *measure*, returned as such and hence samplable/plottable:
  `prob(rv(gaussian(...))) ~ 500` is legal, and for a couple block
  `prob(x0)` is its **marginal** (the stored input for minibatch
  coupling; the frozen block's empirical measure under `~ n`; the lazy
  pushforward `T # base` under `via=paired`) — the gate from the RV
  world back to the samplable world, made exact
  (`examples/couple_marginals.liu`). Scaled or multi-term t-free
  formulas have no closed-form law in the prototype (convolution) —
  teaching error.

## 6. Divergences

- **`reverseKL(p[, q])`** — `KL[q ‖ p]`, target slot `p` (Distribution
  with a score, or Dataset — a Dataset target makes the `nw` field
  descend onto the KDE of the samples). Optional second slot names the
  moving measure and must agree with `descent`'s `from=`.
- **`mmd(data)`** — ½·MMD²(q, p), target a **Dataset** (samples are the
  functional's whole interface; a Distribution errors with a pointer to
  `~ n`). Its W2 descent field is the witness gradient — exact for the
  empirical measures, no estimation involved.
- **`w2(data, eps=0.1)`** — ½·W₂²(q, p), target a **Dataset**. The
  descent field is the barycentric displacement of the entropic plan
  (Sinkhorn, cost normalized by its mean, deterministic). Honest
  property: its minimizer over an empirical target is the empirical
  measure itself — as `eps → 0` the particles snap onto the data points
  (memorization; kernel descents generalize by smoothing). `record=true`
  is not yet supported for `w2` descents.
- `forwardKL` — teaching error: needs an explicit density-ratio estimate
  (the h-transform, Liu et al. 2024 Thm 2.1; the `locallinear` family is
  future work). `estimate`, `mirror`, `W2` — reserved names.

## 7. `descent(D, from=, time=3, metric=w2, family=free)`

The steepest-descent curve of divergence `D` — an initial-value problem
in measure space → `DescentPath`.

| argument | required | notes |
|---|---|---|
| `D` | ✓ | a Divergence (positional) |
| `from=` | ✓ | initial measure — Distribution, Dataset, or a **conditioned block of a frozen joint** `(y0 \| x0)` (2026-07, SBI): the ensemble is the joint, only the y-block rows move, the x-block is pinned. Pinning **is** conditioning for `reverseKL` — ∇_y log p(y\|x) = ∇_y log p(y,x) (log p(x) dies under ∇_y) — so the free rows of the joint nw field descend the **conditional** KL exactly (conditional SVGD; the target stays a joint Dataset). Estimator (2026-07 revision & unification): both conditioning syntaxes lower to the ONE conditional-likelihood estimator — the NW regression `L̂(x0, w_l) = Σ_j k_hy(x0, y_j) k_hl(w_l, w_j) / Σ_j k_hl(w_l, w_j)` with the §10.10 likelihood-role bandwidths — here read as frozen **per-particle** library weights (each ensemble column conditions on its own pinned block); the free-row smoothing keeps the pooled rule, pooled over the free rows only. The deleted assemblies: the pooled-joint-space field (the pinned-row kernel was an ABC window calibrated to the cloud's spread, ~7× the simulator noise — measured: 30% of the posterior mass in valleys carrying 1%) and raw kernel weights (fine at N=1, the exchangeability trap at N>1). Score targets are exempt: pinning evaluates the exact joint score at the observation, no window exists. Requires reverseKL (mmd/w2 do not factor; teaching error); excludes `family=rotation`, `record=`, `into=`; the flow applies via `# (y0 \| x0)` on the SAME conditioned ensemble (hard pin). `examples/sbi_svgd.liu`. **Family transport — spread pins** (2026-07): a NON-degenerate pinned marginal conditions each particle on its **own** observation — one run transports between conditional distributions: `#` returns the paired cloud `(y, x∞)` whose Y-marginal is the frozen pinned draw **exactly** and whose slices are the target's conditionals. The repulsion then switches to the **conditional** score ∇_x log q̂(x\|y_c) — the joint-KDE read of the same log theorem, weighting each particle's repulsion by the pin-channel kernel `k_hy(y_j, y_c)` at the library's likelihood-role scale, so both terms condition at one resolution (pooled repulsion measurably fails twice on the linear-Gaussian closed form: slices collapse, sd 0.16 vs 0.351, and the marginal bulk overshoots the slice response, 1.89 vs 1.096). A point-mass pinned block keeps the exact pooled path — one slice, pooled IS conditional. `examples/cond_family.liu` (closed-form audit: slope 1.12, slice sds 0.35–0.38). — **Or an ensemble conditioned on an OBSERVATION SET** `(q \| Obs)` (2026-07, KL decomposition): `q` a plain parameter-space Dataset (all rows move), `Obs` a Dataset — or a destructured kernel-joint block — whose **columns** are the observations Y₁..Y_N. The N-observation KL splits by Bayes + log-likelihood additivity, `KL[q‖p(·\|Y₁..N)] = Σᵢ KL[q‖p(·\|Yᵢ)] − (N−1)·KL[q‖prior] + c`, and the additivity is assembled in the **log-weights** of the target library (one attraction, one repulsion, one bandwidth — a field-level signed sum measurably diverges; spec §10.10 records the three failure modes). Target must be joint samples `(y; w)`, observed rows first (e.g. `P = (K \| prior) ~ n`); same exclusions; applies via `# (q \| Obs)` (hard pin on both the ensemble and the observation set). **Return shape**: a ONE-observation set is the same object as a single pinned observation, so `#` returns the pinned **joint** (observation rows tiled over the moved ensemble, destructurable) — `plot Post` shows the posterior as the slice line at the observation, `plot P, Post` on the library cloud, exactly like the sibling `(y0 \| x0)` mode; with N > 1 there is no single row to tile, so the parameter **ensemble** comes back alone (its 1-D marginal is `plot_signal`'s job). `examples/sbi_nobs.liu` (exact-posterior check), `examples/manifold_local.liu` (the N-observation door). |
| `time=` | 3 | horizon used by `reverse` |
| `metric=` | `w2` | the geometry — and since 2026-07 it genuinely dispatches: `reverseKL+w2` → normalized NW field, `reverseKL+stein` → exact SVGD update. `stein` is score-driven only (`mmd`/`w2` descents reject it); `fisher_rao` reserved. |
| `family=` | `free` | the **manifold the path moves on** (2026-07). `free` = all of W2 space. `rotation` = the constrained steepest descent on the orbit `{R # from : R ∈ SO(d)}` — the curve is a curve of rotations, so its field is the so(d) projection and `flow` steps by `exp(lr·Ω)`, both **theorems of the path** (see §8). **`rotation(block=L)`** constrains further to `{R ⊗ I_L}` — one **channel** rotation shared across the L lags of `window()`ed data (the process-ICA search space); the projection becomes the skew part of the **lag-averaged** moment, exact when the lag-pooled channel covariance is ~I (whiten the channels *before* windowing). `metric` and `family` are orthogonal axes: metric picks the ambient geometry (which field gets projected), family picks the submanifold. If `from=` is a Dataset whose (lag-pooled channel) covariance is not white, the declaration prints a reachability note right here. Writing `family=` on `field()` is a teaching error pointing back to this slot. |

## 8. `field(path, estimator=...)`

One extractor, two estimator families by path kind:

**Declared path** (`prob(...)`):
`field(pt, estimator=regress(net[, base=v]), steps=6000, lr=0.001, batch=128)`
— or **`estimator=regress(rotation)`**: restrict the regression's
hypothesis class to the skew-linear fields `{x ↦ Ω(t)x}`. A declared
path cannot be re-curved (contrast `family=` on `descent`, which
changes the IVP itself); what can be constrained is the **estimator**,
and for this class the least squares is **closed form** per time slice
(no net, no SGD: with `C(t) = E[x_t x_tᵀ] = U diag(λ) Uᵀ` and
`M(t) = E[dx/dt · x_tᵀ]`, the minimizer is `Ω′ᵢⱼ = (M′ᵢⱼ−M′ⱼᵢ)/(λᵢ+λⱼ)`
in the eigenbasis; 33 slices, `batch=` points each, default 384).
`flow` integrates it on the group (exp steps), so the map is exactly a
rotation and **`inv` is free** (the transpose; roundtrip exact).
**`regress(rotation(block=L))`** tightens the class further to the
block-Kronecker rotations `{(Ω ⊗ I_L)x}` — one **channel** rotation
shared across the L lags of `window()`ed data (process-level
constrained transport). The closed form survives verbatim: the moments
are **lag-pooled** down to channels × channels (only same-lag products
enter, so within-window autocorrelation drops out of `C̃`), and the
same eigenbasis formula solves a C×C problem. Unlike the descent-side
`rotation(block=)`, the λ weighting is exact — no whiteness assumption
(whitening remains an orbit-reachability requirement, not a formula
requirement). No `base=` (nothing to train a residual against).
**Conditional paths** (2026-07, SICA line): law-gate indices join the
closed-form fit as free covariates — the per-slice regression is the
unconstrained linear LS on `[x; conditions]`, and the knot keeps only
the **skew x-part** of the coefficient: the conditions are consumed at
fit time, the field stays skew-linear and self-contained (flow pushes
plain measures, `inv` free). Two measured decisions: the skew-
constrained LS on context-residualized moments buries the signal
(stalls at MCC 0.79 where skew-of-unconstrained reaches 1.000), and
free linear maps applied as the refinement drift into the per-channel
FILTER ambiguity of the self-sufficiency objective. Declare the path
from the decoupled product INTO the current windows — the demixing
rotation is the reverse of the entangling transport (measured). This
is SICA Algorithm 1 (arXiv:2512.00665) restricted to instantaneous
rotations, per-element conditioning and all: `examples/sica_cond.liu`
(Gaussian AR wall, MCC 0.70 → 1.00, closed form, ~0.2 s). Not
combinable with `rotation(block=L)` (the contexts already carry the
lags). **`rotation(net=mlp(…))`** — the NEURAL member: the conditional
field is trained by ordinary conditional flow matching (net dims
`element-dim + condition rows -> … -> element-dim`; contexts enter the
net, not as linear features; `steps=`/`lr=`/`batch=` apply), and the
map is its per-slice **so(d) projection** — the average
element-Jacobian by central differences (contexts held at their drawn
values), skew part kept. For a linear net the Jacobian IS the
coefficient, so this generalizes skew(A_x) verbatim, and the map class
still cannot filter. `examples/sica_cond_net.liu` (same wall, MCC
0.70 → 0.999, ~9 s). Also not combinable with `block=`.
**Measured applicability boundary (2026-07, scratch probes — the
"net wins on nonlinear dynamics" conjecture is REFUTED for this
extraction)**: the projection consumes only the AVERAGE element-slope
E[∂v/∂y], and by Stein's lemma the linear OLS coefficient equals that
average derivative for near-Gaussian inputs — so within this family
the net's extra capacity is discarded by the projection. Measured on
SETAR-vs-AR pairs (linear mixing, nonlinear source dynamics): channels
with DIFFERENT average slopes — linear already opens the wall (0.999;
net ties at 0.991); channels with MATCHED average slopes (equal lag-1
autocorrelation, differing regime structure) — BOTH members are blind
(0.72–0.82; slope heterogeneity is invisible to an average), and the
skewed-context Stein-bias window is too small to open anything
(0.786 vs 0.756). A genuine net advantage requires a different
extraction (beyond the average Jacobian) or the unrestricted neural
refinement — the remaining ledger candidate. (`examples/sica_rf_rotation.liu` — constrained
transport buys back exact recovery, MCC 1.000, where the free
transport reaches 0.984; `examples/sica_rf_process.liu` — the Gaussian
AR wall holds for the marginal transport, MCC ~0.72, and the
block-Kronecker class opens it, 1.000.)
- trains `net` to regress `E[dx_t/dt | x_t]` (L² — flow matching);
  streams `(step, loss)`; returns a self-contained Field.
- **`base=v`** (residual estimator): freezes a pretrained self-contained
  field and regresses only the correction; the L² target becomes
  `dx/dt − v(x_t, t)`. Returned as the combination `1*[v] + 1*[Δ]`, so
  field algebra cancels the base's weight exactly. Checks: `base` must
  be self-contained, of matching dimension, and transport the same t=0
  marginal as this path. Not yet combinable with conditional paths.
- **Conditional paths** (any `|` term): net dims must be
  `(y-dim + condition rows) -> ... -> y-dim`; the resulting field/map is
  conditional — it pushes kernels, not measures, and stays out of field
  algebra for now.

**Descent path**:
`field(qt, estimator=nw(kernel=rbf))` — or `field(qt)` bare for a `w2`
descent (`estimator=sinkhorn` accepted for emphasis) — or
**`estimator=dsm(mlp(d -> … -> d), sigma=, warm=16, steps=1500,
trainlr=0.001, batch=128)`** (2026-07): both scores of the reverseKL
field estimated by fixed-σ **denoising score matching** instead of
kernels. The target joint's ε-net is trained once (`steps=`); the
moving cloud's net is **warm-started** — `warm=` SGD steps per flow
step — so it tracks q_t (a frozen q-net is the static field v₀,
measured to overshoot and collapse). The field is the plain score
difference `(ε̂_q − ε̂_p)/σ`; pinned rows (`from=(x|y)`) condition both
terms through the joint score — no likelihood windows, no bandwidth
rules; `sigma` is the ONE smoothing scale, shared by both nets so the
bias cancels at the fixpoint (the neural mirror of same-kernel-both-
sides). σ replaces the bandwidth, it does not disappear. **Tuning rule
(measured)**: the flow must not outrun the warm start — if the cloud
comes out ragged or leaks *outward*, first lower `lr` (gentle steps the
q-net can track), then raise `sigma` (nearly free in bias: the fixpoint
is q∗N_σ = p∗N_σ, and deconvolution is unique — measured on
uniform→gaussian: σ 0.15→0.5 + lr 0.1→0.05 takes the mean bias 0.15→0.00),
then raise `warm=`. Residual weakness vs nw on kernel-friendly problems
(low-d, unimodal, plentiful samples): neural scores extrapolate weakly
*off*-data where the KDE score always pulls home — a few % of tail
stragglers survive (17/300 beyond 2.5σ vs nw's 0/300 on that toy). Use
nw there; dsm's regime is conditioning and dimension. Requires
reverseKL + a **Dataset** target (a score target needs no p-net — use
nw) + `metric=w2`; no `(q|Obs)` yet, no rotation, no `record=`. Unlike
nw, this descent **consumes RNG every step** (batch picks + noise) —
it sits in the program's stream like any training. Closed-form audit:
`examples/cond_family_dsm.liu` (slope 1.076 vs 1.096, slice sds
0.349–0.361 vs 0.351; frozen-q control: 1.305 / 0.11).

The form of the field is a **theorem of the divergence and the metric**,
not a switch (`normalize=` was removed 2026-07):

| divergence | metric | field |
|---|---|---|
| `reverseKL` | `w2` (default) | normalized NW: consistent estimate of the W2 velocity `∇log(p/q)` |
| `reverseKL` | `stein` | the exact SVGD update (unnormalized; steepest descent in the kernelized Stein geometry) |
| `mmd` | `w2` | the witness gradient `∇(p̂ − q̂)` — the MMD flow (unnormalized, exact plug-in) |
| `w2` | `w2` | barycentric displacement of the entropic plan (Sinkhorn; no kernel) |
- ONE form covers every case — a smoothed score difference, re-estimated
  from the ensemble at every simulation step:
  `Φ(y) = smooth(∇log p)(y)/Z_p(y) − smooth(∇log q)(y)/Z_q(y)`.
  The q-term is always the KDE score of the moving ensemble; the p-term
  depends on what the target provides:
  - **score** (Gaussian, mixture, `unnormalized`): the true score,
    kernel-smoothed over the ensemble points — score evaluations never
    depend on the query point, so recorded replay stays self-contained;
  - **samples** (Dataset target): the KDE score of the target samples —
    same kernel and bandwidth on both sides, so the two smoothing biases
    cancel exactly where `p̂ = q̂` (the field has the right zeros even at
    finite bandwidth).
- unnormalized forms (`reverseKL+stein`, `mmd`): `Z` = point count;
  bandwidth by the Liu & Wang 2016 heuristic
  `h = med{‖xᵢ−xⱼ‖²}/log(n+1)`.
- normalized form (`reverseKL+w2`): `Z` = kernel mass at the query
  point — the genuine NW interpolation, a consistent estimator of the
  W2 velocity `∇log(p/q)` (Liu, Yu, Simons, Yi & Beaumont 2024, eq. 5);
  bandwidth by the Gretton median heuristic `σ = med{‖xᵢ−xⱼ‖}`
  (`h = 2·med{‖xᵢ−xⱼ‖²}`), no log-n scaling.
- Dataset targets pool the ensemble with the target samples before
  taking the median (both sides must share one kernel). The field's
  form travels with `record=true` replay and inversion.

**Constrained paths** (`descent(..., family=rotation)`, 2026-07): the
constraint lives on the **path**, and the extractor inherits it — the
constrained curve's true velocity *is* the L²(q) projection of the free
descent field onto the orbit's tangent space, the linear fields
`{z ↦ Ωz : Ω ∈ so(d)}`:
`M = (1/n)·Σᵢ v(zᵢ)zᵢᵀ`, `Ω = (M − Mᵀ)/2` (exact when `E[zzᵀ] = I` —
hence `whiten`). The estimator stays `nw` — it estimates the *ambient*
field; the projection is exact linear algebra, not estimation. On a
white ensemble Ω is the natural-gradient direction of Amari's ICA, so
a rotation-constrained descent of `reverseKL(decouple(Z) ~ n)` *is*
continuous-time natural-gradient ICA as a constrained WGF
(`examples/sica_rotation.liu`: MCC 0.77 → 1.00 in 8 rounds). `flow`
integrates on the group — `z ← exp(lr·Ω)·z`, exact rotations, so volume
and covariance are conserved to the bit, not to first order (a Euler
step `z += lr·Ωz` would inflate norms every step, since `Ωz ⊥ z`).
Estimation note: the projection averages n·d field values down to
d(d−1)/2 numbers, so it uses the **sharp** Liu & Wang bandwidth even
for the normalized field — a wide interpolation bandwidth Gaussianizes
the KDE scores, and the skew moment of a linear field is zero (the
rotation signal lives in the higher cumulants).
Constraints: `record=` and `into=` are teaching errors (the whole map
is one d×d orthogonal matrix — nothing to record or amortize);
`family=` on `field()` is a teaching error pointing to `descent`.
- Note: `steps`/`lr` here belong to `flow`, not `field` (unknown
  keywords are currently ignored silently).

## 9. `flow(v, steps=50, lr=0.5, record=false[, into=net, batch=128, trainlr=0.001, project=instant])`

Integrates a Field into a Map. Three representations for descent maps —
simulate (particles), record (memory), amortize (optimizer):

- Self-contained fields (`regress`, `reverse`, `transport`): explicit
  Euler over `t ∈ [0, 1]` in `steps` steps; `lr`/`record` unused.
- Descent fields: `steps` interacting-particle updates of size `lr`,
  re-estimating the field each step (mean-field simulation).
  `record=true` stores the per-step ensembles and bandwidths on the
  first forward application — the map becomes replayable at arbitrary
  points and hence invertible.
- **`into=mlp(d -> ... -> d)`** (descent fields only; spec §10.6): the
  amortized flow — a **drifting model**. Trains a one-step generator:
  each optimizer step samples a latent batch `z` from the path's
  `from=`, evaluates `y = net(z)`, estimates the same nw field on the
  generated batch, and regresses `net(z)` onto the frozen target
  `y + lr·Φ(y)`. The pushforward evolves across optimizer steps;
  inference is one forward pass (1-NFE). Under `into=`: `steps` =
  optimizer steps (default 2000), `lr` = drift step ε, `trainlr` = the
  net's Adam rate, `batch` = latent batch. The net has **no time
  input** (the evolution lives in training). The streamed loss equals
  `ε²·mean‖Φ‖²` — the zero-flow diagnostic; it stalls exactly at
  distribution match. Constraints: `record=true` is mutually exclusive;
  the resulting map is **hard-pinned** to `from=` (no record escape)
  and **non-invertible** (see `inv`). Caveat (measured): amortization
  is mode-seeking on well-separated modes — the particle simulation of
  the same field covers modes the amortized generator drops.
  **Exception — the amortized instantaneous demixer** (SICA Level 1): a
  *conditional* descent whose ensemble carries window provenance
  (`from=(E | C)` built by `lagsplit(window(Z, L), L)`, target the
  permuted product, `estimator=dsm` required) trains `g(x) = x + net(x)`
  (net dims `d -> ... -> d`, d = channels) by **structured
  re-simulation**: each outer step pushes the source trajectory through
  `g` per column, rebuilds windows and a fresh permuted product from
  `g`'s own output, estimates the conditional score field there (fresh
  dsm pair, full pretrain), Rao-Blackwells it onto the elements, and
  regresses the net to convergence on the frozen drift target
  (**two-timescale**; the merged one-step form measurably walks off
  along the per-channel-reparametrization valley). `steps` = outer
  field refreshes (~32). The result is a plain instantaneous map of
  `R^d` — **no from= pin** (holdable and reusable is the point), still
  no `inv`. `sica_amort.liu`: 0.73 → 0.90 → 0.99, one net.
- **`project=instant`** (conditional descent fields, `from=(y|x)` only):
  every particle step is projected onto an **instantaneous** map before
  it is applied. The conditional field is evaluated as usual — each
  element at its own pinned context, conditional information fully
  consumed — then the batch of per-column field values is regressed
  onto the element values alone (closed-form ridge on a degree-3
  polynomial in bounded coordinates `u = 2·tanh(z/2)`; raw polynomials
  eject the tails, measured), and the step uses the fitted `ŵ(z)` for
  every column. Each step is one shared map of `z` only, so the
  composed demixer stays in the instantaneous class of the SICA
  identifiability theorem: *contexts enter the objective, not the map*.
  The projection is the Rao–Blackwell average `ŵ(z) ≈ E[v*(z,c) | z]`
  — not the context-blind marginal field, because elements correlate
  with their own contexts. Deterministic, no extra RNG. Teaching
  errors: unconditional fields (already context-free), `(q|Obs)`,
  `family=rotation` (the rotation members are already instantaneous),
  `into=` (Level 1, reserved).
- Applying a map records ~12 trajectory frames on the output Dataset
  (2 frames for a one-step generator: latents and output).

## 10. `inv(T)`

Pointwise inverse of a Map. The price list:

1. **Free** for maps of self-contained fields — integrate backwards.
2. **Memory** for descent maps: legal only under `record=true` after one
   forward application; each step solves `y = x + lr·Φ_t(x)` by
   fixed-point iteration (8 sweeps). Foreign measures are legal with a
   printed note (see `#` above).
3. **Training**: `reverse(...)` upgrades a descent path to a
   self-contained field, after which tier 1 applies.

An amortized map (`flow(..., into=net)`) rejects `inv` unconditionally:
the Euler chain was dissipated into the weights, there is no trajectory
to run backwards, and the net need not be injective — buy the inverse
with `record=true` (memory) or `reverse` (training) instead.

An unrecorded descent map rejects `inv` outright (negating the field
would simulate gradient *ascent*, a different system).

## 11. `reverse(qt, estimator=denoiser(net), steps=4000, lr=0.001)`

Learned time reversal of an *implicit* (descent) path via denoising
score matching (OU forward, closed-form marginals). Returns the
probability-flow ODE Field (self-contained → free `inv`, enters field
algebra). Prototype constraints: the divergence target must be a single
Gaussian (the stationary law); `from=` must be a Dataset (it is what
gets noised); batch is fixed at 128. Declared paths need no `reverse`
(substitute `t → 1-t` in the formula); Maps use `inv`.

## 12. `transport` (legacy sugar)

```liu
v = transport from A to B using mlp(2 -> 64 -> 64 -> 2) for 5000 steps with lr=0.001, batch=128
```

Sugar for `field(prob(t*B + (1-t)*A), estimator=regress(net))`.
Defaults: 5000 steps, lr 1e-3, batch 128. `for`/`with` clauses optional.

## 12b. `kernel(param) { name = expr; ... }` — programmable Markov kernel (spec 10.10)

Registers a straight-line **program** as a Markov kernel p(y | param) —
the escape hatch from the bundled-joint menu (a simulator may reference
values the program computed, e.g. a trained Map). The body is defined
POINTWISE in the parameter and executed **vectorized** at instantiation;
the LAST binding's value is the output y.

**Draw level** (inside the body only): names denote blocks of draws
(columns = draws); arithmetic `+ − * /` is elementwise with 1-row
broadcast (the `W·x` shape); `sin/cos/exp/log/sqrt` apply elementwise;
Distribution values **auto-lift to fresh draws** (same rule as path
formulas — the same canon twice is a teaching error: name the draw and
reuse it); `T # z` applies a trained transport per draw (WG/descent maps
refuse: they re-simulate, they are not pointwise functions). No `~`
(the body is one draw at a time), no `|`/`or` (no measures at draw
level), no nesting, no data-dependent branching.

**Bounded `for` inside bodies (2026-07)**: at draw level `for i in
1..K { ... }` is an **expression** — its value is the row-stack of each
round's final binding (an N-observation block is ⊕ᵢyᵢ). Macro
expansion, literal bounds, cap 64, no nesting; the lift scope resets
per round (each round's distributions are **fresh draws** — iid,
literally; quant-verified in `kernel_probes.liu`). The index is a
per-round scalar.

**Stage-3 primitive (2026-07)**: `rows(a, lo, hi)` — slice a draw
block's rows `[lo, hi)` (0-based, half-open, **literal** integer bounds —
the checker sees every block dimension statically, body-for's
discipline). This is the door to **matrix-valued parameters**: a d×m
matrix travels as a dm-vector whose columns are cut out by `rows`, and
`W·x` is written column by column with the existing 1-row broadcast —
`z = w0 + rows(w,0,3)*rows(x,0,1) + rows(w,3,6)*rows(x,1,2)` is Def. 1
at m = 2 (`manifold_torus.liu`; identity probes in
`kernel_probes.liu`). Out-of-range and non-literal bounds are teaching
errors; scalars have no rows.

**Stage-2 primitives (2026-07)**: `dot(a, b)` — per-draw inner product,
a 1-row block of column-wise dots (the reduction projections are made
of: perp of e against u = `e - u*(dot(u,e)/dot(u,u))`); `jvp(T, z, v)` —
the trained map's Jacobian at z applied to direction v, per draw
(central finite differences through two map applications, h = 1e-3 —
the same discipline as the path-coefficient derivative; exact for
linear maps up to rounding). **Single-column Datasets enter bodies as
constant blocks** (one column broadcast to every draw) — how a computed
point like `w0 = inv(G) # Y0` gets into a simulator. The manifold
latent variable model of Khoo, Liu & Beaumont 2026 is the acceptance
example (`manifold_local.liu`); `kernel_probes.liu` is the
self-verifying identity suite for all of the above.

**Instantiation = the §10.8 kernel surface, zero new syntax**:

| form | meaning |
|---|---|
| `(K \| [w]) ~ n` | run the simulator at a fixed parameter (the sequential-SBI primitive) |
| `(K \| prior) ~ n` | the joint (y; param), observed block first — feed it to any engine |
| `(K \| X) ~ n` | parameters resampled from a Dataset's columns |
| `(y, w) = (K \| z) ~ n` | destructure: freeze ONE joint table of n draws; the two names become its (y; w) row blocks (`~ n` required — an instantiation is a law, not a table) |

Samples only — a programmable kernel has no density and no score
(score-needing paths give teaching errors); `reverseKL((K | prior) ~ n)`
is the likelihood-free route. Reproducibility: body draws consume the
global RNG in statement order; the body's canonical printout is the
kernel's identity. `examples/sbi_kernel.liu` re-derives the
sine_gaussian toy as user code and runs the conditional-descent
posterior on it.

## 13. `mlp(d0 -> d1 -> ... -> dk)`

Network skeleton: ReLU hidden layers, linear head; the input gains one
extra time dimension automatically. For fields, `d0` and `dk` must both
equal the data dimension.

## 14. Math functions

`sqrt exp log sin cos` — unary, on numbers, t-expressions, or
coordinate (`x1`/`x2`) expressions only.

## 15. Verbs

- **`seed n`** — seeds the single global RNG stream. Determinism is
  program text + seed → bit-identical output (single-threaded CPU
  backend); statement order matters because all draws share one stream.
- **`plot e1, e2, ...`** — scatter, any number of comma-separated
  series on one figure; each series anything `~`-able to a Dataset.
  A lazy pushforward is auto-sampled at 500 points. Labels come from
  provenance. **3-D (2026-07)**: when EVERY series is an eligible 3-D
  **cloud** — exactly 3 rows forming one coordinate block — the figure
  goes 3-D: the terminal draws a fixed orthographic projection (azimuth
  0.7, elevation 0.5) and the NDJSON event carries `[x,y,z]` triples,
  which the playground renders with **drag-to-rotate** and depth-cued
  alpha (a 2-D manifold reads as a surface). 2-row series stay classic;
  kernel **joints** of any row count (they are bundles, not clouds — a
  1-D output + 2-D parameter joint has 3 rows too) keep the rows-0,1
  view — destructure or `prob()` a block to see its 3-D marginal.
  Trajectories of 3-row flows project at the fixed view.
- **`plot trajectory of e`** — animation of a Dataset that carries a
  trajectory (the output of a flow, `via svgd`, or a sampled
  pushforward); errors otherwise.
- **`plot_signal e1, e2, ...`** — the **signal/waveform view** of the
  same data `plot` shows as a cloud: columns are drawn in **index
  order**, and each coordinate row of a Dataset becomes one line over
  `i = 0..n-1` (a d-dim Dataset yields d lines, labeled `prov[r]`).
  A bare Distribution or lazy pushforward samples 500 i.i.d. points
  first (order = draw order — honest noise); an RV is a teaching error
  (`plot_signal prob(x)`). Column order is deterministic and preserved
  by `whiten`, rotation flows, and map application, so paired series
  stay aligned across statements. The view is a **window of the first
  200 samples** (a zoom, not a decimation — decimation would alias
  waveforms); the footer and the web card state `of n` honestly when
  truncated. Terminal rendering downsamples oscilloscope-style (per
  character column, the min–max band of the samples it covers — a
  bimodal source reads as two rails); the `LIU_DUMP` event is `signal`
  with per-series `values` (the same first-200 window; `total` carries
  the full length), rendered as polylines in the playground.
  (`examples/signals.liu`)

## 16. Limits

| limit | value |
|---|---|
| samples per Dataset (`~ n`) | 100 000 |
| auto-sample for plotted pushforwards | 500 |
| trajectory frames kept | ~12 |
| playground (server-side) | wall-clock timeout, memory cap, program ≤ 16 KB, concurrency cap — see `web/server.py` |
