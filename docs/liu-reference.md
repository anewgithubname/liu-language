# Liu (流) — Language Reference

*Reference for the v0.3+ prototype. The name means "flow" — measure flow,
gradient flow, probability flow — which is the entire axis of the language.
For a three-minute overview see `docs/liu-cheatsheet.md`; for the complete
per-function/operator usage tables see `docs/liu-api.md`; for the design
history and roadmap (in Chinese) see `docs/liu-spec.md`.*

---

## 1. What Liu is

Liu is an **executable notation for measure transport**: a small language
for describing machine-learning experiments — flow matching, diffusion,
Stein variational gradient descent, and their relatives — as constructions
on probability measures. It is a demonstration front end for the
[Juzhen](https://github.com/anewgithubname/Juzhen) C++ matrix/NN backend.

Four design goals govern every decision:

1. **Sandboxable.** The interpreter is the security boundary: the language
   has no file, network, or environment primitives, and is not Turing
   complete. Data enters only through built-in distributions.
2. **Auditable.** A program is a handful of lines a human can read before
   running.
3. **Reproducible.** A program's text and its `seed` determine the output
   bit for bit (single-threaded CPU backend).
4. **Second-scale feedback.** Interpretation is immediate; training loops
   stream their loss as they run.

A program is a straight-line script — no user-defined functions, no
conditionals, no loops. Every program is therefore a finite dataflow DAG:
dimensions, capabilities, and provenance are all checkable before any
computation runs, and every value remembers its construction (its
*provenance*), which powers both trajectory plotting and safety checks.

### 1.1 The one pipeline

```
flow( field( path ) ) # μ
```

- A **path** is a curve `t ↦ p_t` in the space of probability measures.
  It can be **declared** by an interpolation formula, or **generated** as
  the gradient-flow of a divergence (`descent`). (A third construction,
  boundary-value *bridges*, is on the roadmap.)
- **`field`** extracts a velocity field that realizes the path.
- **`flow`** integrates the field into a map; **`#`** pushes a measure or
  a point set through the map.

Flow matching, diffusion, and SVGD are all instances of this sentence;
they differ only in how the path is constructed and how its field is
obtained.

---

## 2. Lexical structure and program form

- One statement per line; `//` starts a comment.
- Statements are **bindings** (`name = expression`), **destructuring
  binds** (`(y, x) = rv(joint)` — one joint draw, two coordinate blocks),
  or **verbs** (`seed`, `plot`).
- The first statement should be `seed n`; if omitted, the interpreter
  inserts `seed 0` and says so.
- Reserved symbol (cannot be bound): `t` (time, used in path formulas).
  `x1`, `x2` are the coordinate symbols of `unnormalized(...)` but are NOT
  reserved — a user binding shadows them (the coordinate reading applies
  only while the name is unbound; errors hint when shadowing bites).
- Arithmetic: `+ - * /`, unary minus, parentheses, and the unary
  functions `sqrt exp log sin cos`. Arithmetic acts on numbers, on
  `t`-expressions, on coordinate expressions, and (as described in §5) on
  random variables.
- Vector literals: `[a, b, ...]`. Layer-size chains: `2 -> 64 -> 2`.

## 3. Values and their capabilities

| Type | What it is | Constructed by |
|---|---|---|
| `Distribution` | a probability measure (lazy) | `gaussian`, `moons`, …, mixtures (`or`), `unnormalized`, `T # μ` |
| `Dataset` | a finite sample = an empirical measure; carries provenance | `μ ~ n`, `T # X` |
| `Net` | an untrained network skeleton | `mlp(d0 -> … -> dk)` |
| `Divergence` | a functional `D[p, q]` on pairs of measures | `reverseKL(p[, q])` |
| `ProbPath` | a declared path: the law of a written process | `prob(formula)` |
| `DescentPath` | an implicit path: a gradient-flow IVP | `descent(D, from=, time=, metric=, family=)` |
| `Field` | a time-dependent velocity field on ℝᵈ | `field(path, estimator=…)`, `reverse(…)`, `transport …` |
| `Map` | the integral of a field | `flow(v, …)`, `inv(T)` |
| `Kernel` | a transported conditional family with open condition slots | `T # (y\|x)`; instantiated by `\| W` (§7.6a) |

Capabilities decide which operations are legal, and are checked before
computation with messages that state the violated fact:

- A Distribution may or may not admit an **exact sampler**, and may or may
  not have a **score** `∇log p`. Toy distributions have both; Gaussian
  mixtures have both; `unnormalized(L)` has a score but **no** sampler.
- A Map may or may not be **invertible** (see §7.3).
- Datasets carry **provenance** — the canonical description of the measure
  they were drawn from — used by the hard checks of §7.4.

## 4. Distributions

| Primitive | Notes |
|---|---|
| `gaussian(mean, std=1)` | isotropic; mean is a vector, e.g. `[0, 0]` |
| `uniform(lo, hi)` | box; two vectors |
| `moons(noise=0.1)` | two interleaved half-circles |
| `ring(r=2, noise=0.1)` | circle with radial noise |
| `spiral(turns=2, noise=0.1)` | |
| `w1*A or w2*B or …` | mixture; unweighted components are equally weighted |
| `unnormalized(L)` | see below |

**Sampling.** `μ ~ n` draws `n` points into a Dataset (`n ≤ 100 000`).
On a Dataset, `~ n` subsamples with replacement. On a lazy pushforward
(`T # μ`), `~ n` samples the base and transports.

**`unnormalized(L)`** takes a log-density written in the coordinate
symbols `x1`, `x2`, defined up to an additive constant:

```liu
banana = unnormalized( -x1*x1/8 - (x2 - x1*x1/4 + 1)*(x2 - x1*x1/4 + 1)/0.4 )
```

Its score is `∇L` — the partition function cancels under the
log-gradient, which is exactly why score-based inference (SVGD) applies
to Bayesian posteriors. Since it admits no exact sampler, `banana ~ n`
is rejected; write `banana ~ n via svgd(...)`.

## 5. Declared paths: formulas, random variables, `prob`

A declared path is written as an **interpolation formula** — an affine
combination of random variables with time-dependent coefficients:

```liu
xt = t*data + (1-t)*noise             // rectified-flow interpolant
xt = sqrt(1 - t*t)*e + t*data         // variance-preserving (diffusion) interpolant
```

Semantics:

- The formula denotes a stochastic process `x_t = Σᵢ cᵢ(t)·ξᵢ` where each
  `ξᵢ` is a random variable and each `cᵢ` is a function of `t`.
- **Auto-lifting.** A Distribution or Dataset that occurs **once** in the
  formula is lifted to a fresh, independent random variable. This makes
  the common case zero-ceremony.
- **Draw identity.** A distribution occurring **twice** is ambiguous —
  the same draw, or independent copies? (`X − X = 0` only for the
  former.) The checker rejects it and asks for `rv()`:

  ```liu
  x = rv(data)                        // one named draw
  yt = t*x + (1-t)*x                  // same draw twice: coefficients merge, yt = x
  ```

  Distinct `rv(...)` calls are independent; reusing a bound name reuses
  the draw. All RVs in a formula are mutually independent by default.
- **Couplings** (spec §10.3). `couple(A, B, via=...)` is a joint law over
  *pairs* whose marginals are A and B; a draw samples both sides and
  pairs them **within the batch** — `via=independent` (keep draw order),
  `via=ot` (exact Hungarian matching, deterministic given the draws), or
  `via=sinkhorn(eps)` (partner sampled from the entropic plan). The two
  blocks are peer coordinates of one draw:

  ```liu
  (x0, x1) = rv(couple(noise, data, via=ot))
  xt = t*x1 + (1-t)*x0            // trains on OT-matched pairs: OT-CFM
  ```

  The inverse noun exists too: **`decouple(X)`** takes an empirical joint
  and returns the product of its marginals (any dimension) — same
  marginals, dependence forgotten. Iterating a KL descent toward
  `decouple(Z)` is the SICA unmixing loop (`examples/sica.liu`), and the
  same product-measure trick drives MIRI-style imputation. The ICA toy
  `mixed_sources(m)` is itself a pair-joint over (observations, true
  sources): `(X, S) = rv(mixed_sources(0.7)) ~ n` freezes one draw with
  the ground truth riding along, so unmixing quality is *checkable*
  rather than eyeballed. Its time-series sibling `mixed_signals(m)`
  draws **one trajectory** per draw (sin + sawtooth, incommensurate
  periods so the source cloud is independent by ergodicity; random
  phases are the only randomness) — with `plot_signal`, the unmixing
  demos become waveforms-in, waveforms-out
  (`examples/sica_signals.liu`).
  Both blocks may appear as separate terms (their pairing is the point);
  the pair identity lives on the couple, so `rv()` wraps the couple, not
  its marginals. Appending `~ n` to the bind freezes ONE coupling of n
  pairs (matched once over n, subsampled with pairs intact) instead of
  re-matching per batch — minibatch-OT bias vs frozen-sample bias, one
  `~` apart. With an entropic coupling plus a Brownian-bridge noise term,
  `xt = t*x1 + (1-t)*x0 + sqrt(t*(1-t))*sigma*z` is a Schrödinger bridge
  (`examples/schrodinger_bridge.liu`) — no new machinery. Finally,
  `couple(base, T # base, via=paired)` pairs each z with its own image
  T(z) — the reflow coupling (`examples/reflow.liu`).
- **Bounded `for`** (spec §10.1). `for k in 1..K { ... }` with literal
  bounds (≤ 64 iterations) is the only loop — macro-expansion semantics,
  no `while`/`break`/`if`, no data-dependent bounds. Rebinding inside
  the body iterates an object; every emitted event carries the iteration
  stack `iter=[.., k]`, and the playground groups losses and plots per
  round. One round of reflow (paired re-coupling + retraining)
  straightens a flow enough that `flow(v, steps=1)` samples in one
  Euler step.
- **`prob(xt)`** takes the **law** of the process. For a t-dependent
  formula this is a `ProbPath` — a curve of measures, consumed by
  `field`. For a **t-free single draw** the law is a plain measure and
  `prob` returns it as one: `prob(rv(D))` is `D`, and for a couple
  block `prob(x0)` is its **marginal** — samplable and plottable
  (`plot prob(x0) ~ 400, noise ~ 400` visualizes that a coupling never
  moves its marginals). This is the one gate between the two worlds:
  random variables carry draw identity and are never sampled directly
  (two separate sample statements could not say whether their draws
  pair — the ambiguity `rv()` exists to kill); `prob` is where that
  identity is deliberately, visibly forgotten. Mathematically x ~ X
  always means x ~ Law(X); the language just makes the Law(·) step
  explicit.

### 5.1 The field of a declared path

`field(pt, estimator=regress(net), steps=, lr=, batch=)` computes the
**conditional-expectation velocity**

    v(x, t) = E[ ẋ_t | x_t = x ],    ẋ_t = Σᵢ cᵢ′(t)·ξᵢ .

Three equivalent characterizations explain the design:

1. *(Probabilistic)* `v` is the L² projection of `ẋ_t` onto σ(x_t) — the
   best Markov summary of the process's motion.
2. *(PDE)* `v` satisfies the continuity equation ∂ₜp_t + ∇·(p_t v) = 0,
   so the ODE flow of `v` transports p₀ to p_t exactly. This is what
   justifies the pipeline `flow(field(prob(xt)))`.
3. *(Variational)* `v` is the unique minimizer of
   E‖w(x_t, t) − ẋ_t‖² over fields `w` — hence `estimator=regress(net)`:
   the training loss's population minimizer *is* the definition. Flow
   matching and denoising score matching are the same regression with
   different coefficient schedules.

Note the continuity equation alone under-determines the field (one may
add any p_t-divergence-free component); the conditional expectation is
the representative **selected by the coupling** of the process. Changing
the coupling changes the field.

**Training-class expressions.** `field(..., regress)`, `transport`, and
`reverse` fit persistent parameters and stream `(step, loss)` while they
run. Estimators like `nw` hold no parameters and re-solve at every step.

**Residual regression.** `regress(net, base=v_u)` freezes a pretrained
field and regresses only the correction: the target becomes
`ẋ_t − v_u(x_t, t)`, whose minimizer is `v − v_u`. The result is a
genuine field of *this* path, returned as the combination
`1*[v_u] + 1*[Δ]` so that field algebra cancels the base's weight
exactly (§7.1a). Checks: the base must be self-contained (descent
fields are rejected), of matching dimension, and must transport the
same t=0 marginal as this path — "same schedule, same noise endpoint"
holds by construction.
Rule of thumb: *whatever has a loss curve owns parameters; whatever has
no loss curve is re-estimated on the fly.*

## 6. Implicit paths: `descent`

```liu
qt = descent(reverseKL(target), from=cloud, time=3, metric=w2)
```

`descent(D, from=q0)` is the curve traced by **steepest descent of the
divergence `D[·]` in the geometry named by `metric=`**, started at the
initial measure `q0` — an initial-value problem in measure space.

`from=` also accepts a **conditioned block of a frozen joint**,
`from=(y0 | x0)` (2026-07): the ensemble is the joint, only the y-block
rows move, and the x-block stays pinned at its drawn values (e.g. an
observation). Pinning *is* conditioning for `reverseKL` — ∇_y log p(y|x)
= ∇_y log p(y,x), since log p(x) dies under ∇_y — so the free rows of
the joint-space nw field descend the **conditional** KL exactly
(conditional SVGD). The target stays a plain joint Dataset, and both
conditioning syntaxes lower to the **one** conditional-likelihood
estimator (2026-07 revision & unification): the NW regression of the
y-kernel onto the library's parameter rows, with the §10.10
likelihood-role bandwidths — here read as frozen per-particle library
weights (each ensemble column conditions on its own pinned block,
`(q | Obs)` instead sums the same regression's logs over its N
observations); the free-row smoothing keeps the pooled bandwidth,
pooled over the free rows only. Two assemblies were deleted on the
way (both measured on the sine toy): the pooled-joint-space field —
its pinned-row kernel was an ABC acceptance window calibrated to the
cloud's spread rather than the simulator's noise, ~7× too wide, 30%
of the posterior mass in valleys carrying 1% (basin weights right,
shape wrong) — and raw per-draw kernel weights, fine at N=1 but the
exchangeability trap at N>1. Score targets need no window at all:
pinning evaluates the exact joint score at the observation. This is
the descent engine's simulation-based-inference cell
(`sbi_svgd.liu`; the amortized transport twin is `sbi_npse.liu`). `reverseKL` only (the
mmd/w2 fields do not factor through the conditioning); excludes
`family=rotation`, `record=`, `into=`; the flow applies via
`# (y0 | x0)` on the same conditioned ensemble. The
initial measure is part of the object: the curve is determined jointly by
the divergence and its starting point, which is why `from=` is mandatory
and why the maps built from this path are pinned to it (§7.4).

`from=(q | Obs)` (2026-07): conditioning on an **observation set**. `q`
is a plain parameter-space ensemble (every row moves) and the columns of
`Obs` — a Dataset, or a destructured kernel-joint block — are N
observations of ONE parameter. Bayes plus log-likelihood additivity
split the target functional,
`KL[q ‖ p(·|Y₁..N)] = Σᵢ KL[q ‖ p(·|Yᵢ)] − (N−1)·KL[q ‖ prior] + c`,
and the additivity is assembled where KDE estimates can carry it: in the
**log-weights** of the joint library — `log Wⱼ = Σᵢ log L̂ᵢ(wⱼ)` with
each single-observation likelihood the NW regression of the y-kernel
onto the parameter rows — after which each step is a plain descent onto
the likelihood-weighted prior KDE (one attraction, one repulsion, one
bandwidth; the field-level signed sum `Σᵢvᵢ − (N−1)v₀`, exact for true
scores, measurably diverges or collapses for KDE ones — spec §10.10).
The target must be a joint sample library `(y; w)`, observed rows first
(`P = (K | prior) ~ n`); same exclusions as above; applies via
`# (q | Obs)`, hard-pinned to both the ensemble and the observation set.
The return shape follows the observation count: a ONE-observation set is
the same object as a single pinned observation, so `#` returns the
pinned joint — observation rows tiled over the moved ensemble,
destructurable, plotting as the slice line at the observation (the
sibling `(y0 | x0)` mode's contract); with N > 1 there is no single row
to tile and the parameter ensemble comes back alone (`plot_signal` for
its 1-D marginal).
`sbi_nobs.liu` checks the particle cloud against an exact Gaussian
posterior; `manifold_local.liu` uses it to open the single-observation
wall (32 neighbors identify the planted tangent direction).

- `reverseKL(p[, q])` is the divergence `KL[q ‖ p]` with target (static)
  slot `p`; the optional second slot names the moving measure and must
  agree with `from=` if both are given. For `reverseKL` the target may
  be a Distribution with a score, or a Dataset (the `nw` field then
  descends onto the KDE of the samples, §6.1). Two further divergences
  take Dataset targets (samples are their whole interface): **`mmd(data)`**
  — its W2 field is the witness gradient `∇(p̂ − q̂)`, exact for the
  empirical measures — and **`w2(data, eps=)`** — its field is the
  barycentric displacement of the entropic (Sinkhorn) plan; honest
  caveat: the W2 descent onto an empirical target *memorizes* it (the
  minimizer is the empirical measure; particles snap onto data points as
  `eps → 0`), where kernel descents generalize by smoothing.
  `forwardKL` remains a teaching error (needs the h-transform ratio
  estimate, Liu et al. 2024 Thm 2.1 — the locallinear family).
- `metric=` (default `w2`) makes explicit that "steepest" is only defined
  relative to a geometry — and since 2026-07 it genuinely **dispatches**:
  under `w2` the field of the KL descent is `∇log(p/q_t)` and the
  estimator is the normalized NW form; under `stein` (the kernelized
  Stein geometry) the exact gradient flow is the unnormalized SVGD
  update, and that is what runs. `stein` is score-driven only — the
  `mmd` and `w2` descents reject it. `fisher_rao` is reserved.
- `time=` sets the horizon used by `reverse` (§7.5).

### 6.1 The field of a descent path

```liu
v = field(qt, estimator=nw(kernel=rbf))
```

For an implicit path the field comes first — it *defines* the curve — and
`estimator=` names how it is evaluated from samples. `nw` is the
Nadaraya–Watson family, and ONE form covers every case — a smoothed score
difference:

    Φ(y) = smooth(∇log p)(y) / Z_p(y) − smooth(∇log q)(y) / Z_q(y) .

The q-term is always the KDE score of the current ensemble {xⱼ}:
`smooth(∇log q)(y) = Σⱼ ∇_y k(xⱼ, y)`. The p-term depends on what the
target provides:

- **a score** (Gaussian, mixture, `unnormalized`):
  `smooth(∇log p)(y) = Σⱼ k(xⱼ, y) ∇log p(xⱼ)` — the true score,
  smoothed over the same ensemble (so `Z_p = Z_q`). Score evaluations
  live on the ensemble points, never on the query — recorded replay
  stays self-contained.
- **samples** (Dataset target {pᵢ}):
  `smooth(∇log p)(y) = Σᵢ ∇_y k(pᵢ, y)` — the KDE score of the target
  samples, with its own mass `Z_p`. Both sides share one kernel and one
  bandwidth, so the two smoothing biases cancel exactly where `p̂ = q̂`:
  the field has the right zeros even at finite h
  (`examples/svgd_data.liu`).

Which `Z` runs is **a theorem of the divergence and the metric**, not a
switch (the former `normalize=` flag was removed 2026-07):

- `reverseKL` + `metric=stein`: `Z` = point count. This is precisely the
  SVGD update

      Φ(y) = (1/n) Σⱼ [ k(xⱼ, y) ∇log p(xⱼ) + (2/h)(y − xⱼ) k(xⱼ, y) ]

  — the exact steepest descent in the kernelized Stein geometry.
- `reverseKL` + `metric=w2` (default): `Z` = kernel mass at the query
  point — the genuine Nadaraya–Watson interpolation and a **consistent
  estimator of the W2 velocity ∇log(p/q)** (Liu, Yu, Simons, Yi &
  Beaumont 2024, eq. 5: the SVGD update is exactly the numerator; the
  denominator varies with the query point, so it cannot be folded into
  the learning rate).
- `mmd`: `Z` = point count — the witness gradient `∇(p̂ − q̂)`, the MMD
  flow (Arbel et al. 2019); exact for the empirical measures.

Bandwidth conventions follow the form, each pinned to its literature:
the unnormalized forms take the Liu & Wang 2016 heuristic
`h = med{‖xᵢ−xⱼ‖²}/log(n+1)` (each point's neighbourhood contributes
~1/n of the kernel mass); the normalized form takes the Gretton median
heuristic `σ = med{‖xᵢ−xⱼ‖}` (`h = 2·med{‖xᵢ−xⱼ‖²}`), no log-n scaling
— the NW ratio needs smoothing on the scale of the data, not of the
interaction strength. Dataset targets pool the ensemble with the target
samples before taking the median (one kernel for both sides).
Practically, where the ensemble is thin the W2 field stays O(1) while
the Stein one fades — stragglers get pulled in instead of left behind
(`examples/svgd_normalized.liu`, one divergence under two geometries).
The `w2` divergence bypasses kernels entirely: its field is the
barycentric displacement of the entropic plan (`examples/w2_descent.liu`).

### 6.2 Constrained descent: `descent(..., family=rotation)`

`family=` lives on **`descent`**, because it names the manifold the
**path** moves on — a constrained curve is a *different curve*, exactly
as `metric=` yields a different curve; the two are orthogonal axes
(metric picks the ambient geometry, family picks the submanifold). The
default (`free`) is all of W2 space. `family=rotation` declares the
IVP on the orbit `{R # from : R ∈ SO(d)}`: a curve of **rotations** of
the initial ensemble. Everything downstream follows as a theorem of
that declaration, not as switches on `field`/`flow`:

- **the field**: the constrained path's true velocity is the L²(q)
  projection of the free descent field onto the orbit's tangent space —
  the linear fields `{z ↦ Ωz : Ω ∈ so(d)}`. On a white ensemble
  (`E[zzᵀ] = I`) the projection has the closed form

      M = (1/n) Σᵢ v(zᵢ) zᵢᵀ ,   Ω = (M − Mᵀ)/2 ,

  which is why `whiten(X)` (deterministic PCA standardization, no RNG)
  comes first. `estimator=nw` keeps its usual job: it estimates the
  *ambient* field v; the projection is exact linear algebra.
- **the flow**: `flow` integrates on the group, `z ← exp(lr·Ω) z` — a
  Lie-group step, not Euler. Exactness matters: `Ωz ⊥ z`, so a Euler
  step `z += lr·Ωz` inflates every norm and drifts off the orbit;
  the exponential conserves volume and covariance to the bit.

The search space collapses from infinite-dimensional W2 to the
d(d−1)/2 rotation angles (marginals cannot be bent, only mixed), and Ω
is the natural-gradient direction of Amari's ICA, so

```liu
qt = descent(reverseKL(decouple(Z) ~ n), from=Z, family=rotation)
v  = field(qt, estimator=nw(kernel=rbf))
```

*is* continuous-time natural-gradient ICA, expressed as a constrained
Wasserstein gradient flow (`examples/sica_rotation.liu`; the free-flow
companion is `examples/sica.liu`).

**Process independence** (`rotation(block=L)`, 2026-07). Instantaneous
independence has a wall: Gaussian sources are unidentifiable from the
time-marginal cloud (measured: `mixed_ar`'s Gaussian AR pair leaves the
instantaneous flow wandering at MCC ≈ 0.8). The fix is one data
reshape — `window(Z, L)` embeds the trajectory as its sliding-window
cloud (channel-major lag blocks), `decouple(Z, block=L)` bootstraps
whole per-channel windows (each channel keeps its own dynamics: the
**permuted product**), and `family=rotation(block=L)` constrains the
flow to `{R ⊗ I_L}` — one channel rotation shared across lags, whose
projection is the skew part of the lag-averaged moment (exact after
instantaneous channel whitening). Cloud independence of windows *is*
process independence up to horizon L, and the same KL descent that was
blind separates the Gaussian AR pair exactly
(`examples/sica_process.liu`: 0.8 → 1.000; `unwindow(Z, L)` reads the
channel trajectories back off the window cloud for the final
waveform view — the raw cloud would plot L shifted copies per
channel). This is SOBI's regime done
with full-distribution machinery: one objective uses non-Gaussianity,
lagged structure, or both — whichever signal exists.

The same target also runs on the TRANSPORT engine — zero new
vocabulary: each round declares the path from Z to `decouple(Z)`,
regresses the flow-matching field, and pushes Z through the map
(`examples/sica_rf.liu`; the MIRI mechanism with independence as the
reference). One transport round removes most of the dependence; OT
coupling straightens each map and measurably cuts its distortion. Two
honest asymmetries against the descent engine: the guarantee is
independence, not source identity (a free transport bends marginals
slightly — MCC 0.984, not 1.000), and whitening HURTS here — after
whitening, the free transport finds the nearest independent
configuration instead of the big linear move that happens to be the
unmixing (measured: MCC collapses to 0.74). Constrained flows want
whitening; free transports want the correlation left in.

The constraint also runs on the transport engine:
**`estimator=regress(rotation)`** restricts the FM regression's
hypothesis class to skew-linear fields — closed form per time slice,
no network, map exactly a rotation, `inv` free — and the whitening
story flips back, symmetrically: whitened + constrained converges to
exact recovery (MCC 1.000 over 48 cheap rounds; the per-round rotation
is small but consistently signed — an OT coupling gives a clean
signal, not a bootstrap random walk), while unwhitened it stalls
(rotations preserve the covariance spectrum). Doctrine note: a
declared path cannot be re-curved, so the constraint lives in the
estimator's hypothesis class — `family=` (descent: changes the curve)
vs `regress(rotation)` (declared: projects the velocity), a meaningful
asymmetry, not an inconsistency (`examples/sica_rf_rotation.liu`).

The class also comes in the process flavor —
`regress(rotation(block=L))`, one channel rotation shared across lags —
and with it the **2×2 matrix closes**: {descent, transport} ×
{marginal, process}, all four cells rotation-constrained, each with a
quant-asserted example. The Gaussian AR wall holds for BOTH engines'
marginal cells (descent 0.809, transport ~0.72 — theory demands it)
and both process cells open it exactly (1.000/1.000):

| | descent 引擎 | transport 引擎 |
|---|---|---|
| marginal | `sica_rotation.liu` 1.000 | `sica_rf_rotation.liu` 1.000 |
| process (Gaussian AR) | `sica_process.liu` 0.81 → 1.000 | `sica_rf_process.liu` 0.72 → 1.000 | Why constrain when the free flow also
unmixes the toy? Identifiability: a free independence-seeking flow can
reach *any* independent-marginal configuration (the Darmois
construction), so its success on a toy is the toy's mercy; the rotation
flow searches exactly the linear-ICA model class, where non-Gaussian
sources make the answer unique up to permutation and sign. It is the
third instance of one mechanism — project the descent direction onto a
model manifold's tangent space (KiNG: exponential families; ntKiNG: NTK
function classes; here: SO(d)).

Two implementation facts follow from the projection, not from taste.
Bandwidth: the projection averages n·d field values into d(d−1)/2
numbers, so variance is cheap and the **sharp** Liu & Wang bandwidth is
used even for the normalized field — a wide interpolation bandwidth
Gaussianizes the KDE scores, and a linear field has zero skew moment
(the rotation signal lives in the higher cumulants; by Stein's identity
the q-term's skew part vanishes in the exact limit, so everything rides
on resolving the target's non-Gaussianity). Constraints: `record=` and
`into=` are teaching errors — the whole map is the product of its
per-step rotations, a single d×d orthogonal matrix; there is nothing to
record and nothing to amortize (its inverse would be free: the
transpose — storing the composite is future work if a use case shows
up); `family=` on `field()` is a teaching error pointing back to
`descent`. A non-white start triggers a printed reachability note —
at the `descent` declaration when `from=` is a Dataset (deterministic
there), at apply time for a Distribution start — because rotations
preserve the covariance, so the orbit of a non-white ensemble can
never reach a whitened target.

## 7. Fields, maps, pushforward

### 7.1 `flow`

`flow(v, steps=50, lr=, record=false)` integrates a Field into a Map.

- For self-contained fields (from `regress`, `reverse`, `transport`):
  explicit Euler over `t ∈ [0, 1]` in `steps` steps.
- For descent fields: `steps` interacting-particle updates of size `lr`,
  re-estimating the field from the current ensemble at every step. This
  simulation is the **mean-field approximation** of the true gradient
  flow; finite-particle bias is real (shrink `n` and watch it grow).
- `record=true` (descent fields only) stores the per-step ensembles and
  bandwidths, making the map replayable at arbitrary points — and hence
  invertible (§7.3).
- `into=mlp(d -> ... -> d)` (descent fields only) is the third
  representation — the **amortized flow**, a drifting model (spec
  §10.6). Instead of moving particles, move parameters: each optimizer
  step draws a latent batch `z` from the path's `from=`, computes
  `y = net(z)`, and regresses `net(z)` onto the frozen one-Euler-step
  target `y + lr·Φ(y)`, with Φ the same nw field estimated on the
  generated batch. The pushforward `net#from` evolves across optimizer
  steps and training stalls exactly at equilibrium (Φ ≡ 0 where
  p̂ = q̂); inference is a single forward pass. Under `into=`:
  `steps` = optimizer steps (default 2000), `lr` = drift step,
  `trainlr=` the net's own rate, `batch=` latent batch; the net takes
  no time input. The streamed loss is `lr²·mean‖Φ‖²` — the zero-flow
  diagnostic. The resulting Map is hard-pinned to `from=` and refuses
  `inv` (§7.3); `record=true` is mutually exclusive. Honest caveat:
  amortization is mode-seeking — on well-separated modes the generator
  migrates collectively into the heavier one, while the particle
  simulation of the same field covers both (`examples/drifting.liu`).
  Lineage: two-timescale ideal = iNGD (Liu, Wang & Wang 2025);
  merged loops = Drifting Models (Deng et al. 2026).

### 7.1a Field algebra — guidance

Self-contained Fields form a vector space, and Liu grants them their
native algebra: `a*v1 + b*v2`, `-v`, `v/c`. Classifier-free guidance is
the affine extrapolation

```liu
v = v_u + 3*(v_c - v_u)     // w = 3; coefficients sum to 1
```

where `v_u`, `v_c` are the fields of the unconditional and conditional
declared paths (same schedule, same noise endpoint). The combined field
is **synthetic**: it solves the continuity equation of *no* declared
path, and the language makes no claim about what its flow samples —
that question is open in the literature. Its provenance string records
the merged affine weights, e.g. `(-2*[field(...)] + 3*[field(...)])`.

Static checks: dimensions must match; the summands' paths must start
from the same initial measure (hard error otherwise); descent fields
are rejected — they are re-estimated from the evolving ensemble and
have no standalone evaluation. On the descent side, combine upstream
at the divergence level instead (`descent(a*D1 + b*D2)`), whose
stationary law is *exactly* the geometric mixture — same algebra, one
side a theorem, the other an open question.

`inv` remains free on combined fields (the ODE does not ask where its
field came from): DDIM-style inversion of guided flows comes for free.

**Residual estimator** (spec §10.3.2). `regress(net, base=v_u)`
freezes a pretrained field and regresses only the correction: the L²
target becomes `dx/dt − v_u(x_t, t)`, whose minimizer is exactly the
guidance direction `v_c − v_u`. The result `v_c = v_u + Δ` is a
genuine field of the conditional path, so the default usage needs no
guidance formula at all: `flow(v_c) # noise` *is* conditional
sampling. The extrapolation formula exists only for the deliberate
distortion knob `w ≠ 1`. Represented as the combination
`1*[v_u] + 1*[Δ]`, `v_u + w*(v_c − v_u)` merges to `v_u + w*Δ` — the
base's weight cancels *at the representation level* (the w=1 collapse
is structural rather than numerical, visible in the provenance), and
the error amplified by `w` is that of one small residual instead of
the difference of two independently trained networks. Shared-source
provenance holds by construction, and one base amortizes across many
conditions. What it does not change: the `w > 1` extrapolation is
still synthetic — the parameterization improves the estimator, not the
semantics. Runnable: `examples/guidance_residual.liu`; the foreign-base
teaching error: `examples/err_residual_base.liu`.

### 7.2 `#` — pushforward

`T # μ` (Distribution) is a **lazy** pushforward — itself a Distribution;
sampling it draws from μ and transports. `T # X` (Dataset) applies the
map now and returns a Dataset that records its trajectory (§8).

`v # μ` for a bare field is illegal *by design*: a field must first be
integrated into a map. The extra word buys the language's central
distinction — fields are local descriptions, maps are global objects.

### 7.3 `inv` — a three-tier price list

> **`inv` is free iff the field is a self-contained formula.**

1. **Free.** Maps built from trained fields (`regress`, `reverse`,
   `transport`): integrate the same field backwards. The probability-flow
   map of diffusion is invertible this way — DDIM inversion / encoding
   comes for free.
2. **Bought with memory.** A descent map's field is a *readout of the
   evolving ensemble*, not a formula; but under `record=true` the frozen
   ensembles make Φ_t evaluable at arbitrary points, and each step
   `y = x + lr·Φ_t(x)` is solved for `x` by fixed-point iteration.
   Empirically the initial cloud is recovered point-for-point at demo
   scale; conditioning degrades with deep convergence (dissipation).
3. **Bought with training.** `reverse` (§7.5) upgrades a readout field to
   a self-contained one; after that purchase, tier 1 applies.

Naively negating a descent field *without* history simulates gradient
**ascent** from the endpoint — a different dynamical system, not the time
reversal — which is why an unrecorded descent map rejects `inv` outright.

### 7.4 Provenance checks

- An **unrecorded** descent map is pinned to its path's `from=` measure:
  applying it to points drawn from anything else is a **hard error** —
  the map *is* the simulation, its field is re-estimated from the moving
  ensemble, so foreign particles change the dynamics itself (from a
  different start the field is no longer the descent direction of the
  divergence).
- **Recording lifts the pin.** Under `record=true` the frozen per-step
  ensembles make the NW readout pointwise at arbitrary points — like a
  trained model evaluated off its training set — and the replay is
  self-contained in both directions. Foreign measures are then legal;
  a printed **note** states what the answer means: the pullback of a
  foreign measure (e.g. `inv(T) # target`) reads as a transport back to
  `from=` exactly insofar as the forward flow converged (a descent path
  is an IVP — only its start is a declared boundary; the target is the
  t→∞ limit, not the recorded endpoint's law).
- Maps from self-contained fields apply to anything (creative uses such
  as latent interpolation are a feature); provenance is informational.

### 7.5 `reverse` — learned time reversal (implicit paths)

```liu
fwd = descent(reverseKL(noise), from=data, time=3)
v   = reverse(fwd, estimator=denoiser(mlp(2 -> 96 -> 96 -> 2)), steps=, lr=)
```

The forward KL-descent toward a Gaussian is (a time-change of) the OU
process; its marginals `data ⊛ Gaussian` are exactly samplable in closed
form, so the denoiser trains on **true** noised marginals via denoising
score matching, and estimation error is integrated only once, in the
generative direction. `reverse` returns the probability-flow ODE's
velocity field — a self-contained Field, so the resulting map is
invertible for free.

Declared paths never need `reverse`: time reversal is the substitution
`t → 1−t` in the formula itself. The prototype requires the reverseKL
target to be a single Gaussian (the reference process's stationary law).

### 7.6 `transport` — legacy sugar

```liu
v = transport from A to B using mlp(2 -> 64 -> 64 -> 2) for N steps with lr=, batch=
```

is sugar for `field(prob(t*B + (1-t)*A), estimator=regress(net))`, kept
for its readable sentence form.

### 7.6a Kernels — conditional transport with `|`

Conditional distributions enter as first-class **kernels** (spec §10.8;
training per Wang, Wang, Liu & Suzuki 2026, Zero-Flow Encoders, eq. 6).
`(y, x) = rv(joint)` destructures one joint draw into coordinate blocks;
the path formula interpolates **bare** y-blocks (draw level), and the
conditioning indices are attached **at the law gate** —
`prob(yt | x1, x0)`, one index per coupled term, declared in term order
(2026-07; the inline spelling `t*(y1|c1) + ...` is a teaching error:
formulas interpolate draws, conditioning operates on laws). Each
endpoint's condition enters the velocity net; a conditional map then
transports a **kernel** whose condition slots stay open:

```liu
(y0, c0) = rv(linear_gaussian())
(y1, c1) = rv(sine_gaussian())
yt = t*y1 + (1-t)*y0
Tk = flow(field(prob(yt | c1, c0), estimator=regress(mlp(3 -> 64 -> 64 -> 1)))) # (y0 | c0)
plot (Tk | c0) ~ 1000           // paired: joint samples suffice
plot (Tk | [1.2]) ~ 300         // decoupled: one conditional slice
```

**Paired vs decoupled evaluation.** Filling the slot with the block
bound to the y-source (`| c0` — draw identity) needs only joint
samples: each y is already a sample of its own condition's conditional,
and pushing pair-by-pair is slice-by-slice conditional transport.
Filling it with a foreign point/measure needs the source joint's
**conditional sampler** (constructed joints have one; data-only joints
do not — hard error pointing back to the paired route). See
`examples/cond_transfer.liu`; full usage tables in `docs/liu-api.md`.

### 7.6b Programmable kernels — `kernel(param) { ... }` (spec §10.10)

The bundled joints end where your simulator begins: `kernel(w){...}`
registers a straight-line **program** as a Markov kernel p(y | w). The
body is a second, scoped semantic level — the **draw level**: names
denote blocks of draws (columns), arithmetic is elementwise with 1-row
broadcast, `sin/cos/exp/log/sqrt` apply per entry, Distribution values
auto-lift to fresh draws (the t-formula rule; a repeated canon is a
teaching error), and `T # z` applies a *trained* transport per draw —
runtime objects inside simulators are the point (descent maps refuse:
they re-simulate, they are not pointwise functions). No `~`, no
`|`/`or`, no nesting, no data-dependent branching: the program stays a
finite DAG and the reproducibility contract rides on statement order,
exactly as everywhere else. The last binding's value is the output y.

Instantiation reuses §7.6a's surface verbatim: `(K | [w]) ~ n` runs the
simulator at a fixed parameter, `(K | prior) ~ n` draws the joint
(y; param) with the observed block first, `(K | X) ~ n` resamples
parameters from a Dataset. The product is an ordinary joint Dataset —
every engine downstream consumes it unchanged (`sbi_kernel.liu` runs
the conditional-descent posterior on a user-defined simulator).
Programmable kernels are samples-only: no density, no score — the
likelihood-free premise, now user-authored. Inside a body, bounded
`for` is an *expression*: its value row-stacks each round's final
binding, and each round's distributions are fresh draws (iid). Stage 2
adds `dot(a, b)`
(per-draw inner products), `jvp(T, z, v)` (the trained map's Jacobian
along a direction, by central differences), and single-column Datasets
as constant blocks — enough for the manifold latent variable model of
Khoo, Liu & Beaumont 2026 to be written as a body verbatim
(`manifold_local.liu`: signal `G # (w0 + w*x)`, tangent `jvp(G, z, w)`,
normal-projected noise via one dot ratio).

### 7.7 `via` — inference algorithms at the intent level

```liu
Y = target ~ n via svgd(kernel=rbf, steps=, lr=)
```

desugars to
`flow(field(descent(reverseKL(target), from=N(0,I)), nw), steps, lr) # (N(0,I) ~ n)`.
The target must have a score (Gaussian mixtures; `unnormalized`).

## 8. `plot` and data export

```liu
plot a, b, ...                 // overlaid scatter of Datasets
plot trajectory of X           // per-step snapshots of a flow's output
plot_signal x, y, ...          // the waveform view: columns in index order
```

`trajectory of` applies to any Dataset produced by a flow — every value
remembers its history. In the terminal, plots render as ASCII scatter;
with the environment variable `LIU_DUMP=<file>` set, each plot also
appends its series (and full trajectory frames) as JSON lines, which is
how the web demo (`web/demo.html`) is produced.

**`plot_signal`** is the third view (2026-07): the same Dataset that
`plot` shows as a cloud, drawn as **signals** — each coordinate row is
one line over the column index. Column order is deterministic in this
language and survives `whiten`, rotation flows, and map application,
so paired series stay aligned (the unmixing demos read as waveform
recovery). Bare distributions sample 500 first; the view is a window of the
first 200 samples (a zoom, not a decimation, which would alias
waveforms; the footer states the total). I.i.d. draws render honestly
as noise — the terminal uses oscilloscope min–max bands per character
column, so a bimodal source shows as two rails rather than a false
line (`examples/signals.liu`).

## 9. Static checks (errors that teach)

Checks run before computation; messages state the violated mathematics.
A selection:

| Program fragment | Rejection (abbreviated) |
|---|---|
| `moons(0.1) ~ n via svgd(...)` | svgd requires the target's score ∇log p; moons has no analytic score |
| `banana ~ n` (unnormalized) | admits no exact sampler — specify an inference algorithm with `via` |
| `inv(flow(field(descent(...))))` | no pointwise inverse without a recorded history; three price tiers |
| descent map applied to foreign points | provenance mismatch: a gradient flow is an IVP; its trajectory depends on its initial measure |
| same distribution twice in a formula | no draw identity: same draw or independent copies? disambiguate with `rv()` |
| `v # μ` (bare field) | a field must be integrated into a map first (`flow`) — no sugar here, by design |
| `μ ~ 200000` | a Dataset holds at most 100 000 samples |

## 10. Worked examples

All are runnable from the repository root (`./build_liu/liu <file>`):

| File | Demonstrates |
|---|---|
| `examples/path_fm.liu` | flow matching as a declared path; pullback check; trajectory |
| `examples/path_diffusion.liu` | diffusion by changing one coefficient line; free encoder |
| `examples/svgd.liu` | SVGD at intent level (`via`) and mechanism level (`descent`) |
| `examples/svgd_inverse.liu` | recorded-replay inversion of smoothed SVGD |
| `examples/svgd_bayes.liu` | unnormalized (Bayesian) targets: banana ridge, double well |
| `examples/unnorm_pipeline.liu` | descent on an unnormalized target + amortized distillation into a flow |
| `examples/diffusion.liu` | the implicit-path (legacy) diffusion via `reverse` |
| `examples/flow_matching.liu` | the `transport` sugar |
| `examples/err_*.liu` | the static checks of §9, one per file |

## 11. Prototype limits and roadmap

Not implemented (each rejects with a pointer): `locallinear`,
`ratio_then_grad`, and `empirical` estimators; `estimate`, `mirror`, and
the `W2`/Sinkhorn divergence family; `train`/`generate`/`eval` and the
MNIST/text8 datasets; `metric=fisher_rao`. Roadmap (see `liu-spec.md`
§10): bounded `for` (macro-expansion semantics), first-class couplings,
Schrödinger bridges (`bridge`, IPF), the SDE field/flow pair, and
first-class coupling slots in path formulas.

## Appendix: grammar (EBNF)

```
program    = { statement } ;
statement  = "seed" NUMBER | IDENT "=" expr | plot ;
plot       = "plot" plottarget { "," plottarget } ;
plottarget = expr | "trajectory" "of" expr ;

expr       = mixture ;
mixture    = additive { "|" additive } ;
additive   = mult { ("+" | "-") mult } ;
mult       = push { ("*" | "/") push } ;
push       = postfix { "#" postfix } ;
postfix    = primary { "~" NUMBER [ "via" call ] } ;
primary    = "-" primary | transport | call | dims-call
           | IDENT | NUMBER | vector | "(" expr ")" ;
transport  = "transport" "from" expr "to" expr "using" primary
             [ "for" NUMBER "steps" ] [ "with" kwargs ] ;
call       = IDENT "(" [ arg { "," arg } ] ")" ;
dims-call  = IDENT "(" NUMBER "->" NUMBER { "->" NUMBER } ")" ;
arg        = [ IDENT "=" ] expr ;
kwargs     = IDENT "=" NUMBER { "," IDENT "=" NUMBER } ;
vector     = "[" [ NUMBER { "," NUMBER } ] "]" ;
comment    = "//" to end of line ;
reserved   = "t" | "x1" | "x2" ;
```
