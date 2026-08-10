# Liu (流) — Cheat Sheet

Liu is a tiny language for describing machine-learning experiments as
**measure transport**. One pipeline covers flow matching, diffusion, and
SVGD:

```
flow( field( path ) ) # μ        // integrate the field of a path, push a measure
```

A program is a straight-line script: no loops, no branches, no I/O. Its
text plus its `seed` determines the output **bit for bit**.

## Build & run

```bash
./interpreter/build.sh                          # needs OpenBLAS/LAPACK/Boost/spdlog
./build_liu/liu examples/path_fm.liu
```

## Sixty-second tour

```liu
seed 42                                   // first line of every program

data  = moons(0.08) ~ 1000                // sample a toy distribution → Dataset
noise = gaussian([0, 0], 1)
mix   = 0.7*gaussian([-2,0], 0.4) or 0.3*gaussian([2,0], 0.4)  // mixture

plot data, mix ~ 500                      // scatter plot (any number of series)
```

**Flow matching** — write the interpolation as a formula in `t`:

```liu
xt = t*data + (1-t)*noise                 // a stochastic process
pt = prob(xt)                             // its law: a probability path
v  = field(pt, estimator=regress(mlp(2 -> 64 -> 64 -> 2)))   // E[dx/dt | x_t]
T  = flow(v, steps=50)                    // integrate: a Map
plot data, (T # noise) ~ 1000             // pushforward ≈ data?
plot (inv(T) # data) ~ 1000, noise ~ 1000 // pullback ≈ N(0,I)?  (inv is free here)
plot trajectory of (T # noise) ~ 300      // animation frames
```

**Diffusion** — change one line of coefficients; everything else is identical:

```liu
xt = sqrt(1 - t*t)*e + t*data             // VP schedule, generative direction
```

**SVGD** — the gradient flow of reverse KL; no training, just estimation:

```liu
Y = mix ~ 300 via svgd(kernel=rbf, steps=400, lr=0.8)       // intent level
// mechanism level (identical computation):
qt = descent(reverseKL(mix), from=cloud)  // steepest descent: an IVP in P(R^2)
v  = field(qt, estimator=nw(kernel=rbf))
Z  = flow(v, steps=400, lr=0.8) # cloud
```

**Bayesian targets** — a score needs no partition function:

```liu
banana = unnormalized( -x1*x1/8 - (x2 - x1*x1/4 + 1)*(x2 - x1*x1/4 + 1)/0.4 )
Y = banana ~ 400 via svgd(kernel=rbf, steps=600, lr=0.3)    // ~ without via is rejected
```

## Rules to remember

- `~ n` samples a distribution into a Dataset; `via alg(...)` when no
  exact sampler exists (or for teaching).
- Three descents, and the divergence×metric pair DECIDES the field (no
  flags): `reverseKL` under `metric=stein` = exact SVGD update; under
  `metric=w2` (default) = normalized NW estimate of grad log(p/q);
  `mmd(data)` = the witness-gradient MMD flow; `w2(data, eps=)` =
  Sinkhorn barycentric displacement (memorizes an empirical target as
  eps→0 — kernel descents generalize, w2 reproduces).
- `#` is pushforward: `Map # Distribution` is lazy; `Map # Dataset` runs.
- `inv` has a price list: **free** for self-contained fields (regress /
  reverse); **memory** for gradient flows via `flow(..., record=true)`;
  **training** via `reverse(descent(...), estimator=denoiser(net))`.
  Amortized maps (`into=`) refuse `inv` — the trajectory dissipated
  into the weights.
- `flow(v, into=mlp(d -> ... -> d), steps=, lr=, batch=)` amortizes a
  descent into a **one-step generator** (a drifting model): optimizer
  steps replace particle steps, inference is one forward pass. Pinned
  to `from=`; loss curve = the zero-flow diagnostic.
- A distribution appearing **twice** in one formula is ambiguous (same
  draw or independent copies?) — disambiguate with `rv(...)`.
- `(x0, x1) = rv(couple(A, B, via=ot))` draws OT-matched endpoint PAIRS —
  the same interpolation formula becomes OT-CFM (straight paths); add
  `sqrt(t*(1-t))*sigma*z` with `via=sinkhorn(eps)` for a Schrödinger
  bridge. `~ n` after the bind freezes one coupling of n pairs.
- `plot_signal x, y` is the waveform view: columns in index order, one
  line per coordinate row (`plot` is the cloud view of the same data);
  bare distributions sample 500 first.
- Process independence in one reshape: `window(X, L)` embeds a trajectory
  as its sliding-window cloud; `decouple(Z, block=L)` permutes whole
  per-channel windows; `rotation(block=L)` shares one channel rotation
  across lags — Gaussian AR sources separate where instantaneous ICA
  provably cannot (`sica_process.liu`).
- `decouple(X)` is couple's dual: the product of X's empirical marginals
  (dependence forgotten) — iterate a KL descent toward it to unmix (SICA).
  `whiten(X)` standardizes (cov → I, deterministic), and
  `descent(..., family=rotation)` constrains the PATH to the rotation
  orbit — its field is the so(d) projection and `flow` steps by
  `exp(lr·Ω)`, both by theorem: natural-gradient ICA as a constrained
  gradient flow.
- `for k in 1..K { ... }` (K a literal, ≤ 64) is the ONLY loop — macro
  expansion, no while/break/if. Rebinding inside is fine; every event is
  tagged `iter=[k]`. With `couple(noise, T # noise, via=paired)` one
  round of reflow straightens the flow: `flow(v, steps=1)` then samples
  in one step.
- `regress(net, base=v_u)` trains only the correction Δ over a frozen
  field: `flow(base + Δ)` is conditional sampling as-is; guidance
  `v_u + w*(v_c − v_u)` merges to `v_u + w*Δ`, the base cancelling
  exactly.
- `y|x` is conditioning: conditional endpoints in formulas, and filling
  a transported kernel's condition slot (`(Tk | x0) ~ n`, `Tk | [1.2]`).
- `t` is the only reserved symbol (time). `x1`/`x2` are the coordinate
  symbols of `unnormalized(...)` only while unbound — binding them is fine.
- Gradient-flow maps are pinned to their `from=` measure — foreign
  particles are a hard error *until* `record=true` freezes the replay;
  a recorded map applies anywhere (a printed note says what it means).
- Errors are lessons: read them; they state the mathematical fact you
  violated.

Full reference: `docs/liu-reference.md`. Every function/operator with
signatures and defaults: `docs/liu-api.md`. Runnable programs:
`examples/*.liu`.
