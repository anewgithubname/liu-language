# 流 (Liu)

**An executable notation for measure transport.**

Liu is a tiny language for writing flow-based generative-modeling
experiments the way they are written on a blackboard. Programs are
straight-line — no loops, no I/O, not Turing complete — and the program
text plus a `seed` determines every output **bit for bit** (with the
backend pinned; see below).

```liu
seed 42
noise = gaussian([0, 0], 1)
data  = moons(0.08) ~ 1000

xt = t*data + (1-t)*noise         // the interpolation formula IS the program
pt = prob(xt)                     // its law: a probability path
v  = field(pt, estimator=regress(mlp(2 -> 64 -> 64 -> 2)))
T  = flow(v, steps=50)            // flow( field( path ) )

plot data, (T # noise) ~ 1000     // pushforward: generated samples
plot (inv(T) # data) ~ 1000, noise ~ 1000   // pullback, for free
```

Change one line of coefficients — `xt = sqrt(1 - t*t)*e + t*data` — and
the same program is a diffusion model. Swap the path for
`descent(reverseKL(target), from=cloud)` and it is SVGD. One pipeline:

```
flow( field( path ) ) # μ
```

## Layout

| Path | What |
|---|---|
| `interpreter/liu.cpp` | single-file reference interpreter (C++20, CPU BLAS) |
| `examples/` | runnable examples: hello, FM, diffusion, SVGD, SBI, SICA |
| `docs/liu-reference.md` | language reference (English) |
| `docs/liu-api.md` | complete function & operator reference (signatures, defaults, limits) |
| `docs/liu-cheatsheet.md` / `.pdf` | 3-minute cheat sheet |
| `docs/sbi-cookbook.md` | simulation-based inference cookbook (Chinese) |
| `docs/liu-spec.md` | design history and roadmap (Chinese) |
| `external/juzhen` | the [Juzhen](https://github.com/anewgithubname/Juzhen) C++ matrix backend, **vendored & pinned** |

The pinned backend version is part of the reproducibility statement:
*program text + seed + backend version → bit-identical output.* The
interpreter pins BLAS to a single thread for the same reason — the
contract must not depend on the host's core count (multithreaded BLAS
changes reduction order; set `OPENBLAS_NUM_THREADS` explicitly to opt
out, trading the contract for parallelism).

## Build & run

```bash
sudo apt install g++ libopenblas-dev liblapack-dev   # Linux (macOS: brew install openblas)
./interpreter/build.sh          # produces build_liu/liu
./build_liu/liu examples/flow_matching.liu
./build_liu/liu --export-pytorch examples/hello.liu > hello_torch.py
                                # scale-out escape hatch: a runnable PyTorch
                                # mirror (regress-FM/diffusion incl. inv,
                                # OT-CFM/reflow coupling, CFG field algebra,
                                # score-target SVGD; semantically equivalent,
                                # NOT bit-identical — the reproducibility
                                # contract stays on the Liu side)
```
