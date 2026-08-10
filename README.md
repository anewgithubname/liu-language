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
| `examples/` | runnable examples: FM, diffusion, SVGD, inversion, Bayes, and the teaching errors |
| `web/` | zero-dependency playground server + static demo + poster sources |
| `editors/vscode` | VS Code extension: highlighting + run button + error squiggles (symlink install, see its README) |
| `bench/` | CPU benchmarks vs a parameter-matched PyTorch mirror (results + where the differences come from) |
| `docs/liu-reference.md` | language reference (English) |
| `docs/liu-api.md` | complete function & operator reference (signatures, defaults, limits) |
| `docs/liu-cheatsheet.md` / `.pdf` / `.pptx` | 3-minute cheat sheet and the A4 poster |
| `docs/roadmap.md` | implementation status vs roadmap: what lands in v0.4 / v0.5 (Chinese) |
| `docs/liu-spec.md` | design history and roadmap (Chinese) |
| `external/juzhen` | the [Juzhen](https://github.com/anewgithubname/Juzhen) C++ matrix backend, **pinned submodule** |

The pinned submodule commit is part of the reproducibility statement:
*program text + seed + backend commit → bit-identical output.* The
interpreter pins BLAS to a single thread for the same reason — the
contract must not depend on the host's core count (multithreaded BLAS
changes reduction order; set `OPENBLAS_NUM_THREADS` explicitly to opt
out, trading the contract for parallelism).

## Build & run

```bash
sudo apt install g++ libopenblas-dev liblapack-dev   # Linux (macOS: brew install openblas)
./interpreter/build.sh          # auto-inits the submodule; produces build_liu/liu
./build_liu/liu examples/path_fm.liu
./build_liu/liu --export-pytorch examples/hello.liu > hello_torch.py
                                # scale-out escape hatch: a runnable PyTorch
                                # mirror (regress-FM/diffusion incl. inv,
                                # OT-CFM/reflow coupling, CFG field algebra,
                                # score-target SVGD; semantically equivalent,
                                # NOT bit-identical — the reproducibility
                                # contract stays on the Liu side)
```

## Playground

```bash
python3 web/server.py --port 8080    # Python stdlib only
# open http://localhost:8080 — edit, Run, live per-statement loss curves,
# streaming plots, errors highlighted on their source line
```

The language itself is the sandbox (no I/O primitives, not Turing
complete); the server adds process-level guards (wall-clock timeout,
memory limit, program-size limit, concurrency cap) and whole-run
caching, which determinism makes sound.
