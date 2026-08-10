# 流 (Liu)

**An executable notation for measure transport.**

Liu is a tiny language for writing flow-based generative-modeling
experiments the way they are written on a blackboard. Programs are
straight-line — no I/O, not Turing complete. 

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
`descent(reverseKL(target), from=cloud)` and it is SVGD. 

The philosophy of the program is one pipeline:

```
flow( field( path ) ) # μ
```

## Layout

| Path | What |
|---|---|
| `interpreter/liu.cpp` | single-file reference interpreter (C++20, CPU BLAS) |
| `examples/` | runnable examples: hello, FM, diffusion, SVGD, SBI |
| `docs/liu-reference.md` | language reference (English) |
| `docs/liu-api.md` | complete function & operator reference (signatures, defaults, limits) |
| `docs/liu-cheatsheet.md` / `.pdf` | 3-minute cheat sheet |
| `docs/sbi-cookbook.md` | simulation-based inference cookbook (Chinese) |
| `docs/liu-spec.md` | design history and roadmap (Chinese) |
| `external/juzhen` | the [Juzhen](https://github.com/anewgithubname/Juzhen) C++ matrix backend|
| `web/` | browser playground (`playground.html` + `server.py`), demo & intro slides |

## Build & run

```bash
sudo apt install g++ libopenblas-dev liblapack-dev   # Linux (macOS: brew install openblas)
./interpreter/build.sh          # produces build_liu/liu
./build_liu/liu examples/flow_matching.liu
./build_liu/liu --export-pytorch examples/hello.liu > hello_torch.py # export pytorch file
```

## Web playground

A browser front end for the interpreter: an editor with an example
gallery (populated live from `examples/*.liu`), streamed program output,
and live loss curves and plots while a run is still training. The server
is Python stdlib only — nothing to install.

```bash
./interpreter/build.sh          # the server runs build_liu/liu
python3 web/server.py           # http://localhost:8080  (--port to change)
```

The language itself is the sandbox (no I/O primitives, not Turing
complete); the server adds the process-level guards: a wall-clock
timeout, an address-space limit, a program-size limit, and a concurrency
cap. Determinism makes whole-run caching sound — resubmitting an
unchanged program replays the recorded event stream exactly.

