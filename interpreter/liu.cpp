/**
 * @file liu.cpp
 * @brief 流 (Liu) v0.3+ — reference interpreter prototype (single file, on purpose).
 *
 * A tree-walking interpreter for the Liu experiment-description language
 * (spec: docs/liu-spec.md; per-function usage: docs/liu-api.md), backed by
 * Juzhen's CPU matrix/NN kernels. One translation unit, one g++ command —
 * part of the reproducibility statement (program text + seed + backend
 * commit => bit-identical output, single-threaded BLAS).
 *
 * Implemented surface (checked against docs/liu-api.md):
 *   seed, bindings, (y, x) = rv(joint) destructuring, `//` comments
 *   reserved symbol: t only (x1/x2 are unnormalized()'s coordinate symbols,
 *   shadowable by user bindings)
 *   operator precedence, loosest->tightest:
 *     or (mixture)  <  + -  <  * /  <  # (pushforward)  <  | (conditioning)
 *     <  ~ n [via svgd(...)] (postfix)  <  primaries
 *   distributions: gaussian, uniform, ring, moons, spiral, torus (2-D
 *     manifold in R^3; 3-row clouds plot in 3-D — fixed orthographic in
 *     the terminal, [x,y,z] triples in NDJSON, drag-rotate in the
 *     playground; kernel joints keep the rows-0,1 view), unnormalized(L),
 *     mixtures (w1*A or w2*B); JOINTS with conditional samplers:
 *     linear_gaussian, sine_gaussian (spec 10.8); mixed_sources = ICA toy
 *     pair-joint (x, s) — observed mixture + true sources, one draw;
 *     mixed_signals = its TIME-SERIES sibling (one draw = one trajectory,
 *     sin + sawtooth, incommensurate periods, random phases per draw);
 *     mixed_ar = Gaussian AR(1) pair (rho .9/-.6): the identifiability
 *     wall for instantaneous ICA, opened by the process machinery below;
 *     mixed_nl(a) = the SAME AR pair through an instantaneous NONLINEAR
 *     mixture (four residual GELU layers, invertible; depth is
 *     load-bearing, measured) — the rotation members' wall,
 *     project=instant's gate
 *   whiten(X): deterministic PCA standardization of a Dataset (cov -> I)
 *   window(X, L) / unwindow(Z, L): delay embedding and its inverse —
 *     trajectory to sliding-window cloud and back (lag-0 rows),
 *     channel-major lag blocks; decouple(X, block=L) bootstraps whole
 *     blocks (per-channel dynamics survive; the permuted product);
 *     family=rotation(block=L) = R x I_L — PROCESS independence;
 *     lagsplit(W, L): channel-major windows -> [elements; contexts]
 *     (blk-marked, destructurable) — SICA Alg.1 per-element refinement:
 *     conditional path on elements + regress(rotation) keeps the skew
 *     x-part of the closed-form conditional coefficient (sica_cond.liu);
 *     rotation(net=mlp) = the neural member — conditional FM net,
 *     per-slice so(d) projection of its element-Jacobian (sica_cond_net)
 *   declared paths (spec 10.2): xt = t*data + (1-t)*noise; auto-lift,
 *     draw identity via rv(); pt = prob(xt);
 *     law-gate conditioning (2026-07): formulas are built from BARE blocks;
 *     prob(xt | d0[, d1...]) attaches the indices where the law is taken
 *     (term order = slot order); a conditioned term inside the arithmetic
 *     is a teaching error (the retired inline spelling t*(y|x) + ...)
 *     field(pt, estimator=regress(net[, base=v]))  E[dx/dt | x_t] regression
 *     (flow matching; base= trains the residual = guidance direction, 10.3.2)
 *   couplings (spec 10.3): (x0,x1) = rv(couple(A, B, via=independent/ot/
 *     sinkhorn(eps)/paired)) [~ n] — batch-paired joint draw, peer blocks
 *     share one draw (OT-CFM); + sqrt(t*(1-t))*sigma*z = bridge (10.4);
 *     via=paired pairs (z, T#z) — reflow
 *   bounded for (spec 10.1): for k in 1..K { ... } — K literal <= 64,
 *     macro-expansion; events carry iter=[..k]; while/break/if never
 *   conditional kernels (spec 10.8): y|x endpoints, conditional fields
 *     v(y_t; conds, t), T # (y|x) -> Kernel, Kernel | W instantiation
 *     (paired = joint samples; decoupled = conditional sampler)
 *   programmable kernels (spec 10.10): kernel(w){ ... } — a scoped
 *     DRAW-LEVEL language (blocks of draws, elementwise arithmetic,
 *     1-row broadcast, auto-lift with repeat-canon error, T # z per draw,
 *     dot/jvp, rows(a,lo,hi) slicing = matrix-valued parameters, bounded
 *     for-as-expression); samples-only, instantiates through the 10.8
 *     surface; (y, w) = (K | z) ~ n destructures the frozen joint table
 *     by its (y; w) row marker (Dataset::blk)
 *   conditional descent (spec 10.8/10.10, SBI line): from=(y0 | x0) pins
 *     the observed block of a frozen joint — conditional SVGD (the log
 *     factors; reverseKL only); a SPREAD pinned marginal = family
 *     transport (one particle per observation; repulsion goes conditional
 *     via the pin-channel kernel — point-mass pins keep the pooled path);
 *     from=(q | Obs) conditions on an
 *     OBSERVATION SET via the KL decomposition, assembled as likelihood
 *     log-weights on the joint library (kde_score_at tells the story);
 *     both hard-pinned at #, no record=/into=/inv
 *   descent paths (spec 10.2.1): descent(D, from=, time=, metric=w2/stein,
 *     family=free/rotation)
 *     estimator=dsm(mlp, sigma=, warm=) (2026-07): neural score descent —
 *     fixed-sigma DSM nets replace the NW kernels; target net trained
 *     once, cloud net warm-started per flow step; consumes RNG (the
 *     zero-RNG-descent exception); reverseKL + Dataset target + w2 only
 *     divergences: reverseKL (score or Dataset), mmd (Dataset), w2 (Dataset)
 *     divergence x metric DECIDES the field (no normalize= flag, 2026-07):
 *     reverseKL+stein = exact SVGD update; reverseKL+w2 = normalized NW
 *     (consistent W2 velocity, Liu et al. 2024); mmd = witness gradient
 *     (MMD flow); w2 = sinkhorn barycentric displacement (no kernel)
 *     family=rotation: the CURVE is constrained to the rotation orbit of
 *     from= (whiten first) — its field is the so(d) projection and flow
 *     steps by exp(lr*Omega), both theorems of the constrained path;
 *     natural-gradient ICA as a constrained WGF (no record=, no into=)
 *   field algebra (spec 10.3.1): a*v1 + b*v2 for self-contained fields (CFG)
 *   flow(v, steps=, lr=, record=), inv(T) price list, T # mu / T # X
 *     provenance: unrecorded WG maps hard-pinned; recorded replay is free
 *   flow(v, into=net, steps=, lr=, batch=, trainlr=)  amortized flow
 *     (spec 10.6, drifting): optimizer replaces the particle simulator,
 *     one-step generator, loss = zero-flow diagnostic; pinned, no inv;
 *     conditional descent + window provenance = the amortized
 *     INSTANTANEOUS demixer (SICA Level 1, train_amortized_demixer):
 *     residual net g(x)=x+net(x), structured re-simulation,
 *     two-timescale, unpinned
 *   field(pt, estimator=regress(rotation)): skew-linear hypothesis class,
 *     closed-form per-slice regression (no net); map = exact rotation,
 *     inv free — the constrained-transport ICA engine
 *   reverse(qt, estimator=denoiser(net))   DSM probability-flow field
 *   transport from A to B using net ...    (legacy sugar)
 *   plot a, b, ... / plot trajectory of x  (ASCII scatter; LIU_DUMP emits
 *     line/iter-tagged loss, plot, and error events as NDJSON for the web UI)
 *   plot_signal x, y, ...  (columns in index order, one line per coordinate
 *     row — signals/waveforms; bare Distribution samples 500; oscilloscope
 *     min-max downsampling in the terminal, "signal" NDJSON event)
 *
 * Map of this file (grep the banner comments):
 *   lexer/parser -> values & types (Dist/Dataset/FieldV/MapV/KernelV/RTerm)
 *   -> field algebra -> execution site (line attribution) -> training
 *   (train_field / train_dsm) -> SVGD/NW -> plotting -> evaluator (Interp)
 *   -> statements (run) -> export (PyExport: --export-pytorch, template-based
 *   PyTorch mirror — regress-FM/diffusion incl. inv, couple ot/paired + for,
 *   field algebra, score-target SVGD both geometries; refuses the rest) -> main
 *   (the fused CPU Adam lives upstream: external/juzhen ml/util.cuh)
 *
 * After ANY change: ./interpreter/test.sh  (all examples; err_* must fail).
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>   // juzhen f821b52 uses memcpy/strlen without including it
                     // (came transitively via spdlog before the v1.17 pin +
                     // LOGGING_OFF); the single-TU build supplies it here
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cpp/juzhen.hpp>   // from external/juzhen (git submodule)
#include <ml/layer.hpp>     // from external/juzhen (git submodule)

using namespace std;
using namespace Juzhen;
using Mat = Matrix<float>;
using TF = function<float(float)>;

// ────────────────────────────────────────────────────────── errors ──
struct LiuError : runtime_error {
    explicit LiuError(const string& m) : runtime_error(m) {}
};
[[noreturn]] static void err(const string& m) { throw LiuError(m); }

// ─────────────────────────────────────────────────────────── lexer ──
enum class Tok { Ident, Num, Sym, End };
struct Token { Tok kind; string s; double num = 0; int line = 1; };

static vector<Token> lex(const string& src) {
    vector<Token> ts; int line = 1;
    size_t i = 0, n = src.size();
    auto push = [&](Tok k, string s, double v = 0) { ts.push_back({k, move(s), v, line}); };
    auto prev_allows_sign = [&]() {
        if (ts.empty()) return true;
        const Token& p = ts.back();
        if (p.kind == Tok::Num || p.kind == Tok::Ident) return false;
        if (p.kind == Tok::Sym && (p.s == ")" || p.s == "]")) return false;
        return true;
    };
    while (i < n) {
        char c = src[i];
        if (c == '\n') { line++; i++; continue; }
        if (isspace((unsigned char)c)) { i++; continue; }
        if (c == '/' && i + 1 < n && src[i + 1] == '/') { while (i < n && src[i] != '\n') i++; continue; }
        if (isalpha((unsigned char)c) || c == '_') {
            size_t j = i; while (j < n && (isalnum((unsigned char)src[j]) || src[j] == '_')) j++;
            push(Tok::Ident, src.substr(i, j - i)); i = j; continue;
        }
        bool neg_num = (c == '-' && i + 1 < n &&
                        (isdigit((unsigned char)src[i + 1]) || src[i + 1] == '.') && prev_allows_sign());
        if (isdigit((unsigned char)c) || neg_num) {
            size_t j = i + 1; while (j < n && (isdigit((unsigned char)src[j]) || src[j] == '.' || src[j] == 'e' ||
                                               (src[j] == '-' && src[j - 1] == 'e'))) {
                if (src[j] == '.' && j + 1 < n && src[j + 1] == '.') break;   // 1..K is a range, not a decimal
                j++;
            }
            push(Tok::Num, src.substr(i, j - i), atof(src.substr(i, j - i).c_str())); i = j; continue;
        }
        if (c == '.' && i + 1 < n && src[i + 1] == '.') { push(Tok::Sym, ".."); i += 2; continue; }
        if (c == '-' && i + 1 < n && src[i + 1] == '>') { push(Tok::Sym, "->"); i += 2; continue; }
        if (c == '|' && i + 1 < n && src[i + 1] == '>') { push(Tok::Sym, "|>"); i += 2; continue; }
        string one(1, c);
        if (string("=~|#(),.[]*+-/{}").find(c) != string::npos) { push(Tok::Sym, one); i++; continue; }
        err("line " + to_string(line) + ": unrecognized character '" + one + "'");
    }
    push(Tok::End, "");
    return ts;
}

// ──────────────────────────────────────────────────────────── AST ───
struct Expr; using ExprP = shared_ptr<Expr>;
struct Arg { string kw; ExprP e; };
struct Expr {
    enum K { Num, Vec, Ident, Call, Bin, Mix, Sample, Push, Dims, Transport, KernelDef, DrawFor } k;
    double num = 0;
    vector<double> vec;
    string id;                                  // Ident / Call name / Bin op
    vector<Arg> args;                           // Call
    vector<ExprP> parts;                        // Mix
    ExprP a, b;                                 // Bin / Sample / Push
    int n = 0;                                  // Sample count
    ExprP via;                                  // Sample via call
    vector<int> dims;                           // Dims
    ExprP t_from, t_to, t_using;                // Transport
    int t_steps = 5000; double t_lr = 1e-3; int t_batch = 128;
    string kd_param;                            // KernelDef: kernel(param){ ... } (spec 10.10)
    vector<pair<string, ExprP>> kd_body;        // KernelDef/DrawFor: bindings; the LAST value is the output
    int df_lo = 0, df_hi = 0;                   // DrawFor: literal bounds (draw-level for-as-expression)
};
struct Stmt {
    enum K { Seed, Bind, BindPair, Plot, PlotSignal, For } k;
    long seed = 0;
    string name, name2; ExprP e;          // BindPair: (name, name2) = e — destructure one joint draw
    vector<ExprP> plots; vector<bool> is_traj;
    int for_lo = 1, for_hi = 1;           // bounded for (spec 10.1): literal bounds only
    vector<Stmt> body;
    int line = 1;
};

// ─────────────────────────────────────────────────────────── parser ──
struct Parser {
    vector<Token> ts; size_t p = 0;
    Token& cur() { return ts[p]; }
    Token& nxt() { return ts[p + 1 < ts.size() ? p + 1 : p]; }
    bool isSym(const string& s) { return cur().kind == Tok::Sym && cur().s == s; }
    bool isId(const string& s) { return cur().kind == Tok::Ident && cur().s == s; }
    Token eat() { return ts[p++]; }
    void expectSym(const string& s) {
        if (!isSym(s)) err("line " + to_string(cur().line) + ": expected '" + s + "', found '" + cur().s + "'");
        p++;
    }
    string expectIdent() {
        if (cur().kind != Tok::Ident) err("line " + to_string(cur().line) + ": expected an identifier");
        return eat().s;
    }
    double expectNum() {
        if (cur().kind != Tok::Num) err("line " + to_string(cur().line) + ": expected a number");
        return eat().num;
    }

    vector<Stmt> program() {
        vector<Stmt> out;
        while (cur().kind != Tok::End) out.push_back(statement());
        return out;
    }

    Stmt statement() {
        Stmt s; s.line = cur().line;
        if (isId("while") || isId("break") || isId("continue") || isId("if") || isId("else"))
            err("line " + to_string(s.line) + ": " + cur().s + " does not exist — a program must know what it "
                "will do before it runs (spec 10.1). Bounded iteration is for k in 1..K with a literal K; "
                "data-dependent control flow belongs to the host layer.");
        if (isId("for")) {                // bounded for (spec 10.1): macro-expansion semantics
            eat(); s.k = Stmt::For;
            s.name = expectIdent();
            if (s.name == "t")
                err("line " + to_string(s.line) + ": t is a reserved symbol (the time of a path formula) and cannot be a loop index");
            if (!isId("in")) err("line " + to_string(s.line) + ": expected 'in' — the loop form is: for k in 1..K { ... }");
            eat();
            auto literal = [&](const char* which) {
                if (cur().kind != Tok::Num || cur().num != (long)cur().num)
                    err("line " + to_string(s.line) + ": the " + string(which) + " bound of for must be a literal "
                        "integer — the checker sees the fully expanded program, so data-dependent loop bounds "
                        "do not exist (spec 10.1)");
                return (int)eat().num;
            };
            s.for_lo = literal("lower");
            expectSym("..");
            s.for_hi = literal("upper");
            if (s.for_hi < s.for_lo)
                err("line " + to_string(s.line) + ": empty range " + to_string(s.for_lo) + ".." + to_string(s.for_hi) +
                    " — for iterates at least once (lo <= hi)");
            if (s.for_hi - s.for_lo + 1 > 64)
                err("line " + to_string(s.line) + ": " + to_string(s.for_hi - s.for_lo + 1) +
                    " iterations exceed the cap of 64 (spec 10.1: the body is macro-expanded K times; "
                    "the whole program must stay a small finite DAG)");
            expectSym("{");
            while (!isSym("}")) {
                if (cur().kind == Tok::End)
                    err("line " + to_string(s.line) + ": unclosed for block — expected }");
                s.body.push_back(statement());
            }
            eat();
            return s;
        }
        if (isId("seed")) { eat(); s.k = Stmt::Seed; s.seed = (long)expectNum(); return s; }
        if (isId("plot")) {
            eat(); s.k = Stmt::Plot;
            do {
                if (isId("trajectory")) { eat(); if (!isId("of")) err("expected 'of' after 'plot trajectory'"); eat();
                    s.plots.push_back(expr()); s.is_traj.push_back(true); }
                else { s.plots.push_back(expr()); s.is_traj.push_back(false); }
            } while (isSym(",") && (eat(), true));
            return s;
        }
        if (isId("plot_signal")) {        // draw columns in index order: each coordinate row = one line
            eat(); s.k = Stmt::PlotSignal;
            do { s.plots.push_back(expr()); } while (isSym(",") && (eat(), true));
            return s;
        }
        if (isSym("(")) {                 // (y, x) = joint — destructure one draw (spec 10.8)
            eat();
            s.k = Stmt::BindPair;
            s.name = expectIdent(); expectSym(","); s.name2 = expectIdent(); expectSym(")");
            expectSym("="); s.e = expr();
            return s;
        }
        s.k = Stmt::Bind; s.name = expectIdent(); expectSym("=");
        s.e = expr();
        return s;
    }

    ExprP expr() { return mixture(); }

    ExprP mixture() {                 // w1*A or w2*B — probabilistic disjunction
        auto first = additive();
        if (!isId("or")) return first;
        auto e = make_shared<Expr>(); e->k = Expr::Mix;
        e->parts.push_back(first);
        while (isId("or")) { eat(); e->parts.push_back(additive()); }
        return e;
    }

    ExprP additive() {
        auto lhs = mult();
        while (isSym("+") || isSym("-")) {
            string op = eat().s;
            auto e = make_shared<Expr>(); e->k = Expr::Bin; e->id = op; e->a = lhs; e->b = mult();
            lhs = e;
        }
        return lhs;
    }

    ExprP mult() {
        auto lhs = push();
        while (isSym("*") || isSym("/")) {
            string op = eat().s;
            auto e = make_shared<Expr>(); e->k = Expr::Bin; e->id = op; e->a = lhs; e->b = push();
            lhs = e;
        }
        return lhs;
    }

    ExprP push() {
        auto lhs = cond();
        while (isSym("#")) {
            eat();
            auto e = make_shared<Expr>(); e->k = Expr::Push; e->a = lhs; e->b = cond();
            lhs = e;
        }
        return lhs;
    }

    ExprP cond() {                    // y | x — conditioning (spec 10.8); binds
        auto lhs = postfix();         // tighter than #, so T # y0|c0 = T # (y0|c0)
        while (isSym("|")) {
            eat();
            auto e = make_shared<Expr>(); e->k = Expr::Bin; e->id = "|"; e->a = lhs; e->b = postfix();
            lhs = e;
        }
        return lhs;
    }

    ExprP postfix() {
        auto e = primary();
        for (;;) {
            if (isSym("~")) {
                eat();
                auto s = make_shared<Expr>(); s->k = Expr::Sample; s->a = e; s->n = (int)expectNum();
                if (isId("via")) { eat(); s->via = primary(); }
                e = s;
            } else break;
        }
        return e;
    }

    ExprP primary() {
        if (isSym("-")) {   // unary minus: -e ≡ 0 - e
            eat();
            auto z = make_shared<Expr>(); z->k = Expr::Num; z->num = 0;
            auto e = make_shared<Expr>(); e->k = Expr::Bin; e->id = "-"; e->a = z; e->b = primary();
            return e;
        }
        if (isId("transport")) return transport();
        if (cur().kind == Tok::Num) {
            auto e = make_shared<Expr>(); e->k = Expr::Num; e->num = eat().num; return e;
        }
        if (isSym("[")) {
            eat(); auto e = make_shared<Expr>(); e->k = Expr::Vec;
            if (!isSym("]")) { e->vec.push_back(expectNum()); while (isSym(",")) { eat(); e->vec.push_back(expectNum()); } }
            expectSym("]"); return e;
        }
        if (isSym("(")) { eat(); auto e = expr(); expectSym(")"); return e; }
        if (cur().kind == Tok::Ident) {
            int name_line = cur().line;
            string name = eat().s;
            // kernel(param) { name = expr; ... } — a programmable Markov
            // kernel (spec 10.10): the body is a straight-line program
            // defined POINTWISE in the parameter, vectorized over a batch
            // at instantiation. The last binding's value is the output y.
            if (name == "kernel" && isSym("(")) {
                eat();
                auto e = make_shared<Expr>(); e->k = Expr::KernelDef;
                e->kd_param = expectIdent();
                if (e->kd_param == "t")
                    err("line " + to_string(cur().line) + ": t is a reserved symbol (the time of a path formula) "
                        "and cannot be a kernel parameter");
                expectSym(")");
                expectSym("{");
                // bounded for INSIDE a body (spec 10.10, 2026-07): at draw
                // level `for` is an EXPRESSION — its value is the row-stack
                // of each iteration's final binding (an N-observation block
                // is exactly ⊕ᵢ yᵢ). Macro expansion, literal bounds, cap
                // 64, no nesting; the lift scope resets per copy (each
                // round's distributions are FRESH draws — iid, literally).
                auto drawfor = [&](auto&& self) -> ExprP {
                    eat();                                   // 'for'
                    auto f = make_shared<Expr>(); f->k = Expr::DrawFor;
                    f->id = expectIdent();
                    if (f->id == "t") err("t is a reserved symbol and cannot be a loop index");
                    if (f->id == e->kd_param)
                        err("the loop index '" + f->id + "' would shadow the kernel parameter — pick another name");
                    if (!isId("in")) err("expected 'in' — the loop form is: for i in 1..K { ... }");
                    eat();
                    auto literal = [&](const char* which) {
                        if (cur().kind != Tok::Num || cur().num != (long)cur().num)
                            err(string("the ") + which + " bound of a body for must be a literal integer "
                                "(the checker sees the fully expanded body, spec 10.1/10.10)");
                        return (int)eat().num;
                    };
                    f->df_lo = literal("lower");
                    expectSym("..");
                    f->df_hi = literal("upper");
                    if (f->df_hi < f->df_lo) err("empty range — for iterates at least once (lo <= hi)");
                    if (f->df_hi - f->df_lo + 1 > 64)
                        err(to_string(f->df_hi - f->df_lo + 1) + " iterations exceed the cap of 64 (spec 10.1)");
                    expectSym("{");
                    while (!isSym("}")) {
                        if (cur().kind == Tok::End) err("unclosed body for — expected }");
                        string bn = expectIdent();
                        if (bn == "t") err("t is a reserved symbol and cannot be bound in a kernel body");
                        expectSym("=");
                        if (isId("for"))
                            err("body for does not nest in v1 — one stacking level; widen the outer loop instead");
                        f->kd_body.emplace_back(bn, expr());
                    }
                    eat();
                    if (f->kd_body.empty()) err("an empty body for stacks nothing — bind at least one draw");
                    (void)self;
                    return f;
                };
                while (!isSym("}")) {
                    if (cur().kind == Tok::End)
                        err("unclosed kernel body — expected }");
                    string nm = expectIdent();
                    if (nm == "t") err("t is a reserved symbol and cannot be bound in a kernel body");
                    expectSym("=");
                    if (isId("for")) e->kd_body.emplace_back(nm, drawfor(drawfor));
                    else             e->kd_body.emplace_back(nm, expr());
                }
                eat();
                if (e->kd_body.empty())
                    err("a kernel body needs at least one binding — the LAST binding's value is the kernel's output y (spec 10.10)");
                return e;
            }
            // a call's ( must open on the identifier's own line: statements
            // are newline-free, so `plot_signal Q` followed by a destructure
            // `(y, w) = ...` would otherwise read as the call Q(y, w)
            if (isSym("(") && cur().line == name_line) {
                eat();
                auto e = make_shared<Expr>(); e->k = Expr::Call; e->id = name;
                if (cur().kind == Tok::Num && nxt().kind == Tok::Sym && nxt().s == "->") {
                    e->k = Expr::Dims;
                    e->dims.push_back((int)expectNum());
                    while (isSym("->")) { eat(); e->dims.push_back((int)expectNum()); }
                    e->id = name;
                    expectSym(")"); return e;
                }
                if (!isSym(")")) {
                    do {
                        Arg a;
                        if (cur().kind == Tok::Ident && nxt().kind == Tok::Sym && nxt().s == "=") {
                            a.kw = eat().s; eat();
                        }
                        a.e = expr(); e->args.push_back(move(a));
                    } while (isSym(",") && (eat(), true));
                }
                expectSym(")");
                return e;
            }
            auto e = make_shared<Expr>(); e->k = Expr::Ident; e->id = name; return e;
        }
        err("line " + to_string(cur().line) + ": cannot parse expression starting at '" + cur().s + "'");
    }

    ExprP transport() {
        eat();
        if (!isId("from")) err("expected 'from' after 'transport'"); eat();
        auto e = make_shared<Expr>(); e->k = Expr::Transport;
        e->t_from = expr();
        if (!isId("to")) err("expected 'to' in a transport expression"); eat();
        e->t_to = expr();
        if (!isId("using")) err("expected 'using' in a transport expression"); eat();
        e->t_using = primary();
        if (isId("for")) { eat(); e->t_steps = (int)expectNum(); if (!isId("steps")) err("expected 'steps' after 'for N'"); eat(); }
        if (isId("with")) {
            eat();
            do { string k = expectIdent(); expectSym("="); double v = expectNum();
                 if (k == "lr") e->t_lr = v; else if (k == "batch") e->t_batch = (int)v;
                 else err("transport with: unknown parameter " + k);
            } while (isSym(",") && (eat(), true));
        }
        return e;
    }
};

// ─────────────────────────────────────────────────────────── values ──
struct MapV;
// Provenance strings are identity for the static checks, but under bounded
// for-loops they compose recursively (each round's canon embeds the previous
// round's several times — exponential growth, real OOMs). Cap them: long
// canons keep a readable prefix plus a deterministic FNV-1a hash of the full
// string, so identity survives (equal full strings ⇒ equal capped strings,
// and collisions are negligible) while length stays bounded.
static string canon_cap(string s) {
    const size_t CAP = 512;               // generous: real one-shot canons stay intact;
    if (s.size() <= CAP) return s;        // only loop-composed canons (3^k growth) get capped
    unsigned long long h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    char hex[20]; snprintf(hex, sizeof hex, "%016llx", h);
    return s.substr(0, 400) + "…⟨" + hex + "⟩";
}

// Canonical printout of an expression AST — the identity of programmable
// kernel bodies (spec 10.10). Free identifiers print by NAME (late binding:
// they resolve from the program environment at instantiation).
static string expr_canon(const shared_ptr<Expr>& e) {
    ostringstream o;
    switch (e->k) {
    case Expr::Num: o << e->num; break;
    case Expr::Vec: { o << "["; for (size_t i = 0; i < e->vec.size(); i++) o << (i ? "," : "") << e->vec[i]; o << "]"; break; }
    case Expr::Ident: o << e->id; break;
    case Expr::Call: {
        o << e->id << "(";
        for (size_t i = 0; i < e->args.size(); i++) {
            o << (i ? "," : "");
            if (!e->args[i].kw.empty()) o << e->args[i].kw << "=";
            o << expr_canon(e->args[i].e);
        }
        o << ")"; break;
    }
    case Expr::Bin: o << "(" << expr_canon(e->a) << e->id << expr_canon(e->b) << ")"; break;
    case Expr::Push: o << "(" << expr_canon(e->a) << "#" << expr_canon(e->b) << ")"; break;
    case Expr::Sample: o << "(" << expr_canon(e->a) << "~" << e->n << ")"; break;
    case Expr::Mix: { for (size_t i = 0; i < e->parts.size(); i++) o << (i ? " or " : "") << expr_canon(e->parts[i]); break; }
    case Expr::DrawFor: {
        o << "for " << e->id << " in " << e->df_lo << ".." << e->df_hi << "{";
        for (size_t i = 0; i < e->kd_body.size(); i++)
            o << (i ? ";" : "") << e->kd_body[i].first << "=" << expr_canon(e->kd_body[i].second);
        o << "}"; break;
    }
    default: o << "<expr>"; break;
    }
    return o.str();
}

struct Dataset {
    Mat X;                       // d × n
    string prov;
    vector<Mat> traj;
    int blk = 0;                 // kernel-sampled joints (spec 10.8/10.10): the parameter
                                 //   block starts at row blk ((y; w) layout, observed rows
                                 //   first); 0 = not a kernel joint. Lets (y, w) = (K|z) ~ n
                                 //   destructure the frozen table by rows.
    shared_ptr<Dataset> wsrc;    // window()/lagsplit() provenance: the source TRAJECTORY and
    int wlen = 0;                //   window length — what lets flow(into=) re-simulate the
                                 //   window/lagsplit/decouple assembly from a generator's own
                                 //   output (the amortized instantaneous demixer, SICA line)
    Dataset(Mat x, string p) : X(move(x)), prov(canon_cap(move(p))) {}
};

struct Dist {
    struct Comp { string kind; vector<float> mean; float s1 = 1, s2 = 0; float w = 1; };
    vector<Comp> comps;
    string canon;
    int dim = 2;
    int dy = 0, dx = 0;                   // joint distributions: coordinate blocks (rows [0,dy) = y, [dy,dy+dx) = x); dy=0 means not a joint
    bool has_score = false;
    bool samplable = true;
    bool has_cond_sampler = false;        // joints: can sample y | x = z in closed form
    bool pair_blocks = false;             // couple (spec 10.3): the two blocks are PEER coordinates
                                          //   of one paired draw — both may appear as formula terms
    shared_ptr<Dist>    cpl_a, cpl_b;     // couple only: the two marginals (Dist or Dataset per side)
    shared_ptr<Dataset> cpl_ax, cpl_bx;
    shared_ptr<MapV>    cpl_map;          // via=paired (reflow): b-side IS map(a-draw) — one draw, pushed
    shared_ptr<Dataset> dcp_src;          // decouple(X): product of X's empirical marginals
    int dcp_block = 1;                    // decouple block size: rows [b*L,(b+1)*L) bootstrap JOINTLY
                                          //   (process independence: within-block structure survives)
    string cpl_via;                       // couple only: independent | ot | sinkhorn | paired
    float  cpl_eps = 0.1f;                // sinkhorn temperature (cost normalized by its mean)
    function<float(float, float)> logp;   // unnormalized targets: log density up to +const
};

struct NetSpec { vector<int> dims; };

struct TrainedNet {
    vector<unique_ptr<Layer<float>>> owned;
    list<Layer<float>*> net;
    int d_in = 0, d_out = 0;
    Mat eval(const Mat& inp) { return forward(net, inp); }
};

struct DivV;

struct FieldV {
    enum Kind { FM, REV, WG, COMBO, AMORT, ROTFM } kind;
                                     // ROTFM: rotation-constrained regression of a declared
                                     //   path (estimator=regress(rotation)) — the field is
                                     //   skew-linear, x -> Omega(t)x, closed form, no net
                                     // FM: net(x,t) is the velocity; REV: DSM prob-flow;
                                     // WG: Wasserstein steepest-descent field of a divergence
                                     //     (nw estimator; simulation interleaves estimation)
                                     // COMBO: linear combination of self-contained fields
                                     //        (spec 10.3.1 field algebra; synthetic provenance)
                                     // AMORT: drifting-trained one-step generator net(z)
                                     //        (flow(v, into=net); spec 10.6 — not a velocity!)
    shared_ptr<TrainedNet> net;
    shared_ptr<DivV> div;            // WG only
    bool nw_norm = false;            // WG only: normalized NW (consistent W2 velocity)
    bool rotation = false;           // WG only (family=rotation): velocity projected onto so(d) —
                                     //   the flow moves on the orthogonal orbit O(d)·q0 (ICA line)
    int rot_block = 1;               // family=rotation(block=L): R ⊗ I_L (channel rotations
                                     //   shared across lags — process ICA on window()ed data)
    int free_lo = -1, free_hi = -1;  // WG only (from=(y|x)): only the y-block rows move —
                                     //   the free rows of the joint field descend the
                                     //   CONDITIONAL KL (see WGPathV); -1 = unconditioned
    shared_ptr<Dataset> obs;         // WG only (from=(q|Obs), 2026-07 KL decomposition): the
    int obs_lo = -1, obs_hi = -1;    //   observation source; rows [obs_lo, obs_hi), one column
                                     //   per observation. The field is Σ_i v_i − (N−1)·v_0
                                     //   (see WGPathV::obs for the theorem). null = plain.
    bool dsm = false;                // WG only (estimator=dsm, 2026-07): neural score descent —
    shared_ptr<NetSpec> dsm_spec;    //   fixed-sigma DSM nets replace the NW kernels: the target
    float dsm_sigma = 0.1f;          //   joint's score net is trained once, the moving cloud's is
    float dsm_trainlr = 1e-3f;       //   WARM-STARTED (dsm_warm SGD steps per flow step). Same
    int dsm_warm = 16;               //   sigma on both sides, so the smoothing bias cancels where
    int dsm_pre = 1500;              //   p_sigma = q_sigma — the neural mirror of the shared-h
    int dsm_batch = 128;             //   rule. Consumes RNG every step (noise draws), unlike nw.
    bool residual = false;           // AMORT only (Level 1, SICA line): g(x) = x + net(x) — the
                                     //   amortized INSTANTANEOUS demixer. Trained by structured
                                     //   re-simulation (window/lagsplit/decouple rebuilt from the
                                     //   net's own output each optimizer step); a full function of
                                     //   R^d, applied freely (no from= pin — being holdable and
                                     //   reusable is the point).
    int ydim = 0, cond_dim = 0;      // conditional fields (spec 10.8): net input is (y, conditions, t); cond_dim = total condition rows (0 = unconditional)
    string from_prov;                // WG/AMORT: the path's pinned initial measure
    shared_ptr<Dist> wg_from_d;      // WG only: the actual from= measure (into= samples
    shared_ptr<Dataset> wg_from_x;   //   its latent batches from it, spec 10.6)
    vector<pair<float, shared_ptr<FieldV>>> terms;   // COMBO only: (coeff, leaf)
    vector<Mat> rot_knots;           // ROTFM only: Omega(t_k) on a uniform knot grid over [0,1]
    string start_prov;               // FM/COMBO: the declared path's t=0 marginal ("" = unknown)
    int dim = 2;
    float T = 3.0f;
    string desc;
    Mat vel(const Mat& X, float s) { return vel_rows(X, Mat::ones(1, (int)X.num_col()) * s); }
    Mat vel_rows(const Mat& X, const Mat& srow) {    // per-column times (training-loop queries)
        int d = (int)X.num_row(), n = (int)X.num_col();
        if (kind == ROTFM) {                          // skew-linear field: v = Omega(t) x
            Mat R("rotv", d, n);
            for (int j = 0; j < n; j++) {
                int k = (int)(srow.elem(0, j) * (float)(rot_knots.size() - 1) + 0.5f);
                k = max(0, min((int)rot_knots.size() - 1, k));
                for (int r = 0; r < d; r++) {
                    float s = 0;
                    for (int c = 0; c < d; c++) s += rot_knots[k].elem(r, c) * X.elem(c, j);
                    R.elem(r, j) = s;
                }
            }
            return R;
        }
        if (kind == COMBO) {
            Mat acc = terms[0].second->vel_rows(X, srow) * terms[0].first;
            for (size_t i = 1; i < terms.size(); i++)
                acc = acc + terms[i].second->vel_rows(X, srow) * terms[i].first;
            return acc;
        }
        if (kind == FM) return net->eval(vstack<float>({X, srow}));
        Mat trow("t", 1, n), isig("isig", 1, n);
        for (int j = 0; j < n; j++) {
            float t = max(T * (1.0f - srow.elem(0, j)), 0.02f);
            trow.elem(0, j) = t;
            isig.elem(0, j) = 1.0f / sqrt(max(1e-6f, 1.0f - exp(-2.0f * t)));
        }
        auto epshat = net->eval(vstack<float>({X, trow}));
        return (X - hadmd(epshat, Mat::ones(d, 1) * isig)) * T;
    }
    // conditional field (spec 10.8): velocity of the y-block, conditions ride
    // along as extra input rows — v(y_t; conditions, t)
    Mat vel_cond(const Mat& Y, const Mat& C, const Mat& srow) {
        return net->eval(vstack<float>({Y, C, srow}));
    }
};

struct MapV {
    shared_ptr<FieldV> f;
    int steps = 50;
    float lr = 0.5f;                 // WG fields only: particle step size
    bool inverse = false;
    bool invertible = true;          // WG maps: invertible only with record=true
    bool proj_instant = false;       // WG conditional descent (2026-07, SICA line): before each
                                     //   step, Rao-Blackwell the per-column conditional field onto
                                     //   a function of the ELEMENTS alone (closed-form polynomial
                                     //   ridge) — every step becomes one shared INSTANTANEOUS map,
                                     //   the class the identifiability theorem certifies. Works for
                                     //   score-type fields only: Fisher's identity collapses the
                                     //   repulsion to the marginal score but leaves the attraction's
                                     //   cross-channel residual (an RF/conditional-mean field would
                                     //   tower-collapse to the degenerate marginal fit).
    bool record = false;             // WG: store per-step ensembles on forward apply
    shared_ptr<vector<Mat>> hist;    // recorded ensembles X_0..X_{K-1}
    shared_ptr<vector<float>> hist_h;
    shared_ptr<string> endpoint_prov;  // provenance of the recorded forward endpoint
    string desc;
    Mat apply(Mat X, vector<Mat>* traj = nullptr);   // defined after svgd_step
    // conditional maps: Euler on the y-block, condition rows C fixed per sample
    Mat apply_cond(Mat Y, const Mat& C, vector<Mat>* traj = nullptr) {
        if (traj) traj->push_back(Y);
        float ds = 1.0f / steps;
        for (int i = 0; i < steps; i++) {
            float s = inverse ? 1.0f - (float)i / steps : (float)i / steps;
            auto srow = Mat::ones(1, (int)Y.num_col()) * s;
            auto V = f->vel_cond(Y, C, srow);
            Y = inverse ? Y - V * ds : Y + V * ds;
            if (traj && ((i + 1) % max(1, steps / 12) == 0 || i == steps - 1)) traj->push_back(Y);
        }
        return Y;
    }
};

// transported kernel (spec 10.8): `T # (Y | X)` keeps the condition slots
// open; `| W` instantiates them. inst: 0 open, 1 paired (W is the block
// bound with the y-source — draw identity), 2 fixed point, 3 measure, 4 dataset.
struct KernelV {
    shared_ptr<MapV> map;
    shared_ptr<Dist> joint;          // the source joint (y, x)
    long src = 0;                    // draw identity of the source pair
    int inst = 0;
    vector<float> zfix;              // inst=2
    shared_ptr<Dist> zdist;          // inst=3
    shared_ptr<Dataset> zdata;       // inst=4
    // programmable Markov kernel (spec 10.10): the body is a straight-line
    // draw-level program p(y | param); no map, no source joint, no density —
    // samples only. Instantiation reuses the same inst modes above (paired
    // inst=1 excluded: there is no source joint whose draw could be reused).
    bool prog = false;
    string kd_param;
    vector<pair<string, ExprP>> kd_body;
    string canon;
};

struct DivV {
    string family = "f_div", name;
    float w2_eps = 0.1f;              // w2 divergence: sinkhorn temperature (cost normalized by its mean)
    shared_ptr<Dist>    target_d;
    shared_ptr<Dataset> target_x;
    string moving_prov;
    string canon;
};

// implicit path: W2-geometry steepest descent of a divergence (IVP), spec §10.2.1
struct WGPathV {
    shared_ptr<DivV> div;
    shared_ptr<Dist> from_d;         // exactly one of from_d/from_x set
    shared_ptr<Dataset> from_x;
    string from_prov;
    float T = 3.0f;
    string metric = "w2";
    bool rotation = false;           // family=rotation: the CURVE is constrained to the
                                     //   orbit {R # from : R in SO(d)} (projected field
                                     //   and Lie-group integration follow as theorems)
    int rot_block = 1;               // family=rotation(block=L): rotations act on CHANNELS,
                                     //   identically at each of the L lags (R ⊗ I_L) —
                                     //   the process-ICA constraint on window()ed data
    int free_lo = -1, free_hi = -1;  // from=(y | x) (2026-07, SBI line): the ensemble is a
                                     //   frozen JOINT and only the y-block rows [free_lo,
                                     //   free_hi) move; the conditioning block is pinned.
                                     //   Pinning IS conditioning for reverseKL:
                                     //   ∇_y log p(y|x) = ∇_y log p(y,x) (the log splits;
                                     //   ∇_y kills log p(x)), so the free rows of the joint
                                     //   nw field descend the conditional KL exactly —
                                     //   conditional SVGD. -1 = unconditioned (all rows move).
                                     //   Estimator revision (2026-07): for a SAMPLES target
                                     //   the pinned-row kernel plays the LIKELIHOOD and gets
                                     //   the sharp hy rule (likelihood_bandwidths) as frozen
                                     //   per-particle library weights; free rows keep the
                                     //   pooled rule over free rows only. Score targets need
                                     //   no window (exact scores at pinned queries).
    shared_ptr<Dataset> obs;         // from=(q | Obs) (2026-07, KL decomposition): q is a plain
    int obs_lo = -1, obs_hi = -1;    //   parameter-space ensemble and Obs holds N FOREIGN
                                     //   observation columns (rows [obs_lo, obs_hi) of obs).
                                     //   Bayes + log-likelihood additivity split the target:
                                     //     KL[q ‖ p(·|Y_1..N)]
                                     //       = Σ_i KL[q ‖ p(·|Y_i)] − (N−1)·KL[q ‖ p_prior] + c.
                                     //   For KDE estimates the additivity must land in the
                                     //   log-WEIGHTS of one attraction, not as a signed sum of
                                     //   N+1 fields (three measured failures — kde_score_at):
                                     //   log W_j = Σ_i log L̂_i(w_j) on the library's parameter
                                     //   rows, then a plain descent onto the weighted prior
                                     //   KDE. The split is a property of the LOG — reverseKL
                                     //   only.
    string canon;
};

struct PushedDist { shared_ptr<MapV> map; shared_ptr<Dist> base; string canon; };

// random-variable layer (probability paths)
struct RTerm {
    TF coeff;                    // c(t)
    string cdesc;                // for provenance strings
    long src = 0;                // draw identity; equal src ⇒ same draw
    bool anon = true;            // auto-lifted (no rv() name)?
    shared_ptr<Dist> dist;       // exactly one of dist/data set
    shared_ptr<Dataset> data;
    int blo = -1, bhi = -1;      // coordinate block of a joint draw ([blo,bhi) rows); -1 = whole vector
    int clo = -1, chi = -1;      // conditioner block (same draw, spec 10.8); -1 = unconditional term
    shared_ptr<Dataset> obsd;    // N-observation conditioning (2026-07, KL decomposition):
    int olo = -1, ohi = -1;      //   (q | Obs) — data is the moving ensemble, obsd rows
                                 //   [olo, ohi) are the observation columns Y_1..Y_N. Only
                                 //   descent(from=) and # consume this term; it has no law.
    bool peer = false;           // couple block (spec 10.3): peer coordinate of one paired draw —
                                 //   two peer blocks of the same src coexist as separate formula terms
    const void* origin() const { return dist ? (const void*)dist.get() : (const void*)data.get(); }
    string srcname() const {
        if (dist && dist->pair_blocks && blo >= 0 &&   // couple blocks: name the marginal, not the pair
            (dist->cpl_a || dist->cpl_ax))             // (constructed pair-joints store no marginals)
            return blo == 0 ? (dist->cpl_a ? dist->cpl_a->canon : dist->cpl_ax->prov)
                            : (dist->cpl_b ? dist->cpl_b->canon : dist->cpl_bx->prov);
        string base = dist ? dist->canon : data->prov;
        if (blo >= 0) base += (blo == 0 ? ":y" : ":x");
        return base;
    }
    int dim() const {
        if (blo >= 0) return bhi - blo;
        return dist ? dist->dim : (int)data->X.num_row();
    }
    int cdim() const { return clo >= 0 ? chi - clo : 0; }
};
struct RVal { vector<RTerm> terms; };
struct PathV { shared_ptr<RVal> rv; string canon; };

struct Value {
    enum K { Num, Vec, DistV, Data, Net, Field, Map, Div, WGPath, Pushed, Kernel,
             Symbol, TFun, XYFun, RV, Path } k;
    double num = 0; vector<double> vec;
    shared_ptr<Dist> dist; shared_ptr<Dataset> data; shared_ptr<NetSpec> net;
    shared_ptr<FieldV> field; shared_ptr<MapV> map; shared_ptr<DivV> div;
    shared_ptr<WGPathV> wgpath; shared_ptr<PushedDist> pushed;
    shared_ptr<KernelV> kernel;
    string sym;
    TF tf; string tfdesc;
    function<float(float, float)> xy; string xydesc;
    shared_ptr<RVal> rv;
    shared_ptr<PathV> path;
};

static long fresh_src() { static long c = 0; return ++c; }

// ─────────────────────────────── field algebra (spec 10.3.1) ────────
// Fields form a vector space; a*v1 + b*v2 is legal for self-contained
// fields. The combination is SYNTHETIC: it is field(p) of no declared
// path — the language makes no claim about what its flow samples
// (classifier-free guidance lives exactly here, and honestly so).
static void field_algebra_leaf_check(const shared_ptr<FieldV>& f) {
    if (f->cond_dim > 0)
        err("field algebra with conditional fields is not yet supported (spec 10.8)");
    if (f->kind == FieldV::WG)
        err("field algebra combines self-contained fields only — a descent field is re-estimated from the "
            "evolving ensemble each step and has no standalone evaluation. On the descent side, combine "
            "upstream at the divergence level: descent(a*D1 + b*D2) is the steepest descent of the combined "
            "divergence, and (unlike CFG) its stationary law is exactly the geometric mixture (spec 10.3.1).");
}
static Value field_lincomb(vector<pair<float, shared_ptr<FieldV>>> ts) {
    // flatten happened at the call site; merge repeated leaves so the
    // provenance string shows the net affine weights
    vector<pair<float, shared_ptr<FieldV>>> m;
    for (auto& t : ts) {
        bool found = false;
        for (auto& u : m)
            if (u.second.get() == t.second.get()) { u.first += t.first; found = true; break; }
        if (!found) m.push_back(t);
    }
    // drop cancelled terms: v_u + 1*(v_c - v_u) collapses to v_c itself,
    // so the w=1 theorem (guidance = the conditional field, exact) is
    // visible in the provenance rather than buried under a 0-weight term
    m.erase(remove_if(m.begin(), m.end(),
                      [](const pair<float, shared_ptr<FieldV>>& t) { return fabs(t.first) < 1e-7f; }),
            m.end());
    for (auto& t : m) field_algebra_leaf_check(t.second);
    if (m.empty())
        err("this combination cancels to the zero field (its flow is the identity map) — almost surely not what was meant");
    // dims must agree
    for (auto& t : m)
        if (t.second->dim != m[0].second->dim)
            err("cannot combine fields of different dimensions (" + to_string(m[0].second->dim) +
                " vs " + to_string(t.second->dim) + ")");
    // starting measures must agree when both are known (10.3.1 provenance check)
    string sp;
    for (auto& t : m) {
        const string& s = t.second->start_prov;
        if (s.empty()) continue;
        if (sp.empty()) { sp = s; continue; }
        if (s != sp)
            err("these fields transport different initial measures (" + sp + " vs " + s + "): "
                "a combined field has no coherent starting point. Guidance combines fields whose paths "
                "share the same source — declare both interpolation formulas from the same noise (spec 10.3.1).");
    }
    if (m.size() == 1 && fabs(m[0].first - 1.0f) < 1e-9f) {   // degenerate: just the field itself
        Value r; r.k = Value::Field; r.field = m[0].second; return r;
    }
    auto fv = make_shared<FieldV>(); fv->kind = FieldV::COMBO;
    fv->terms = m; fv->dim = m[0].second->dim; fv->start_prov = sp;
    ostringstream d; d << "(";
    for (size_t i = 0; i < m.size(); i++)
        d << (i ? " + " : "") << m[i].first << "*[" << m[i].second->desc << "]";
    d << ")";
    fv->desc = d.str();
    Value r; r.k = Value::Field; r.field = fv; return r;
}
static vector<pair<float, shared_ptr<FieldV>>> field_terms(const shared_ptr<FieldV>& f, float c) {
    if (f->kind == FieldV::COMBO) {
        auto ts = f->terms;
        for (auto& t : ts) t.first *= c;
        return ts;
    }
    return {{c, f}};
}

// ─────────────────────────────────────────────── sampling & scores ──
// Exact assignment for the OT coupling (spec 10.3): Hungarian / Jonker-
// Volgenant with potentials, O(n^3), deterministic. Returns, for each
// column j of B, the row i of A it is matched to (a permutation).
static vector<int> ot_assign(const vector<vector<float>>& C) {
    int n = (int)C.size();
    const float INF = 1e30f;
    vector<float> u(n + 1), v(n + 1);
    vector<int> p(n + 1), way(n + 1);
    for (int i = 1; i <= n; i++) {
        p[0] = i; int j0 = 0;
        vector<float> minv(n + 1, INF);
        vector<char> used(n + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0; float delta = INF;
            for (int j = 1; j <= n; j++) if (!used[j]) {
                float cur = C[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    vector<int> row_of_col(n);
    for (int j = 1; j <= n; j++) row_of_col[j - 1] = p[j] - 1;
    return row_of_col;
}

static Mat sample_dist(const Dist& D, int n);

// One draw of a coupled pair (spec 10.3): sample n from each marginal, pair
// WITHIN the batch, return the pair stacked ([0,dy) = a-block, [dy,dy+dx) = b).
// The pairing itself is deterministic given the two draws (Hungarian); the
// sinkhorn plan is sampled row-wise (n extra uniforms — documented RNG).
static Mat sample_couple(const Dist& D, int n) {
    auto side = [&](const shared_ptr<Dist>& d, const shared_ptr<Dataset>& x) {
        if (d) return sample_dist(*d, n);
        uniform_int_distribution<int> pick(0, (int)x->X.num_col() - 1);
        Mat out("side", (int)x->X.num_row(), n);
        for (int i = 0; i < n; i++) { int j = pick(global_rand_gen);
            for (size_t r = 0; r < x->X.num_row(); r++) out.elem(r, i) = x->X.elem(r, j); }
        return out;
    };
    Mat A = side(D.cpl_a, D.cpl_ax);            // RNG order: a-side first,
    if (D.cpl_via == "paired") {                // reflow (spec 10.1/10.3): ONE draw,
        Mat B = D.cpl_map->apply(A);            // pushed — pairs (z, T(z)) by construction
        Mat X("pair", D.dim, n);
        for (int i = 0; i < n; i++) {
            for (int r = 0; r < D.dy; r++) X.elem(r, i) = A.elem(r, i);
            for (int r = 0; r < D.dx; r++) X.elem(D.dy + r, i) = B.elem(r, i);
        }
        return X;
    }
    Mat B = side(D.cpl_b, D.cpl_bx);            // then b-side, then the plan
    if (D.cpl_map) B = D.cpl_map->apply(B);     // pushforward marginal: fresh draw, pushed
    vector<int> match(n);                       // for column i of A: column of B
    for (int i = 0; i < n; i++) match[i] = i;   // independent: keep the draw order
    if (D.cpl_via != "independent") {
        vector<vector<float>> C(n, vector<float>(n));
        double csum = 0;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            float s = 0;
            for (int r = 0; r < D.dy && r < (int)B.num_row(); r++) {
                // cost in the shared coordinate space (pairing needs dA == dB)
                float df = A.elem(r, i) - B.elem(r, j); s += df * df;
            }
            C[i][j] = s; csum += s;
        }
        if (D.cpl_via == "ot") {
            auto row_of_col = ot_assign(C);
            vector<int> col_of_row(n);
            for (int j = 0; j < n; j++) col_of_row[row_of_col[j]] = j;
            match = col_of_row;
        } else {                                 // sinkhorn(eps): entropic plan
            float mc = (float)(csum / ((double)n * n));
            float eps = max(1e-4f, D.cpl_eps);
            vector<vector<float>> K(n, vector<float>(n));
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
                K[i][j] = expf(-C[i][j] / (mc * eps));   // cost normalized by its mean
            vector<float> u(n, 1.0f), v(n, 1.0f);
            for (int it = 0; it < 100; it++) {
                for (int i = 0; i < n; i++) { float s = 0; for (int j = 0; j < n; j++) s += K[i][j] * v[j]; u[i] = 1.0f / max(s, 1e-30f); }
                for (int j = 0; j < n; j++) { float s = 0; for (int i = 0; i < n; i++) s += K[i][j] * u[i]; v[j] = 1.0f / max(s, 1e-30f); }
            }
            uniform_real_distribution<float> U01(0, 1);
            for (int i = 0; i < n; i++) {        // sample partner j ~ plan row i
                float tot = 0; for (int j = 0; j < n; j++) tot += u[i] * K[i][j] * v[j];
                float r = U01(global_rand_gen) * tot, acc = 0; int j = 0;
                for (; j < n - 1; j++) { acc += u[i] * K[i][j] * v[j]; if (acc >= r) break; }
                match[i] = j;
            }
        }
    }
    Mat X("pair", D.dim, n);
    for (int i = 0; i < n; i++) {
        for (int r = 0; r < D.dy; r++) X.elem(r, i) = A.elem(r, i);
        for (int r = 0; r < D.dx; r++) X.elem(D.dy + r, i) = B.elem(r, match[i]);
    }
    return X;
}

// decouple (dual of couple, any dimension): the product of the empirical
// BLOCK marginals of a Dataset. A draw bootstraps each block of L rows
// jointly — one column pick shared by the rows of a block, independent picks
// across blocks. block=1 (default) is the fully decoupled product (each
// coordinate independent, d*n RNG picks — the RNG order of the original
// decouple, unchanged). block=L on window()ed data is the paper's permuted
// product: each channel keeps its OWN within-window law (its dynamics), only
// the cross-channel dependence is forgotten — PROCESS independence.
static Mat sample_decouple(const Dist& D, int n) {
    const Mat& S = D.dcp_src->X;
    int d = (int)S.num_row(), m = (int)S.num_col(), L = D.dcp_block;
    uniform_int_distribution<int> pick(0, m - 1);
    Mat X("dcp", d, n);
    for (int b = 0; b < d / L; b++)
        for (int i = 0; i < n; i++) {
            int j = pick(global_rand_gen);
            for (int r = b * L; r < (b + 1) * L; r++) X.elem(r, i) = S.elem(r, j);
        }
    return X;
}

static Mat sample_dist(const Dist& D, int n) {
    if (!D.comps.empty() && D.comps[0].kind == "couple") return sample_couple(D, n);
    if (!D.comps.empty() && D.comps[0].kind == "decouple") return sample_decouple(D, n);
    if (!D.comps.empty() && D.comps[0].kind == "arjoint") {
        // TIME-SERIES draw (mixed_ar): one trajectory of two stationary unit-
        // variance Gaussian AR(1) sources, rho = 0.9 and -0.6 (very different
        // spectra), mixed like mixed_sources. RNG: per time step, one normal
        // per channel, interleaved (frozen order); stationary start.
        normal_distribution<float> N01(0, 1);
        float m = D.comps[0].s1;
        const float r1 = 0.9f, r2 = -0.6f;
        const float q1 = sqrtf(1.0f - r1 * r1), q2 = sqrtf(1.0f - r2 * r2);
        Mat X("X", 4, n);
        float s1 = 0, s2 = 0;
        for (int i = 0; i < n; i++) {
            float e1 = N01(global_rand_gen), e2 = N01(global_rand_gen);
            if (i == 0) { s1 = e1; s2 = e2; }                 // stationary init (unit variance)
            else        { s1 = r1 * s1 + q1 * e1; s2 = r2 * s2 + q2 * e2; }
            X.elem(0, i) = s1 + m * s2;                       // pair-joint rows: [0,2) = x, [2,4) = s
            X.elem(1, i) = m * s1 + s2;
            X.elem(2, i) = s1;
            X.elem(3, i) = s2;
        }
        return X;
    }
    if (!D.comps.empty() && D.comps[0].kind == "nljoint") {
        // TIME-SERIES draw (mixed_nl): the SAME Gaussian AR(1) source pair
        // as mixed_ar (rho 0.9/-0.6, one normal per channel per step,
        // interleaved, stationary start — the s-block RNG order is frozen),
        // pushed through an INSTANTANEOUS nonlinear mixture: FOUR residual
        // GELU layers x <- x + a*gelu(W_j x) with fixed W's. Each layer is a
        // residual contraction for a*Lip(gelu)*||W|| < 1 (Lip ~1.13,
        // ||W||~1.7 => invertible up to a ~ 0.5), so the mixture stays in
        // the identifiability theorem's instantaneous invertible class.
        // Depth is load-bearing (measured): at two layers the warp is
        // shallow enough that a rotation of the whitened cloud still
        // correlates ~0.97 with the sources — four layers compound the
        // even-component warp until the linear class genuinely sticks.
        // GELU is non-odd: the warp has a genuine even component that no
        // whitening + rotation can linearize away — the linear members'
        // wall, by construction.
        normal_distribution<float> N01(0, 1);
        float a = D.comps[0].s1;
        const float r1 = 0.9f, r2 = -0.6f;
        const float q1 = sqrtf(1.0f - r1 * r1), q2 = sqrtf(1.0f - r2 * r2);
        const float W[4][4] = { { 0.6f, 1.2f, 1.1f, -0.5f },
                                { -0.7f, 0.9f, 1.3f, 0.4f },
                                { 0.8f, -1.1f, 0.9f, 0.7f },
                                { -0.5f, 1.0f, 1.2f, -0.6f } };
        auto gelu = [](float u) {
            return 0.5f * u * (1.0f + tanhf(0.7978845608f * (u + 0.044715f * u * u * u)));
        };
        Mat X("X", 4, n);
        float s1 = 0, s2 = 0;
        for (int i = 0; i < n; i++) {
            float e1 = N01(global_rand_gen), e2 = N01(global_rand_gen);
            if (i == 0) { s1 = e1; s2 = e2; }
            else        { s1 = r1 * s1 + q1 * e1; s2 = r2 * s2 + q2 * e2; }
            float x1 = s1, x2 = s2;
            for (int l = 0; l < 4; l++) {
                float g1 = gelu(W[l][0] * x1 + W[l][1] * x2);
                float g2 = gelu(W[l][2] * x1 + W[l][3] * x2);
                x1 += a * g1; x2 += a * g2;
            }
            X.elem(0, i) = x1;                   // pair-joint rows: [0,2) = x, [2,4) = s
            X.elem(1, i) = x2;
            X.elem(2, i) = s1;
            X.elem(3, i) = s2;
        }
        return X;
    }
    if (!D.comps.empty() && D.comps[0].kind == "signals") {
        // TIME-SERIES draw (mixed_signals): the n columns are ONE trajectory,
        // i = 0..n-1. Sources: s1 = sin (arcsine marginal), s2 = sawtooth
        // (uniform marginal), periods ~39.6 / ~46.8 SAMPLES with an
        // IRRATIONAL, well-separated ratio — the per-sample phase pair walks
        // a dense line on the torus, so for n in the hundreds the source
        // CLOUD is genuinely independent (ergodicity; a rational ratio draws
        // a closed Lissajous curve and independence-seeking flows rightly
        // refuse — measured 2026-07). Filling quality is a joint Diophantine
        // property, not a cycle count: these periods measure MI ≈ 0.04 nats
        // at n = 1024, better than periods a third their length, while a
        // near-1 period RATIO is far worse (strands run parallel). Randomness
        // = two fresh phases per DRAW (2 uniforms, regardless of n); the
        // waveforms themselves are deterministic.
        uniform_real_distribution<float> U01(0, 1);
        float m = D.comps[0].s1;
        float ph1 = U01(global_rand_gen), ph2 = U01(global_rand_gen);
        const float P1 = 28.0f * sqrtf(2.0f), P2 = 27.0f * sqrtf(3.0f);
        Mat X("X", 4, n);
        for (int i = 0; i < n; i++) {
            float s1 = sinf(2.0f * 3.14159265f * ((float)i / P1 + ph1));
            float fr = (float)i / P2 + ph2; fr -= floorf(fr);
            float s2 = 2.0f * fr - 1.0f;
            X.elem(0, i) = s1 + m * s2;              // pair-joint rows: [0,2) = x, [2,4) = s
            X.elem(1, i) = m * s1 + s2;
            X.elem(2, i) = s1;
            X.elem(3, i) = s2;
        }
        return X;
    }
    Mat X("X", D.dim, n);
    normal_distribution<float> N01(0, 1);
    uniform_real_distribution<float> U01(0, 1);
    vector<float> cw; float tot = 0;
    for (auto& c : D.comps) { tot += c.w; cw.push_back(tot); }
    for (int i = 0; i < n; i++) {
        float r = U01(global_rand_gen) * tot;
        size_t k = 0; while (k + 1 < cw.size() && r > cw[k]) k++;
        const auto& c = D.comps[k];
        float x = 0, y = 0;
        if (c.kind == "gaussian") {
            for (int d = 0; d < D.dim; d++)
                X.elem(d, i) = c.mean[d] + c.s1 * N01(global_rand_gen);
            continue;
        } else if (c.kind == "uniform") {
            for (int d = 0; d < D.dim; d++)
                X.elem(d, i) = c.mean[d] + c.s1 * (2 * U01(global_rand_gen) - 1);
            continue;
        } else if (c.kind == "mixed") {              // pair-joint rows: [0,2) = x (observed), [2,4) = s (true sources)
            float m = c.s1;
            float s1 = 3.0f * U01(global_rand_gen) - 1.5f;                       // uniform source
            float s2 = (U01(global_rand_gen) < 0.5f ? -1.0f : 1.0f) + 0.3f * N01(global_rand_gen);  // bimodal source
            X.elem(0, i) = s1 + m * s2;
            X.elem(1, i) = m * s1 + s2;
            X.elem(2, i) = s1;
            X.elem(3, i) = s2;
            continue;
        } else if (c.kind == "moons") {
            float noise = c.s1;
            bool upper = U01(global_rand_gen) < 0.5f;
            float th = 3.1415926f * U01(global_rand_gen);
            if (upper) { x = cosf(th); y = sinf(th) - 0.25f; }
            else       { x = 1 - cosf(th); y = 0.25f - sinf(th); }
            x += noise * N01(global_rand_gen); y += noise * N01(global_rand_gen);
            x = 2 * x - 1; y = 2 * y;
        } else if (c.kind == "ring") {
            float th = 2 * 3.1415926f * U01(global_rand_gen);
            x = c.s1 * cosf(th) + c.s2 * N01(global_rand_gen);
            y = c.s1 * sinf(th) + c.s2 * N01(global_rand_gen);
        } else if (c.kind == "torus") {
            // uniform in the two ANGLES (ring's convention — a data toy, not
            // the area-uniform law: the inner rim is denser), isotropic noise.
            // RNG order per sample: theta, phi, then three normals — frozen.
            float th = 2 * 3.1415926f * U01(global_rand_gen);
            float ph = 2 * 3.1415926f * U01(global_rand_gen);
            float nz = c.mean.empty() ? 0.0f : c.mean[0];
            X.elem(0, i) = (c.s1 + c.s2 * cosf(ph)) * cosf(th) + nz * N01(global_rand_gen);
            X.elem(1, i) = (c.s1 + c.s2 * cosf(ph)) * sinf(th) + nz * N01(global_rand_gen);
            X.elem(2, i) = c.s2 * sinf(ph) + nz * N01(global_rand_gen);
            continue;
        } else if (c.kind == "spiral") {
            float u = U01(global_rand_gen);
            float th = c.s1 * 2 * 3.1415926f * sqrtf(u);
            float r0 = 0.15f + 1.8f * sqrtf(u);
            x = r0 * cosf(th) + c.s2 * N01(global_rand_gen);
            y = r0 * sinf(th) + c.s2 * N01(global_rand_gen);
        } else if (c.kind == "linjoint") {           // rows: 0 = y, 1 = x
            float xc = N01(global_rand_gen);
            X.elem(0, i) = c.s1 * xc + c.s2 * N01(global_rand_gen);
            X.elem(1, i) = xc;
            continue;
        } else if (c.kind == "sinejoint") {
            float xc = N01(global_rand_gen);
            X.elem(0, i) = sinf(c.s1 * xc) + c.s2 * N01(global_rand_gen);
            X.elem(1, i) = xc;
            continue;
        } else if (c.kind == "unnorm") {
            err("internal error: unnormalized distributions admit no exact sampler (the capability check should have rejected this earlier)");
        } else err("unknown distribution component: " + c.kind);
        X.elem(0, i) = x; X.elem(1, i) = y;
    }
    return X;
}

static Mat rows_of(const Mat& X, int lo, int hi) {   // slice rows [lo, hi)
    if (lo < 0) return X;
    Mat R("blk", hi - lo, (int)X.num_col());
    for (int r = lo; r < hi; r++)
        for (size_t i = 0; i < X.num_col(); i++) R.elem(r - lo, i) = X.elem(r, i);
    return R;
}

// conditional sampler of a joint: draw y ~ p(y | x = z_i) column-by-column.
// Only constructed joints have this capability (spec 10.8: decoupled kernel
// evaluation needs it; paired evaluation never does).
static Mat sample_cond(const Dist& D, const Mat& Z) {
    if (!D.has_cond_sampler)
        err(D.canon + " has no conditional sampler — a data-only joint supports only paired evaluation "
            "(| the block bound with it); decoupled evaluation (| a foreign point/measure) needs a "
            "constructed joint (spec 10.8).");
    int n = (int)Z.num_col();
    normal_distribution<float> N01(0, 1);
    Mat Y("Yc", D.dy, n);
    const auto& c = D.comps.at(0);
    for (int i = 0; i < n; i++) {
        float z = Z.elem(0, i);
        if (c.kind == "linjoint")       Y.elem(0, i) = c.s1 * z + c.s2 * N01(global_rand_gen);
        else if (c.kind == "sinejoint") Y.elem(0, i) = sinf(c.s1 * z) + c.s2 * N01(global_rand_gen);
        else err("internal error: unknown joint kind " + c.kind);
    }
    return Y;
}

static Mat score_dist(const Dist& D, const Mat& X) {
    if (!D.has_score)
        err("" + D.canon + " has no analytic score — the nw estimator requires the score \u2207log p of its target.");
    if (D.logp) {   // unnormalized target: ∇log p̃ = ∇L,配分函数在求导中消失
        int n = (int)X.num_col();
        const float h = 1e-3f;
        Mat S("S", 2, n);
        for (int i = 0; i < n; i++) {
            float a = X.elem(0, i), b = X.elem(1, i);
            S.elem(0, i) = (D.logp(a + h, b) - D.logp(a - h, b)) / (2 * h);
            S.elem(1, i) = (D.logp(a, b + h) - D.logp(a, b - h)) / (2 * h);
        }
        return S;
    }
    int d = D.dim, n = (int)X.num_col();
    size_t K = D.comps.size();
    Mat S("S", d, n);
    for (int i = 0; i < n; i++) {
        vector<double> lw(K);
        double mx = -1e30;
        for (size_t k = 0; k < K; k++) {
            const auto& c = D.comps[k];
            double q = 0;
            for (int dd = 0; dd < d; dd++) { double z = X.elem(dd, i) - c.mean[dd]; q += z * z; }
            lw[k] = log((double)c.w) - q / (2.0 * c.s1 * c.s1) - d * log((double)c.s1);
            mx = max(mx, lw[k]);
        }
        double Z = 0; for (size_t k = 0; k < K; k++) { lw[k] = exp(lw[k] - mx); Z += lw[k]; }
        for (int dd = 0; dd < d; dd++) {
            double acc = 0;
            for (size_t k = 0; k < K; k++)
                acc += (lw[k] / Z) * (D.comps[k].mean[dd] - X.elem(dd, i)) / (D.comps[k].s1 * D.comps[k].s1);
            S.elem(dd, i) = (float)acc;
        }
    }
    return S;
}

// ─────────────────────────────── execution site (line attribution) ──
// A Liu program is straight-line (bounded `for` is the only planned
// iteration, spec 10.1), so every runtime event is statically attributable
// to a source site: the statement's line plus the iteration stack (empty
// until `for` lands — the protocol carries it from day one so v0.4 needs
// no format change). Streamed loss, plot rows, and errors all carry it.
static FILE* dump_file();
static int cur_line = 0;
static vector<int> iter_stack;
static string iter_json() {
    string s = "[";
    for (size_t i = 0; i < iter_stack.size(); i++) s += (i ? "," : "") + to_string(iter_stack[i]);
    return s + "]";
}
static string site_text() {          // "line 9" / "line 9 (k=2.3)" / ""
    if (cur_line <= 0) return "";
    string s = "line " + to_string(cur_line);
    if (!iter_stack.empty()) {
        s += " (k=";
        for (size_t i = 0; i < iter_stack.size(); i++) s += (i ? "." : "") + to_string(iter_stack[i]);
        s += ")";
    }
    return s;
}
static string json_escape(const string& m) {
    string o;
    for (char c : m) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, 8, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}

// ─────────────────────────────────────────────────────── training ───
static shared_ptr<TrainedNet> build_net(const NetSpec& spec, int batch, bool timedim = true) {
    auto tn = make_shared<TrainedNet>();
    const auto& d = spec.dims;
    if (d.size() < 2) err("mlp needs at least input -> output dimensions");
    tn->d_in = d.front(); tn->d_out = d.back();
    int in = d[0] + (timedim ? 1 : 0);       // velocity nets get a time row; a one-step generator (into=) does not
    vector<Layer<float>*> fwd;
    for (size_t i = 1; i + 1 < d.size(); i++) {
        tn->owned.push_back(make_unique<ReluLayer<float>>(d[i], in, batch));
        fwd.push_back(tn->owned.back().get());
        in = d[i];
    }
    tn->owned.push_back(make_unique<LinearLayer<float>>(d.back(), in, batch));
    fwd.push_back(tn->owned.back().get());
    for (auto it = fwd.rbegin(); it != fwd.rend(); ++it) tn->net.push_back(*it);
    return tn;
}

static void set_lr(TrainedNet& tn, float lr) {
    for (auto* l : tn.net) {
        if (auto* ll = dynamic_cast<LinearLayer<float>*>(l)) { ll->adamWstate().alpha = lr; ll->adambstate().alpha = lr; }
        if (auto* rl = dynamic_cast<ReluLayer<float>*>(l))   { rl->adamWstate().alpha = lr; rl->adambstate().alpha = lr; }
    }
}

using Sampler = function<Mat(int)>;

static Sampler dataset_sampler(const Mat& X0) {
    Mat X = X0;
    return [X](int b) {
        uniform_int_distribution<int> pick(0, (int)X.num_col() - 1);
        Mat out("b", X.num_row(), b);
        for (int i = 0; i < b; i++) { int j = pick(global_rand_gen);
            for (size_t r = 0; r < X.num_row(); r++) out.elem(r, i) = X.elem(r, j); }
        return out;
    };
}

// the ONE regression that computes E[dx_t/dt | x_t] for a declared path.
// flow matching and DSM are both special cases of this loop.
// With base (spec 10.3.2): the target becomes dx_t/dt - base(x_t, t), whose
// L2 minimizer is exactly the guidance direction v_path - base. The base
// enters as data (forward only) — frozen by construction, no gradient needed.
static shared_ptr<TrainedNet> train_field(const RVal& rv, const NetSpec& spec,
                                          int steps, int batch, float lr,
                                          const shared_ptr<FieldV>& base = nullptr) {
    int d = rv.terms.at(0).dim();
    for (auto& tm : rv.terms) if (tm.dim() != d) err("the random variables in this path formula have inconsistent dimensions");
    int cond_total = 0;
    for (auto& tm : rv.terms) cond_total += tm.cdim();
    if (spec.dims.front() != d + cond_total || spec.dims.back() != d)
        err("expected network dims " + to_string(d + cond_total) + " -> ... -> " + to_string(d) +
            (cond_total ? " (y-dim + condition rows in, y-dim out; spec 10.8)" : ""));
    auto tn = build_net(spec, batch);
    set_lr(*tn, lr);
    printf("[field] regressing %s: %zu-term formula%s, %d steps, batch %d, lr %g\n",
           base ? "residual E[dx_t/dt | x_t] - base(x_t, t)" : "E[dx_t/dt | x_t]",
           rv.terms.size(), cond_total ? " (conditional)" : "", steps, batch, lr);
    const float h = 1e-3f;
    vector<Sampler> samplers;
    for (auto& tm : rv.terms)
        samplers.push_back(tm.dist ? Sampler([D = tm.dist](int b) { return sample_dist(*D, b); })
                                   : dataset_sampler(tm.data->X));
    for (int i = 0; i < steps; i++) {
        auto trow = Mat::rand(1, batch) * 0.96f + 0.02f;    // clamp away from endpoints
        Mat Xt = Mat::ones(d, batch) * 0.0f, dXt = Mat::ones(d, batch) * 0.0f;
        vector<Mat> conds;                                   // condition rows, in term order (spec 10.8)
        map<long, Mat> draws;                                // ONE draw per SOURCE per step: terms sharing
                                                             // a src (couple's peer blocks, spec 10.3) see
                                                             // the same joint draw — the pairing survives
        for (size_t k = 0; k < rv.terms.size(); k++) {
            if (!draws.count(rv.terms[k].src)) draws.emplace(rv.terms[k].src, samplers[k](batch));
            const Mat& Xi = draws.at(rv.terms[k].src);
            Mat Yi = rows_of(Xi, rv.terms[k].blo, rv.terms[k].bhi);
            Mat crow("c", 1, batch), drow("dc", 1, batch);   // shared by x_t and dx_t (identity!)
            for (int j = 0; j < batch; j++) {
                float tj = trow.elem(0, j);
                crow.elem(0, j) = rv.terms[k].coeff(tj);
                drow.elem(0, j) = (rv.terms[k].coeff(tj + h) - rv.terms[k].coeff(tj - h)) / (2 * h);
            }
            Xt  = Xt  + hadmd(Yi, Mat::ones(d, 1) * crow);
            dXt = dXt + hadmd(Yi, Mat::ones(d, 1) * drow);
            if (rv.terms[k].cdim() > 0) conds.push_back(rows_of(Xi, rv.terms[k].clo, rv.terms[k].chi));
        }
        if (base) dXt = dXt - base->vel_rows(Xt, trow);
        vector<MatrixView<float>> stackv; stackv.push_back(Xt);
        for (auto& c : conds) stackv.push_back(c);
        stackv.push_back(trow);
        auto inp = vstack<float>(stackv);
        LossLayer<float> LL(batch, dXt);
        tn->net.push_front(&LL);
        float loss = item(forward(tn->net, inp));
        backprop(tn->net, inp);
        tn->net.pop_front();
        if (i % max(1, steps / 8) == 0 || i == steps - 1)
            printf("  step %6d   loss %.5f\n", i, loss);
        if (i % max(1, steps / 1000) == 0 || i == steps - 1)
            if (FILE* fp = dump_file()) {
                fprintf(fp, "{\"type\":\"loss\",\"line\":%d,\"iter\":%s,\"step\":%d,\"value\":%.6g}\n",
                        cur_line, iter_json().c_str(), i, (double)loss);
                fflush(fp);
            }
    }
    return tn;
}

static shared_ptr<TrainedNet> train_dsm(Sampler sdata, const NetSpec& spec, float T,
                                        int steps, int batch, float lr) {
    auto tn = build_net(spec, batch);
    set_lr(*tn, lr);
    int d = spec.dims.front();
    printf("[reverse] denoising score matching: T=%.1f, %d steps, batch %d, lr %g\n", T, steps, batch, lr);
    for (int i = 0; i < steps; i++) {
        auto X0  = sdata(batch);
        auto t   = Mat::rand(1, batch) * (T - 0.02f) + 0.02f;
        auto a   = exp(t * -1.0f);
        auto sig = exp(log(Mat::ones(1, batch) - hadmd(a, a)) * 0.5f);
        auto EPS = Mat::randn(d, batch);
        auto Xt  = hadmd(X0, Mat::ones(d, 1) * a) + hadmd(EPS, Mat::ones(d, 1) * sig);
        auto inp = vstack<float>({Xt, t});
        LossLayer<float> LL(batch, EPS);
        tn->net.push_front(&LL);
        float loss = item(forward(tn->net, inp));
        backprop(tn->net, inp);
        tn->net.pop_front();
        if (i % max(1, steps / 8) == 0 || i == steps - 1)
            printf("  step %6d   loss %.5f\n", i, loss);
        if (i % max(1, steps / 1000) == 0 || i == steps - 1)
            if (FILE* fp = dump_file()) {
                fprintf(fp, "{\"type\":\"loss\",\"line\":%d,\"iter\":%s,\"step\":%d,\"value\":%.6g}\n",
                        cur_line, iter_json().c_str(), i, (double)loss);
                fflush(fp);
            }
    }
    return tn;
}

// ─────────────────────────────────────────────────────────── SVGD ───
// Exact k-th smallest (0-based rank) of the clamped squared distances,
// straight off the matrix buffer: non-negative floats order like their
// uint32 bit patterns, so two 16-bit counting passes select the value —
// no n²-element copy, no nth_element partition shuffles (that pair was
// ~half the kernel engine's runtime at n=2048, bench/B4). Selection is
// order-free, so the VALUE is identical to the old sort-based median:
// bit-identity holds by construction. Layout-safe: the multiset of
// elements does not depend on the transpose flag.
static float select_sq_clamped(const Mat& Dm, size_t k) {
    const float* p = Dm.data();
    const size_t N = Dm.num_row() * Dm.num_col();
    if (N == 0) return 1.0f;
    auto key = [](float x) -> uint32_t {
        x = x > 0.0f ? x : 0.0f;                    // max(0, ·), as before
        uint32_t u; memcpy(&u, &x, 4); return u;    // ≥ 0 ⇒ monotone bits
    };
    vector<size_t> hist(65536, 0);
    for (size_t i = 0; i < N; i++) hist[key(p[i]) >> 16]++;
    size_t rank = k; uint32_t hi = 0;
    while (rank >= hist[hi]) { rank -= hist[hi]; hi++; }
    fill(hist.begin(), hist.end(), 0);
    for (size_t i = 0; i < N; i++) {
        uint32_t u = key(p[i]);
        if ((u >> 16) == hi) hist[u & 0xffffu]++;
    }
    uint32_t lo = 0;
    while (rank >= hist[lo]) { rank -= hist[lo]; lo++; }
    uint32_t u = (hi << 16) | lo;
    float x; memcpy(&x, &u, 4); return x;
}

// Bandwidth of the RBF kernel exp(-||.||^2 / h), from the current ensemble.
// Two conventions, matched to the two field estimators:
//   unnormalized (SVGD):  h = med{||xi-xj||^2} / log(n+1)  — Liu & Wang
//     2016, paper and official code: n exp(-med^2/h) ~ 1, so each point's
//     neighbourhood contributes ~1/n of the kernel mass, balancing
//     attraction against repulsion;
//   normalized (NW):      h = 2 med{||xi-xj||^2}  — the Gretton median
//     heuristic sigma = med(d) for exp(-d^2/(2 sigma^2)), no log-n
//     scaling: the NW ratio needs smoothing on the scale of the data,
//     not of the interaction strength.
// Samples-only targets pool the ensemble with the target samples first
// (pool_cols below): both sides of the score difference must use the SAME
// kernel and bandwidth, or the two KDE biases stop cancelling at p = q.
static float svgd_bandwidth(const Mat& X, bool normalize = false) {
    int n = (int)X.num_col();
    auto G  = X.T() * X;
    auto sq = sum(square(X), 0);
    auto Dm = Mat::ones(n, 1) * sq + sq.T() * Mat::ones(1, n) - G * 2.0f;
    float med = select_sq_clamped(Dm, ((size_t)n * n) / 2);
    return max(1e-4f, normalize ? 2.0f * med : med / logf((float)n + 1.0f));
}

// Columns of A followed by columns of B (the pooled ensemble+target set
// used for the shared bandwidth of samples-only targets).
static Mat pool_cols(const Mat& A, const Mat& B) {
    int d = (int)A.num_row(), n = (int)A.num_col(), m = (int)B.num_col();
    Mat P("pool", d, n + m);
    for (int r = 0; r < d; r++) {
        for (int j = 0; j < n; j++) P.elem(r, j)     = A.elem(r, j);
        for (int j = 0; j < m; j++) P.elem(r, n + j) = B.elem(r, j);
    }
    return P;
}

// NW field of the frozen ensemble E (samples of the moving q), evaluated at
// arbitrary query points Y (d x q). ONE form covers every case — a smoothed
// score difference:
//     Phi(y) = smooth(score p)(y) / Z_p(y)  -  smooth(score q)(y) / Z_q(y)
// The q-term is always the KDE score of the ensemble:
//     smooth(score q)(y) = sum_j grad_y k(e_j, y).
// The p-term depends on what the target provides:
//   score  (tgt_d): sum_j k(e_j,y) score(e_j) — the true score, smoothed over
//     the ensemble points (n score evaluations per step, none per query:
//     recorded replay stays self-contained); Z_p = Z_q (same kernel sums);
//   samples (tgt_P): sum_i grad_y k(p_i, y) — the KDE score of the target
//     samples; same kernel, same h on both sides, so the two smoothing
//     biases cancel exactly where p_hat = q_hat (the field has the right
//     zeros even at finite h).
// normalize=false: Z = point count. With a score this is exactly the SVGD
//     update (steepest descent in the kernelized Stein geometry).
// normalize=true:  Z = kernel mass at the query point — the genuine
//     Nadaraya-Watson interpolation, a consistent estimator of the W2
//     velocity grad log(p/q) (Liu, Yu, Simons, Yi & Beaumont 2024, eq. 5:
//     the SVGD update is exactly the numerator; the denominator varies with
//     the query point, so it cannot be absorbed into the learning rate).
static Mat svgd_field_at(const Mat& E, float h, const Dist* tgt_d, const Mat* tgt_P,
                         const Mat& Y, bool normalize = false) {
    int d = (int)E.num_row(), n = (int)E.num_col();
    auto G   = E.T() * Y;                                       // n x q
    auto sqe = sum(square(E), 0);                               // 1 x n
    auto sqy = sum(square(Y), 0);                               // 1 x q
    auto Dm  = sqe.T() * Mat::ones(1, (int)Y.num_col()) + Mat::ones(n, 1) * sqy - G * 2.0f;
    auto Kk  = exp(Dm * (-1.0f / h));                           // n x q
    int q = (int)Y.num_col();
    if (tgt_d) {
        auto S   = score_dist(*tgt_d, E);                       // d x n
        auto sumK = sum(Kk, 0);                                 // 1 x q
        auto attr = S * Kk;                                     // d x q
        auto rep  = (hadmd(Y, Mat::ones(d, 1) * sumK) - E * Kk) * (2.0f / h);
        if (!normalize) return (attr + rep) * (1.0f / (float)n);
        Mat inv("inv", 1, q);
        for (int i = 0; i < q; i++) inv.elem(0, i) = 1.0f / max(sumK.elem(0, i), 1e-12f);
        return hadmd(attr + rep, Mat::ones(d, 1) * inv);
    }
    // samples-only target: p-term is the KDE score of tgt_P, with its own mass
    const Mat& P = *tgt_P;
    int m = (int)P.num_col();
    auto sumK = sum(Kk, 0);                                     // 1 x q
    auto rep  = (hadmd(Y, Mat::ones(d, 1) * sumK) - E * Kk) * (2.0f / h);
    auto Gp   = P.T() * Y;                                      // m x q
    auto sqp  = sum(square(P), 0);                              // 1 x m
    auto Dp   = sqp.T() * Mat::ones(1, q) + Mat::ones(m, 1) * sqy - Gp * 2.0f;
    auto Kp   = exp(Dp * (-1.0f / h));                          // m x q
    auto sumKp = sum(Kp, 0);                                    // 1 x q
    auto attr  = (P * Kp - hadmd(Y, Mat::ones(d, 1) * sumKp)) * (2.0f / h);
    if (!normalize) return attr * (1.0f / (float)m) + rep * (1.0f / (float)n);
    Mat invp("invp", 1, q), invq("invq", 1, q);
    for (int i = 0; i < q; i++) {
        invp.elem(0, i) = 1.0f / max(sumKp.elem(0, i), 1e-12f);
        invq.elem(0, i) = 1.0f / max(sumK.elem(0, i),  1e-12f);
    }
    return hadmd(attr, Mat::ones(d, 1) * invp) + hadmd(rep, Mat::ones(d, 1) * invq);
}

// (Weighted) KDE score of the sample set S at query points Y — the building
// block of the N-observation conditional descent. The KL decomposition
//   Σ_i KL[q‖p(·|Y_i)] − (N−1)·KL[q‖prior]
// is exact for true scores, but assembling it FIELD-BY-FIELD from KDE
// scores fails in three measured ways (Gaussian location toy, exact
// posterior known): (1) summing N+1 complete nw fields double-counts
// repulsions whose bandwidths never cancel — net anti-diffusion plus an
// anti-prior drift, cloud runs to ±12 on a ±1 posterior; (2) with one
// repulsion but per-term JOINT bandwidths (systematically wider than the
// parameter-space one — they add the y-spread to the median), the
// far-field slopes miss and the tails still blow up; (3) with one
// repulsion and one shared bandwidth, the fixpoint multiplies N
// INDIVIDUALLY SMOOTHED likelihood factors, which demands a q̂ narrower
// than the kernel floor h/2 — the cloud collapses (sd 0.09 on an 0.24
// posterior). The assembly that survives puts the additivity in the
// log-WEIGHTS (see MapV::apply): one weighted attraction (wrow), one
// repulsion, one bandwidth — both sides smoothed once, neutral tails,
// fixpoint = the weighted posterior itself.
// normalize mirrors svgd_field_at: true = genuine ∇log p̂_S, false = the
// smoothed (mass-divided) form of the Stein geometry.
static Mat kde_score_at(const Mat& S, float h, const Mat& Y, bool normalize,
                        const Mat* wrow = nullptr, const Mat* wmat = nullptr) {
    int d = (int)S.num_row(), m = (int)S.num_col(), q = (int)Y.num_col();
    auto G   = S.T() * Y;                                       // m x q
    auto sqs = sum(square(S), 0);                               // 1 x m
    auto sqy = sum(square(Y), 0);                               // 1 x q
    auto Dm  = sqs.T() * Mat::ones(1, q) + Mat::ones(m, 1) * sqy - G * 2.0f;
    Mat Kk   = exp(Dm * (-1.0f / h));                           // m x q
    if (wrow) Kk = hadmd(Kk, wrow->T() * Mat::ones(1, q));      // scale library row j by u_j
    if (wmat) Kk = hadmd(Kk, *wmat);                            // per-QUERY weights (m x q) — each
                                                                // query column carries its own library
                                                                // weighting (pin channel: its own
                                                                // pinned block)
    auto sumK = sum(Kk, 0);                                     // 1 x q
    auto num  = (S * Kk - hadmd(Y, Mat::ones(d, 1) * sumK)) * (2.0f / h);
    if (!normalize) {
        if (wmat) {                                             // per-query weight mass
            auto wq = sum(*wmat, 0);                            // 1 x q
            Mat winv("winv", 1, q);
            for (int i = 0; i < q; i++) winv.elem(0, i) = 1.0f / max(wq.elem(0, i), 1e-12f);
            return hadmd(num, Mat::ones(d, 1) * winv);
        }
        float wsum = (float)m;
        if (wrow) { wsum = 0; for (int j = 0; j < m; j++) wsum += wrow->elem(0, j); }
        return num * (1.0f / max(wsum, 1e-12f));
    }
    Mat inv("inv", 1, q);
    for (int i = 0; i < q; i++) inv.elem(0, i) = 1.0f / max(sumK.elem(0, i), 1e-12f);
    return hadmd(num, Mat::ones(d, 1) * inv);
}

// Likelihood-role bandwidths (spec 10.10) — read off the library's
// RESOLUTION, not its spread. Dw = pairwise squared distances of the
// library's parameter rows (M x M); Yl = the library's observed rows.
//   hl — the k-NN-quantile regression window in parameter space
//        (k = max(8, M/100)): keeps ~k draws per window to marginalize
//        the simulator's noise;
//   hy — the y-kernel scale: the Gamma / nearest-neighbor estimate of the
//        simulator's own conditional noise E Var[y|w] (median squared
//        y-distance between W-nearest draws), shrunk by the KDE-optimal
//        kq^(-2/(4+dY)) rate.
// Pooled-median here flattens the likelihood to an isotropic prior shrink;
// a plain distance-quantile rule collapses below the noise floor on dense
// low-dim libraries (both measured — see the (q|Obs) precompute). Shared
// by the two conditional-descent channels: (q|Obs) uses both scales,
// from=(y|x) uses hy for its pinned-row kernel.
struct LikBW { float hl, hy; int kq; };
static LikBW likelihood_bandwidths(const Mat& Dw, const Mat& Yl) {
    int M = (int)Yl.num_col(), dY = (int)Yl.num_row();
    int kq = max(8, M / 100);
    vector<float> buf((size_t)M), knn(M);
    for (int j = 0; j < M; j++) {
        for (int l = 0; l < M; l++) buf[l] = Dw.elem(l, j);
        nth_element(buf.begin(), buf.begin() + min(kq, M - 1), buf.end());
        knn[j] = buf[min(kq, M - 1)];
    }
    nth_element(knn.begin(), knn.begin() + M / 2, knn.end());
    float hl = max(1e-4f, knn[M / 2]);
    vector<float> gam(M);
    for (int j = 0; j < M; j++) {
        int best = 0; float bd = 1e30f;
        for (int l = 0; l < M; l++)
            if (l != j && Dw.elem(l, j) < bd) { bd = Dw.elem(l, j); best = l; }
        float s = 0;
        for (int r = 0; r < dY; r++) { float df = Yl.elem(r, j) - Yl.elem(r, best); s += df * df; }
        gam[j] = s;
    }
    nth_element(gam.begin(), gam.begin() + M / 2, gam.end());
    float hy = max(1e-4f, gam[M / 2] * powf((float)kq, -2.0f / (4.0f + (float)dY)));
    return {hl, hy, kq};
}

// THE conditional-likelihood estimator (spec 10.10) — the single NW-ratio
// core both conditioning syntaxes lower to (2026-07 unification; the raw
// kernel-weight and pooled-joint-field assemblies are deleted). Given the
// library's observed rows Yl (dY x M), parameter rows Wl (dW x M), and K
// condition columns C, the regression-smoothed single-condition likelihood
// at library point l is
//     L̂(C_i, w_l) = Σ_j k_hy(C_i, y_j) k_hl(w_l, w_j) / Σ_j k_hl(w_l, w_j)
// — the NW regression of the y-kernel onto the parameter rows. A library
// point's likelihood is assessed through its parameter-space NEIGHBORS,
// never through its own single draw: the exchangeability trap of raw
// kernel weights (one draw explaining every observation) cannot arise,
// and N=1 is the same estimator with one factor. Bandwidths are the
// likelihood-role rules above. Returns Lh = U·Kw (K x M) and the window
// masses Zw (1 x M): L̂_il = Lh_il / Zw_l. Callers aggregate by their
// syntax's semantics — (q|Obs) sums logs over its N observation rows
// (all observations condition every particle), (y|x) reads row c as
// particle c's own weights (one condition per particle).
static void likelihood_regression(const Mat& Yl, const Mat& Wl, const Mat& C,
                                  Mat& Lh, Mat& Zw) {
    int M = (int)Yl.num_col(), K = (int)C.num_col();
    auto Gw  = Wl.T() * Wl;                                 // M x M parameter distances
    auto sqw = sum(square(Wl), 0);
    Mat Dw   = sqw.T() * Mat::ones(1, M) + Mat::ones(M, 1) * sqw - Gw * 2.0f;
    LikBW bw = likelihood_bandwidths(Dw, Yl);
    Mat Kw   = exp(Dw * (-1.0f / bw.hl));                   // regression window
    auto Go  = C.T() * Yl;                                  // K x M
    auto sqo = sum(square(C), 0);
    auto sql = sum(square(Yl), 0);
    Mat Dy   = sqo.T() * Mat::ones(1, M) + Mat::ones(K, 1) * sql - Go * 2.0f;
    Mat U    = exp(Dy * (-1.0f / bw.hy));                   // K x M y-kernels k_hy(C_i, y_l)
    Lh = U * Kw;                                            // K x M: Σ_l k_hy(C_i,y_l) k_hl(w_j,w_l)
    Zw = sum(Kw, 0);                                        // 1 x M: Σ_l k_hl(w_j,w_l)
}

// W2 descent (spec ledger, 2026-07): the velocity of F[q] = 1/2 W2^2(q, p)
// is the displacement toward the OT map — estimated as the barycentric
// projection of the entropic plan between the ensemble and the target
// samples: v(e_i) = sum_j pi_ij p_j / sum_j pi_ij − e_i. Cost is normalized
// by its mean so eps is scale-free; uniform marginals; deterministic.
static Mat w2_displacement(const Mat& E, const Mat& P, float eps) {
    int d = (int)E.num_row(), n = (int)E.num_col(), m = (int)P.num_col();
    vector<vector<float>> C(n, vector<float>(m));
    double csum = 0;
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) {
        float sq = 0;
        for (int r = 0; r < d; r++) { float df = E.elem(r, i) - P.elem(r, j); sq += df * df; }
        C[i][j] = sq; csum += sq;
    }
    float mc = max(1e-12f, (float)(csum / ((double)n * m)));
    float ep = max(1e-4f, eps);
    vector<vector<float>> K(n, vector<float>(m));
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) K[i][j] = expf(-C[i][j] / (mc * ep));
    vector<float> u(n, 1.0f), v(m, 1.0f);
    for (int it = 0; it < 100; it++) {
        for (int i = 0; i < n; i++) { float t = 0; for (int j = 0; j < m; j++) t += K[i][j] * v[j]; u[i] = 1.0f / max(t, 1e-30f); }
        for (int j = 0; j < m; j++) { float t = 0; for (int i = 0; i < n; i++) t += K[i][j] * u[i]; v[j] = 1.0f / max(t, 1e-30f); }
    }
    Mat D("disp", d, n);
    for (int i = 0; i < n; i++) {
        float mass = 0; vector<float> bary(d, 0.0f);
        for (int j = 0; j < m; j++) {
            float w = u[i] * K[i][j] * v[j]; mass += w;
            for (int r = 0; r < d; r++) bary[r] += w * P.elem(r, j);
        }
        for (int r = 0; r < d; r++) D.elem(r, i) = bary[r] / max(mass, 1e-30f) - E.elem(r, i);
    }
    return D;
}

// family=rotation (ICA line): the exponential of a skew-symmetric matrix, by
// scaling-and-squaring plus a short Taylor series. d is the ensemble dimension
// (tiny); deterministic; skew input ⇒ orthogonal output up to float precision.
static Mat so_exp(Mat A) {
    int d = (int)A.num_row();
    Mat I("eye", d, d);
    for (int r = 0; r < d; r++) for (int c = 0; c < d; c++) I.elem(r, c) = r == c ? 1.0f : 0.0f;
    float m = 0;
    for (int r = 0; r < d; r++) for (int c = 0; c < d; c++) m = max(m, fabsf(A.elem(r, c)));
    int sq = 0; float bound = m * (float)d;                 // ‖A‖ ≤ d·max|a_ij|
    while (bound > 0.25f && sq < 40) { A = A * 0.5f; bound *= 0.5f; sq++; }
    Mat R = I, T = I;
    for (int k = 1; k <= 12; k++) { T = T * A * (1.0f / (float)k); R = R + T; }
    for (int i = 0; i < sq; i++) R = R * R;
    return R;
}

// rotation-constrained regression (estimator=regress(rotation), declared
// paths): restrict the hypothesis class of the conditional-expectation
// regression to SKEW-LINEAR fields {x -> Omega(t) x}. The least squares then
// has a CLOSED FORM per time slice: with C(t) = E[x_t x_tT] = U diag(lam) UT
// and M(t) = E[dx/dt x_tT], the minimizer of E|v - Omega x|^2 over skew Omega
// is, in the eigenbasis, Omega'_ij = (M'_ij - M'_ji)/(lam_i + lam_j). No
// network, no SGD — 'training' is one moment computation per knot; the flow
// map is a rotation by construction and its inverse is free (transpose).
// block=L (process level): the hypothesis class tightens to the block-
// Kronecker rotations {x -> (Omega (x) I_L) x} — ONE channel rotation shared
// across the L lags of window()ed data. The objective keeps the exact same
// quadratic shape with the moments LAG-POOLED down to channels x channels:
//   Mt_cc' = sum_l M[(cL+l),(c'L+l)],  Ct likewise (only same-lag products
// enter — the within-block temporal autocorrelation drops out of Ct), and
// the same Lyapunov equation / eigenbasis formula solves a CxC problem.
// Unlike the descent-side rotation(block=L) (which assumes the lag-pooled
// channel covariance ~ I), the lambda weighting here is exact — least
// squares does the step the descent projection approximates.
static vector<Mat> fit_rot_field(const RVal& rv, int knots, int batch, int block = 1) {
    int d = rv.terms.at(0).dim();
    const float h = 1e-3f;
    int q = 0;
    for (auto& tm : rv.terms) q += tm.cdim();
    vector<Sampler> samplers;
    for (auto& tm : rv.terms)
        samplers.push_back(tm.dist ? Sampler([D = tm.dist](int b) { return sample_dist(*D, b); })
                                   : dataset_sampler(tm.data->X));
    printf("[field] rotation-constrained regression: closed form, %d time slices, batch %d%s%s\n",
           knots, batch, block > 1 ? (" (block=" + to_string(block) + ": channel rotations shared across lags)").c_str() : "",
           q > 0 ? (" (conditional: " + to_string(q) + " partialled covariate rows)").c_str() : "");
    vector<Mat> out;
    for (int k = 0; k < knots; k++) {
        float t = 0.02f + 0.96f * (float)k / (float)(knots - 1);
        Mat Xt = Mat::ones(d, batch) * 0.0f, dXt = Mat::ones(d, batch) * 0.0f;
        map<long, Mat> draws;                       // one draw per src: couple pairing survives
        for (size_t i = 0; i < rv.terms.size(); i++) {
            if (!draws.count(rv.terms[i].src)) draws.emplace(rv.terms[i].src, samplers[i](batch));
            const Mat& Xi = draws.at(rv.terms[i].src);
            Mat Yi = rows_of(Xi, rv.terms[i].blo, rv.terms[i].bhi);
            float c  = rv.terms[i].coeff(t);
            float dc = (rv.terms[i].coeff(t + h) - rv.terms[i].coeff(t - h)) / (2 * h);
            Xt  = Xt  + Yi * c;
            dXt = dXt + Yi * dc;
        }
        // Conditional path (2026-07, SICA line): the law-gate indices join the
        // regression as free covariates — fit the UNCONSTRAINED linear field
        // v = A·[x; c] per slice (normal equations in doubles) and keep the
        // SKEW PART of the x-block as the knot. Two measured decisions live
        // here (mixed_ar wall, floor 0.765): (1) the proper skew-constrained
        // LS on context-residualized moments buries the signal — it stalls at
        // 0.79 while skew(A_x) reaches 1.000; the identifiable turn is the
        // antisymmetric component OF the conditional coefficient, not the
        // best skew approximation of the residual velocity. (2) a field with
        // free x-terms and context terms APPLIED as a map drifts into the
        // per-channel FILTER ambiguity of the self-sufficiency objective
        // (lag-0 MCC decays monotonically) — so only the skew x-part is kept:
        // the conditions are CONSUMED at fit time, the field stays
        // skew-linear and self-contained, and the flow can only turn the
        // cloud. Declare the path from the decoupled product INTO the current
        // windows (t=1 = current): the demixing rotation is the REVERSE of
        // the entangling transport, and this orientation makes the forward
        // flow descend (measured; the other orientation walks away).
        if (q > 0) {
            Mat Cc("cc", q, batch);
            int rr = 0;
            for (size_t i = 0; i < rv.terms.size(); i++) {
                if (rv.terms[i].cdim() == 0) continue;
                const Mat& Xi = draws.at(rv.terms[i].src);
                for (int r = rv.terms[i].clo; r < rv.terms[i].chi; r++, rr++)
                    for (int c = 0; c < batch; c++) Cc.elem(rr, c) = Xi.elem(r, c);
            }
            int m = d + q;
            Mat U("u", m, batch);
            for (int r = 0; r < d; r++) for (int c = 0; c < batch; c++) U.elem(r, c) = Xt.elem(r, c);
            for (int r = 0; r < q; r++) for (int c = 0; c < batch; c++) U.elem(d + r, c) = Cc.elem(r, c);
            auto G = U * U.T();                     // m x m Gram; Gauss-Jordan inverse in doubles
            vector<vector<double>> Gd(m, vector<double>(2 * m, 0));
            for (int r = 0; r < m; r++) {
                for (int c = 0; c < m; c++) Gd[r][c] = G.elem(r, c);
                Gd[r][r] += 1e-6 * (double)batch;
                Gd[r][m + r] = 1;
            }
            for (int p = 0; p < m; p++) {
                int piv = p;
                for (int r = p + 1; r < m; r++) if (fabs(Gd[r][p]) > fabs(Gd[piv][p])) piv = r;
                swap(Gd[p], Gd[piv]);
                double dv = Gd[p][p];
                for (int c = 0; c < 2 * m; c++) Gd[p][c] /= dv;
                for (int r = 0; r < m; r++) if (r != p) {
                    double f2 = Gd[r][p];
                    for (int c = 0; c < 2 * m; c++) Gd[r][c] -= f2 * Gd[p][c];
                }
            }
            Mat Gi("gi", m, m);
            for (int r = 0; r < m; r++) for (int c = 0; c < m; c++) Gi.elem(r, c) = (float)Gd[r][m + c];
            Mat A = (dXt * U.T()) * Gi;             // d x m unconstrained coefficient
            Mat Ok("Ok", d, d);
            for (int a = 0; a < d; a++)
                for (int b = 0; b < d; b++)
                    Ok.elem(a, b) = (A.elem(a, b) - A.elem(b, a)) * 0.5f;
            out.push_back(Ok);
            continue;
        }
        Mat Mfull = dXt * Xt.T() * (1.0f / (float)batch);
        Mat Cfull = Xt * Xt.T() * (1.0f / (float)batch);
        int L = block, dc2 = d / max(1, block);      // solve at channel level
        Mat Mm("Mp", dc2, dc2), Cm("Cp", dc2, dc2);
        if (L <= 1) { Mm = Mfull; Cm = Cfull; }
        else
            for (int a = 0; a < dc2; a++)
                for (int b = 0; b < dc2; b++) {
                    float sm = 0, sc = 0;
                    for (int l = 0; l < L; l++) {
                        sm += Mfull.elem(a * L + l, b * L + l);
                        sc += Cfull.elem(a * L + l, b * L + l);
                    }
                    Mm.elem(a, b) = sm; Cm.elem(a, b) = sc;
                }
        int dd = dc2;                                 // eigen-solve dimension (channels when block>1)
        // cyclic Jacobi on Cm (same scheme as whiten; dd is tiny)
        vector<vector<double>> C(dd, vector<double>(dd)), V(dd, vector<double>(dd, 0));
        for (int r = 0; r < dd; r++) { V[r][r] = 1; for (int s = 0; s < dd; s++) C[r][s] = Cm.elem(r, s); }
        for (int sweep = 0; sweep < 64; sweep++) {
            double off = 0;
            for (int p = 0; p < dd; p++) for (int q = p + 1; q < dd; q++) off += C[p][q] * C[p][q];
            if (off < 1e-18) break;
            for (int p = 0; p < dd; p++) for (int q = p + 1; q < dd; q++) {
                if (fabs(C[p][q]) < 1e-15) continue;
                double th = 0.5 * atan2(2 * C[p][q], C[q][q] - C[p][p]);
                double cs = cos(th), sn = sin(th);
                for (int r = 0; r < dd; r++) { double a = C[p][r], b = C[q][r]; C[p][r] = cs * a - sn * b; C[q][r] = sn * a + cs * b; }
                for (int r = 0; r < dd; r++) {
                    double a = C[r][p], b = C[r][q];
                    C[r][p] = cs * a - sn * b; C[r][q] = sn * a + cs * b;
                    a = V[r][p]; b = V[r][q];
                    V[r][p] = cs * a - sn * b; V[r][q] = sn * a + cs * b;
                }
            }
        }
        // M' = V^T M V; Omega'_ij = (M'_ij - M'_ji)/(lam_i + lam_j); Omega = V Omega' V^T
        vector<vector<double>> Mp(dd, vector<double>(dd, 0));
        for (int a = 0; a < dd; a++) for (int b = 0; b < dd; b++) {
            double s = 0;
            for (int r = 0; r < dd; r++) for (int c2 = 0; c2 < dd; c2++) s += V[r][a] * Mm.elem(r, c2) * V[c2][b];
            Mp[a][b] = s;
        }
        Mat Oc("Oc", dd, dd);
        for (int a = 0; a < dd; a++) for (int b = 0; b < dd; b++) Oc.elem(a, b) = 0.0f;
        for (int a = 0; a < dd; a++) for (int b = a + 1; b < dd; b++) {
            double lam = C[a][a] + C[b][b];
            double w = (Mp[a][b] - Mp[b][a]) / max(lam, 1e-9);
            for (int r = 0; r < dd; r++) for (int c2 = 0; c2 < dd; c2++)
                Oc.elem(r, c2) += (float)(w * (V[r][a] * V[c2][b] - V[r][b] * V[c2][a]));
        }
        if (L <= 1) { out.push_back(Oc); continue; }
        Mat Om("Om", d, d);                           // expand Omega (x) I_L to the full knot
        for (int a = 0; a < d; a++) for (int b = 0; b < d; b++) Om.elem(a, b) = 0.0f;
        for (int a = 0; a < dd; a++)
            for (int b = 0; b < dd; b++)
                for (int l = 0; l < L; l++) Om.elem(a * L + l, b * L + l) = Oc.elem(a, b);
        out.push_back(Om);
    }
    return out;
}

// rotation(net=mlp): the NEURAL member of the conditional-rotation family
// (2026-07, SICA line). The conditional field is trained by the ordinary
// conditional flow-matching regression (train_field — contexts enter the
// NET, not as linear features), and the applied map is then its so(d)
// projection: per time slice, the net's average element-Jacobian by
// central differences (contexts held fixed at their drawn values), skew
// part kept. For a linear net the difference Jacobian IS the coefficient,
// so this reduces exactly to sica_cond's measured winner skew(A_x); the
// nonlinear net buys context-dependence in the FIT while the map class
// still cannot filter (the identifiability lesson survives verbatim).
static vector<Mat> project_rot_net(const RVal& rv, TrainedNet& tn, int knots, int batch) {
    int d = rv.terms.at(0).dim();
    vector<Sampler> samplers;
    for (auto& tm : rv.terms)
        samplers.push_back(tm.dist ? Sampler([D = tm.dist](int b) { return sample_dist(*D, b); })
                                   : dataset_sampler(tm.data->X));
    const float eps = 1e-2f;
    vector<Mat> out;
    for (int k = 0; k < knots; k++) {
        float t = 0.02f + 0.96f * (float)k / (float)(knots - 1);
        Mat Xt = Mat::ones(d, batch) * 0.0f;
        vector<Mat> conds;
        map<long, Mat> draws;
        for (size_t i = 0; i < rv.terms.size(); i++) {
            if (!draws.count(rv.terms[i].src)) draws.emplace(rv.terms[i].src, samplers[i](batch));
            const Mat& Xi = draws.at(rv.terms[i].src);
            Mat Yi = rows_of(Xi, rv.terms[i].blo, rv.terms[i].bhi);
            Xt = Xt + Yi * rv.terms[i].coeff(t);
            if (rv.terms[i].cdim() > 0) conds.push_back(rows_of(Xi, rv.terms[i].clo, rv.terms[i].chi));
        }
        Mat trow = Mat::ones(1, batch) * t;
        Mat J("J", d, d);
        for (int jd = 0; jd < d; jd++) {
            Mat Xp = Xt, Xm = Xt;
            for (int c = 0; c < batch; c++) {
                Xp.elem(jd, c) = Xp.elem(jd, c) + eps;
                Xm.elem(jd, c) = Xm.elem(jd, c) - eps;
            }
            vector<MatrixView<float>> sp; sp.push_back(Xp);
            for (auto& cc : conds) sp.push_back(cc);
            sp.push_back(trow);
            Mat Vp = tn.eval(vstack<float>(sp));
            vector<MatrixView<float>> sm; sm.push_back(Xm);
            for (auto& cc : conds) sm.push_back(cc);
            sm.push_back(trow);
            Mat Vm = tn.eval(vstack<float>(sm));
            for (int r = 0; r < d; r++) {
                float s = 0;
                for (int c = 0; c < batch; c++) s += Vp.elem(r, c) - Vm.elem(r, c);
                J.elem(r, jd) = s / (2.0f * eps * (float)batch);
            }
        }
        Mat Ok("Ok", d, d);
        for (int a = 0; a < d; a++)
            for (int b = 0; b < d; b++)
                Ok.elem(a, b) = (J.elem(a, b) - J.elem(b, a)) * 0.5f;
        out.push_back(Ok);
    }
    return out;
}

static void svgd_step(Mat& X, const Dist* tgt_d, const Mat* tgt_P, float lr,
                      bool normalize = false) {
    float h = svgd_bandwidth(tgt_P ? pool_cols(X, *tgt_P) : X, normalize);
    X = X + svgd_field_at(X, h, tgt_d, tgt_P, X, normalize) * lr;
}

// project=instant (2026-07, SICA line): Rao-Blackwell the per-column
// conditional field V onto a function of the ELEMENTS E alone — closed-form
// ridge regression on the degree-3 polynomial features of E, returning the
// fitted values w(E). Every flow step then applies ONE shared instantaneous
// map z + lr*w(z), the class the identifiability theorem certifies
// (Jacobian one-nonzero-per-row => permutation + channel-wise scalar maps).
// Deterministic, no RNG. Degree 3 is the fixed v1 class: compositions of
// small cubic-field steps approximate smooth instantaneous demixers; a
// LINEAR feature set could never accumulate a nonlinear demixer.
static Mat instant_project(const Mat& E, const Mat& V) {
    int d = (int)E.num_row(), n = (int)E.num_col();
    if (d > 4) err("project=instant: the polynomial feature basis is kept small — element blocks of "
                   "more than 4 rows are not supported in this prototype");
    // BOUNDED coordinates: u = 2*tanh(z/2) is ~identity near the origin and
    // saturates at ±2, so the cubic monomials below stay globally Lipschitz —
    // raw polynomials eject tail particles (measured: climbs to 0.80, then
    // the |z|^3 extrapolation blows the tails and MCC crashes to 0.48), and
    // a bounded field also keeps the small-step map safely invertible.
    vector<vector<int>> monos;                     // exponent multi-sets, degree 0..3
    monos.push_back({});
    for (int i = 0; i < d; i++) monos.push_back({i});
    for (int i = 0; i < d; i++) for (int j = i; j < d; j++) monos.push_back({i, j});
    for (int i = 0; i < d; i++) for (int j = i; j < d; j++) for (int k = j; k < d; k++)
        monos.push_back({i, j, k});
    int m = (int)monos.size();
    Mat U("u", d, n);
    for (int r = 0; r < d; r++)
        for (int c = 0; c < n; c++) U.elem(r, c) = 2.0f * tanhf(E.elem(r, c) * 0.5f);
    Mat Phi("phi", m, n);
    for (int c = 0; c < n; c++)
        for (int r = 0; r < m; r++) {
            float v = 1.0f;
            for (int idx : monos[r]) v *= U.elem(idx, c);
            Phi.elem(r, c) = v;
        }
    auto G = Phi * Phi.T();
    vector<vector<double>> Gd(m, vector<double>(2 * m, 0));
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < m; c++) Gd[r][c] = G.elem(r, c);
        Gd[r][r] += 1e-4 * (double)n;
        Gd[r][m + r] = 1;
    }
    for (int p = 0; p < m; p++) {
        int piv = p;
        for (int r = p + 1; r < m; r++) if (fabs(Gd[r][p]) > fabs(Gd[piv][p])) piv = r;
        swap(Gd[p], Gd[piv]);
        double dv = Gd[p][p];
        for (int c = 0; c < 2 * m; c++) Gd[p][c] /= dv;
        for (int r = 0; r < m; r++) if (r != p) {
            double f2 = Gd[r][p];
            for (int c = 0; c < 2 * m; c++) Gd[r][c] -= f2 * Gd[p][c];
        }
    }
    Mat Gi("gi", m, m);
    for (int r = 0; r < m; r++) for (int c = 0; c < m; c++) Gi.elem(r, c) = (float)Gd[r][m + c];
    Mat A = (V * Phi.T()) * Gi;                    // d x m coefficients
    return A * Phi;                                // fitted instantaneous field at E
}

// Amortized flow (spec 10.6): flow(v, into=net) — the drifting training loop.
// Instead of moving particles, move parameters: each optimizer step samples a
// latent batch Z from the path's from= measure, evaluates Y = net(Z), estimates
// the SAME nw field on the generated batch, and regresses net(Z) onto the
// frozen target Y + eps*Phi(Y) (one Euler step of the descent, applied to the
// generator's own pushforward). The pushforward q_theta evolves across
// optimizer steps; equilibrium (Phi = 0) makes the target the output itself
// and training stalls exactly when the pushforward matches the target law.
// Two-timescale ideal: iNGD Algorithm 1 (Liu, Wang & Wang 2025); inner/outer
// loops merged (one gradient step per drift): Drifting Models (Deng, Li, Li,
// Du & He 2026). The loss printed IS eps^2 * mean||Phi||^2 — the zero-flow
// diagnostic: it converges to 0 iff the two laws merge (finite-sample floor).
static shared_ptr<TrainedNet> train_amortized(FieldV& v, const NetSpec& spec, int steps,
                                              int batch, float eps, float trainlr) {
    Sampler sz = v.wg_from_d ? Sampler([D = v.wg_from_d](int b) { return sample_dist(*D, b); })
                             : dataset_sampler(v.wg_from_x->X);
    const Dist* td = v.div->target_d.get();
    const Mat*  tp = v.div->target_x ? &v.div->target_x->X : nullptr;
    auto tn = build_net(spec, batch, /*timedim=*/false);
    set_lr(*tn, trainlr);
    printf("[flow/amortized] drifting a one-step generator: %d optimizer steps, batch %d, drift eps %g, train lr %g\n",
           steps, batch, eps, trainlr);
    for (int i = 0; i < steps; i++) {
        Mat Z = sz(batch);
        Mat Y = tn->eval(Z);
        Mat target = Y;                                                     // frozen (stop-gradient by construction)
        if (v.div->name == "w2") target = Y + w2_displacement(Y, *tp, v.div->w2_eps) * eps;
        else {
            float h = svgd_bandwidth(tp ? pool_cols(Y, *tp) : Y, v.nw_norm);
            target = Y + svgd_field_at(Y, h, td, tp, Y, v.nw_norm) * eps;
        }
        LossLayer<float> LL(batch, target);
        tn->net.push_front(&LL);
        float loss = item(forward(tn->net, Z));
        backprop(tn->net, Z);
        tn->net.pop_front();
        if (i % max(1, steps / 8) == 0 || i == steps - 1)
            printf("  step %6d   drift loss %.6f\n", i, loss);
        if (i % max(1, steps / 1000) == 0 || i == steps - 1)
            if (FILE* fp = dump_file()) {
                fprintf(fp, "{\"type\":\"loss\",\"line\":%d,\"iter\":%s,\"step\":%d,\"value\":%.6g}\n",
                        cur_line, iter_json().c_str(), i, (double)loss);
                fflush(fp);
            }
    }
    return tn;
}

// Amortized INSTANTANEOUS demixer (Level 1 of the SICA line): the drifting
// loop for a conditional descent whose ensemble carries window provenance.
// The outer per-iteration loop of sica_instant/sica_nl dissolves into the
// optimizer: each step (1) pushes the SOURCE TRAJECTORY through the residual
// generator g(x) = x + net(x), per column — instantaneous by construction,
// the identifiability theorem's class; (2) rebuilds the window/lagsplit
// ensemble and a FRESH per-channel permuted product from g's own output
// (the structured re-simulation — plain into= only re-draws latents from a
// fixed measure); (3) warm-tracks BOTH dsm score nets (the target moves too:
// it is the permuted product of the current windows); (4) regresses net(x_t)
// onto the frozen drift target net(x_t) + eps*V_t, where V_t is the free-row
// (element) block of the conditional score difference at column t's own
// pinned context. Contexts enter the objective through the field; the net
// never sees them. The printed loss is eps^2*mean||V||^2 — the zero-flow
// diagnostic. Consumes RNG every step (product picks, dsm noise, batches).
static shared_ptr<TrainedNet> train_amortized_demixer(FieldV& v, const NetSpec& spec, int steps,
                                                      int batch, float eps, float trainlr) {
    const Mat& T0 = v.wg_from_x->wsrc->X;
    int L = v.wg_from_x->wlen;
    int d = (int)T0.num_row(), n = (int)T0.num_col(), m = n - L + 1, dL = d * L;
    int N = (int)v.div->target_x->X.num_col();
    float sg = v.dsm_sigma;
    auto tn = build_net(spec, batch, false);
    set_lr(*tn, trainlr);
    auto sp = build_net(*v.dsm_spec, v.dsm_batch, false);   // score net of the permuted product
    auto sq = build_net(*v.dsm_spec, v.dsm_batch, false);   // score net of the current windows
    set_lr(*sp, v.dsm_trainlr); set_lr(*sq, v.dsm_trainlr);
    auto dsm_step = [dL, sg, &v](TrainedNet& net_, const Mat& D) {
        int b = min(v.dsm_batch, (int)D.num_col());
        uniform_int_distribution<int> pick(0, (int)D.num_col() - 1);
        Mat xb("xb", dL, b);
        for (int c = 0; c < b; c++) {
            int j = pick(global_rand_gen);
            for (int r = 0; r < dL; r++) xb.elem(r, c) = D.elem(r, j);
        }
        Mat eb = Mat::randn(dL, b);
        Mat xt = xb + eb * sg;
        LossLayer<float> LL(b, eb);
        net_.net.push_front(&LL);
        float loss = item(forward(net_.net, xt));
        backprop(net_.net, xt);
        net_.net.pop_front();
        return loss;
    };
    // lagsplit table straight off a trajectory: element rows [0,d) = lag 0,
    // context row d+(k-1)d+c = channel c at lag k (same layout as the verbs)
    auto build_lag = [d, L, m](const Mat& Z) {
        Mat Y("lag", d * L, m);
        for (int c = 0; c < d; c++)
            for (int i = 0; i < m; i++) {
                Y.elem(c, i) = Z.elem(c, i);
                for (int k = 1; k < L; k++) Y.elem(d + (k - 1) * d + c, i) = Z.elem(c, i + k);
            }
        return Y;
    };
    // per-channel permuted product (decouple block=L), fresh picks per step
    auto build_prod = [d, L, m, N, dL](const Mat& Z) {
        uniform_int_distribution<int> pick(0, m - 1);
        Mat P("prod", dL, N);
        for (int c = 0; c < d; c++)
            for (int i = 0; i < N; i++) {
                int j = pick(global_rand_gen);
                P.elem(c, i) = Z.elem(c, j);
                for (int k = 1; k < L; k++) P.elem(d + (k - 1) * d + c, i) = Z.elem(c, j + k);
            }
        return P;
    };
    printf("[flow/amortized-instant] drifting an instantaneous demixer: %d optimizer steps, batch %d, "
           "drift eps %g, train lr %g (dsm: sigma %g, pretrain %d, warm %d/step; product %d/step)\n",
           steps, batch, eps, trainlr, (double)sg, v.dsm_pre, v.dsm_warm, N);
    // start at the identity: regress the residual to zero first, so the
    // drift loop opens inside whatever basin the caller's coordinates are
    // already in (measured at Level 0: the basin is what a coarse stage buys)
    {
        uniform_int_distribution<int> pick(0, n - 1);
        Mat z0 = Mat::zeros(d, batch);
        for (int i = 0; i < 200; i++) {
            Mat xb("xb", d, batch);
            for (int c = 0; c < batch; c++) {
                int j = pick(global_rand_gen);
                for (int r = 0; r < d; r++) xb.elem(r, c) = T0.elem(r, j);
            }
            LossLayer<float> LL(batch, z0);
            tn->net.push_front(&LL);
            forward(tn->net, xb);
            backprop(tn->net, xb);
            tn->net.pop_front();
        }
    }
    // TWO-TIMESCALE loop, and measurably not optional. The merged
    // one-gradient-step-per-drift form (Drifting Models, the plain into=
    // loop) fails HERE in every schedule tried (0.896 -> 0.53-0.84, raw or
    // projected targets, persistent or reinitialized score nets): near the
    // demixer the drift target's signal shrinks while its noise — and the
    // criterion's flat valley of per-channel reparametrizations — does not,
    // and Adam's per-coordinate normalization turns whatever small
    // persistent component remains into a constant-speed walk along that
    // valley. The particle loop never sees this failure because a particle
    // takes the FULL lr*V step of a freshly estimated field. So the net
    // must do the same: each OUTER step estimates a fresh field (new dsm
    // pair, full pretrain — the Level-0 cadence, bias resampled every
    // step), Rao-Blackwells it onto the elements (the Level-0 projection),
    // freezes the drift target g_old(x) + eps*W(g_old(x)), and the INNER
    // loop regresses the net to convergence on that frozen target — the
    // net's version of taking the whole step (iNGD Algorithm 1's
    // two-timescale ideal, which the merged form only approximates).
    const int inner = 200;
    for (int i = 0; i < steps; i++) {
        Mat Zt = T0 + tn->eval(T0);
        Mat Y  = build_lag(Zt);
        Mat P  = build_prod(Zt);
        sp = build_net(*v.dsm_spec, v.dsm_batch, false);
        sq = build_net(*v.dsm_spec, v.dsm_batch, false);
        set_lr(*sp, v.dsm_trainlr); set_lr(*sq, v.dsm_trainlr);
        for (int s = 0; s < v.dsm_pre; s++) { dsm_step(*sp, P); dsm_step(*sq, Y); }
        Mat Ep = sp->eval(Y), Eq = sq->eval(Y);
        // free rows of the joint score difference = the conditional score
        // difference at each column's own pinned context (WGPathV::free_lo)
        Mat Ee("Ee", d, m), Vr("Vr", d, m);
        for (int r = 0; r < d; r++)
            for (int j = 0; j < m; j++) {
                Ee.elem(r, j) = Y.elem(r, j);
                Vr.elem(r, j) = (Eq.elem(r, j) - Ep.elem(r, j)) / sg;
            }
        Mat Vp = instant_project(Ee, Vr);
        // frozen full-trajectory target for this outer step
        Mat TGT("tgt", d, m);
        for (int r = 0; r < d; r++)
            for (int j = 0; j < m; j++)
                TGT.elem(r, j) = (Zt.elem(r, j) - T0.elem(r, j)) + eps * Vp.elem(r, j);
        float loss = 0;
        int b = min(batch, m);
        uniform_int_distribution<int> pick(0, m - 1);
        for (int s = 0; s < inner; s++) {
            Mat xb("xb", d, b), tb("tb", d, b);
            for (int c = 0; c < b; c++) {
                int j = pick(global_rand_gen);
                for (int r = 0; r < d; r++) {
                    xb.elem(r, c) = T0.elem(r, j);
                    tb.elem(r, c) = TGT.elem(r, j);
                }
            }
            LossLayer<float> LL(b, tb);
            tn->net.push_front(&LL);
            loss = item(forward(tn->net, xb));
            backprop(tn->net, xb);
            tn->net.pop_front();
        }
        if (i % max(1, steps / 8) == 0 || i == steps - 1)
            printf("  step %6d   drift loss %.6f\n", i, loss);
        if (FILE* fp = dump_file()) {
            fprintf(fp, "{\"type\":\"loss\",\"line\":%d,\"iter\":%s,\"step\":%d,\"value\":%.6g}\n",
                    cur_line, iter_json().c_str(), i, (double)loss);
            fflush(fp);
        }
    }
    return tn;
}

Mat MapV::apply(Mat X, vector<Mat>* traj) {
    if (traj) traj->push_back(X);
    if (f->kind == FieldV::AMORT) {
        // one-step generator: the whole Euler chain lives in the weights (1-NFE)
        if ((int)X.num_row() != f->dim)
            err(string(f->residual ? "this instantaneous demixer" : "this one-step generator") +
                " was trained on " + to_string(f->dim) + "-dimensional " +
                (f->residual ? "channels" : "latents") + ", got " + to_string((int)X.num_row()));
        Mat Y = f->residual ? X + f->net->eval(X) : f->net->eval(X);
        if (traj) traj->push_back(Y);
        return Y;
    }
    if (f->kind == FieldV::WG) {
        const Dist* td = f->div->target_d.get();
        const Mat*  tp = f->div->target_x ? &f->div->target_x->X : nullptr;
        if (inverse) {
            // replay inverse: per step solve y = x + lr*Phi_t(x) by fixed point
            if (!hist || hist->empty())
                err("inversion needs a recorded history: apply this map forward once first (a forward pass under record=true stores the per-step ensembles)");
            for (int i = steps - 1; i >= 0; i--) {
                Mat Y = X;                       // solve for x: x = y - lr*Phi_t(x)
                for (int it = 0; it < 8; it++)
                    X = Y - svgd_field_at((*hist)[i], (*hist_h)[i], td, tp, X, f->nw_norm) * lr;
                if (traj && (i % max(1, steps / 12) == 0)) traj->push_back(X);
            }
            return X;
        }
        if (record && hist && !hist->empty()) {
            // history exists: replay the frozen pointwise maps (consistent map!)
            for (int i = 0; i < steps; i++) {
                X = X + svgd_field_at((*hist)[i], (*hist_h)[i], td, tp, X, f->nw_norm) * lr;
                if (traj && ((i + 1) % max(1, steps / 12) == 0)) traj->push_back(X);
            }
            return X;
        }
        // family=rotation: project the descent velocity onto the Lie algebra
        // so(d) and move along the orthogonal group instead of freely —
        //   M = (1/n) Σ_i v(z_i) z_iᵀ,   Ω = (M − Mᵀ)/2,   z ← exp(lr·Ω) z.
        // Ω is the so(d) component of the velocity's first moment; on a white
        // ensemble (E[zzᵀ] = I) this is the natural-gradient direction of
        // Amari's ICA, so the rotation flow IS continuous-time natural-gradient
        // ICA with the score replaced by the nw estimate. The search space
        // collapses from all of W2 to the d(d−1)/2-dimensional rotation orbit;
        // every step is exactly volume- and covariance-preserving.
        if (f->rotation) {
            int d = (int)X.num_row(), n = (int)X.num_col();
            const int L = f->rot_block, C = d / max(1, f->rot_block);
            if (d % max(1, f->rot_block) != 0)
                err("family=rotation(block=" + to_string(L) + "): the ensemble dimension " + to_string(d) +
                    " is not a multiple of the lag count");
            if (f->wg_from_d) {
                // reachability note: rotations preserve the covariance, so a
                // non-white ensemble can never reach a whitened target's orbit.
                // Dataset starts are checked at the descent() declaration site
                // (deterministic there); a Dist start has no covariance until
                // sampled, so it is checked here on the concrete ensemble.
                vector<float> mu(d, 0);
                for (int r = 0; r < d; r++) { double s = 0; for (int i = 0; i < n; i++) s += X.elem(r, i); mu[r] = (float)(s / n); }
                float dev = 0;
                for (int r = 0; r < d; r++) for (int c2 = 0; c2 < d; c2++) {
                    double s = 0;
                    for (int i = 0; i < n; i++) s += (double)(X.elem(r, i) - mu[r]) * (X.elem(c2, i) - mu[c2]);
                    dev = max(dev, fabsf((float)(s / n) - (r == c2 ? 1.0f : 0.0f)));
                }
                if (dev > 0.25f)
                    printf("note: family=rotation moves on the rotation orbit of the initial ensemble, and this "
                           "ensemble is not white (max |cov − I| = %.2f); rotations preserve the covariance, so "
                           "whiten(...) first if the target is a whitened/decoupled law.\n", dev);
            }
            // bandwidth: the SHARP (Liu & Wang) choice even for the normalized
            // field — the projection averages n·d field values down to
            // d(d−1)/2 numbers, so variance is cheap here, while a wide
            // interpolation bandwidth Gaussianizes the KDE scores and the
            // skew moment of a linear field is ZERO (the rotation signal
            // lives entirely in the higher cumulants).
            for (int i = 0; i < steps; i++) {
                Mat V = f->div->name == "w2"
                            ? w2_displacement(X, *tp, f->div->w2_eps)
                            : svgd_field_at(X, svgd_bandwidth(tp ? pool_cols(X, *tp) : X, /*normalize=*/false),
                                            td, tp, X, f->nw_norm);
                if (L <= 1) {
                    Mat M = V * X.T() * (1.0f / (float)n);
                    Mat R = so_exp((M - M.T()) * (0.5f * lr));
                    X = R * X;
                } else {
                    // block=L (process ICA): project onto {ω ⊗ I_L, ω ∈ so(C)} —
                    // ONE channel rotation shared across the L lags. With the
                    // lag-pooled channel covariance ~ I (whiten the channels
                    // BEFORE window()ing), the metric projection is the skew
                    // part of the LAG-AVERAGED moment M_cc' = mean over lags
                    // and samples of v[c,lag]·z[c',lag].
                    Mat Mfull = V * X.T() * (1.0f / (float)n);
                    Mat Mc("Mc", C, C);
                    for (int a = 0; a < C; a++)
                        for (int b = 0; b < C; b++) {
                            float s = 0;
                            for (int l = 0; l < L; l++) s += Mfull.elem(a * L + l, b * L + l);
                            Mc.elem(a, b) = s / (float)L;
                        }
                    Mat Rc = so_exp((Mc - Mc.T()) * (0.5f * lr));
                    Mat Rbig("Rb", d, d);
                    for (int a = 0; a < d; a++) for (int b = 0; b < d; b++) Rbig.elem(a, b) = 0.0f;
                    for (int a = 0; a < C; a++)
                        for (int b = 0; b < C; b++)
                            for (int l = 0; l < L; l++) Rbig.elem(a * L + l, b * L + l) = Rc.elem(a, b);
                    X = Rbig * X;
                }
                if (traj && ((i + 1) % max(1, steps / 12) == 0 || i == steps - 1)) traj->push_back(X);
            }
            return X;
        }
        // N-observation conditioning (from=(q|Obs)): the observations and the
        // library never move, so the whole Bayes update is precomputed as ONE
        // weight per library point,
        //     log W_j = Σ_i log L̂_i(w_j),
        // the user's log-likelihood additivity landing at the WEIGHT level.
        // Each single-observation likelihood is the NW regression of the
        // y-kernel onto the parameter rows,
        //     L̂_i(w_j) = Σ_l k_y(Y_i, y_l) k_w(w_j, w_l) / Σ_l k_w(w_j, w_l),
        // so a library draw needs to explain each observation through its
        // NEIGHBORS in parameter space, not through its own single y — the
        // exchangeability trap of stacking (one draw explaining all N) never
        // arises. Assembling the additivity at the FIELD level instead
        // (Σ_i v_i − (N−1)·v_0) is exact for true scores but diverges or
        // collapses for KDE ones — see kde_score_at for the two measured
        // failure modes. The y-bandwidth is the SHARP (Liu & Wang) choice:
        // k_y plays the likelihood, and the wide interpolation bandwidth
        // flattens it toward the prior (same measured reason as
        // family=rotation's exception). Log-sum with max subtraction — a
        // weight scale cancels in the normalized score and in the
        // mass-divided one.
        Mat obsW("obsW", 0, 0), obsWm("obsWm", 0, 0);
        if (f->obs) {
            const Mat& Ob = f->obs->X;
            int dY = f->obs_hi - f->obs_lo, dj = (int)tp->num_row();
            Mat Oc = rows_of(Ob, f->obs_lo, f->obs_hi);         // dY x N observation columns
            Mat Yl = rows_of(*tp, 0, dY);                       // library y-block
            obsWm  = rows_of(*tp, dY, dj);                      // library parameter rows = prior samples
            int N = (int)Oc.num_col(), M = (int)Yl.num_col();
            Mat Lh("Lh", 0, 0), Zw("Zw", 0, 0);
            likelihood_regression(Yl, obsWm, Oc, Lh, Zw);
            obsW = Mat("obsW", 1, M);
            float lmax = -1e30f;
            for (int j = 0; j < M; j++) {
                float lw = 0;
                for (int i2 = 0; i2 < N; i2++)
                    lw += logf(max(Lh.elem(i2, j), 1e-30f)) - logf(max(Zw.elem(0, j), 1e-30f));
                obsW.elem(0, j) = lw; lmax = max(lmax, lw);
            }
            for (int j = 0; j < M; j++) obsW.elem(0, j) = expf(obsW.elem(0, j) - lmax);
        }
        // Pin-channel likelihood weights (from=(y|x), samples target, 2026-07
        // revision): pinning IS conditioning for the true score — and for a
        // SCORE target it stays exact (score_dist evaluates ∇log p at the
        // pinned queries; no window anywhere). But the KDE attraction of a
        // samples target manufactures the slice with a kernel in the pinned
        // rows, and under the pooled JOINT bandwidth that kernel was an
        // ABC-style acceptance window calibrated to the cloud's spread, not
        // the simulator's noise (measured on sbi_svgd: ~7x too wide — 30% of
        // the posterior mass in valleys carrying 1%). The slice is therefore
        // manufactured by the ONE conditional-likelihood estimator
        // (likelihood_regression above — the same core (q|Obs) aggregates),
        // read as PER-PARTICLE weights: each ensemble column carries its own
        // pinned block, so row c of the regression is particle c's weights,
        //     v(θ_c) = ∇log Σ_l L̂(x0_c, w_l) k_h(θ_c, w_l) − ∇log q̂(θ_c),
        // and a non-degenerate pinned marginal conditions every particle on
        // its own observation. Free-row smoothing pools over free rows only.
        // The weights never change during the flow (library and pinned rows
        // are both frozen); columns are max-normalized — a per-query scale
        // cancels in both score forms.
        Mat pinW("pinW", 0, 0), pinWl("pinWl", 0, 0), pinU("pinU", 0, 0);
        if (f->free_lo >= 0 && tp) {
            int n0 = (int)X.num_col(), M = (int)tp->num_col();
            auto comp_rows = [&](const Mat& A) {                // rows outside [free_lo, free_hi)
                int da = (int)A.num_row(), na = (int)A.num_col();
                Mat R("cmp", da - (f->free_hi - f->free_lo), na);
                int rr = 0;
                for (int r = 0; r < da; r++) {
                    if (r >= f->free_lo && r < f->free_hi) continue;
                    for (int c = 0; c < na; c++) R.elem(rr, c) = A.elem(r, c);
                    rr++;
                }
                return R;
            };
            Mat Yl   = comp_rows(*tp);                          // library observed rows (dY x M)
            pinWl    = rows_of(*tp, f->free_lo, f->free_hi);    // library parameter rows (dW x M)
            Mat Xpin = comp_rows(X);                            // pinned blocks, one per particle (dY x n)
            Mat Lh("Lh", 0, 0), Zw("Zw", 0, 0);
            likelihood_regression(Yl, pinWl, Xpin, Lh, Zw);
            pinW = Mat("pinW", M, n0);                          // W_{l,c} = L̂(x0_c, w_l)
            for (int c = 0; c < n0; c++) {
                float mx = 0;
                for (int l = 0; l < M; l++) {
                    float v = Lh.elem(c, l) / max(Zw.elem(0, l), 1e-30f);
                    pinW.elem(l, c) = v; if (v > mx) mx = v;
                }
                float inv = 1.0f / max(mx, 1e-30f);
                for (int l = 0; l < M; l++) pinW.elem(l, c) = pinW.elem(l, c) * inv;
            }
            // Conditional repulsion (2026-07, family transport): a NON-
            // degenerate pinned marginal makes the ensemble a family cloud —
            // one slice per particle — and the repulsion must then be the
            // conditional score ∇_x log q̂(x | y_c), not the pooled marginal's.
            // Measured on the linear-Gaussian closed-form audit: pooled
            // repulsion under-repels within slices (sd 0.16 vs 0.351) and its
            // central bulk pushes outer slices outward (slice response 1.89
            // vs 1.096). The user's identity ∇_x log q(x|y) = ∇_x log q(x,y)
            // makes the fix a JOINT-KDE read on the free rows: weight each
            // particle's repulsion by the pin-channel kernel k_hy(y_j, y_c),
            // with hy the library's likelihood-role scale — both terms then
            // condition at the SAME resolution, so the far-field slopes still
            // cancel slice by slice. A point-mass pinned block takes the
            // exact pooled path (one slice: pooled IS conditional), which
            // also keeps every existing golden bit-identical.
            bool pins_vary = false;
            int dp = (int)Xpin.num_row();
            for (int c = 1; c < n0 && !pins_vary; c++)
                for (int r = 0; r < dp; r++)
                    if (Xpin.elem(r, c) != Xpin.elem(r, 0)) { pins_vary = true; break; }
            if (pins_vary) {
                auto Gw2  = pinWl.T() * pinWl;              // likelihood-role hy, same rule as
                auto sqw2 = sum(square(pinWl), 0);          // likelihood_regression reads off the library
                Mat Dw2   = sqw2.T() * Mat::ones(1, M) + Mat::ones(M, 1) * sqw2 - Gw2 * 2.0f;
                LikBW bw  = likelihood_bandwidths(Dw2, Yl);
                auto Gq   = Xpin.T() * Xpin;                // pin-pin distances across the ensemble
                auto sqq  = sum(square(Xpin), 0);
                Mat Dq    = sqq.T() * Mat::ones(1, n0) + Mat::ones(n0, 1) * sqq - Gq * 2.0f;
                pinU = exp(Dq * (-1.0f / bw.hy));           // n x n slice-sharing kernel
            }
        }
        // —— estimator=dsm (2026-07): warm-started neural score descent ——
        // Two epsilon-predictor nets at ONE fixed sigma: score s(x) = -net(x)/sigma
        // estimates grad log (law * N(0, sigma^2)). The p-net is trained once on
        // the frozen target joint; the q-net is PRE-trained on the initial cloud
        // and then warm-started — dsm_warm SGD steps per flow step on the moved
        // particles — so it tracks q_t without a per-step retrain. The field is
        // the plain score difference (the W2 velocity); pinned rows (free_lo)
        // condition both terms through the joint-score theorem, no likelihood
        // window and no bandwidth anywhere — sigma is the one smoothing scale,
        // shared by both nets so the bias cancels at the fixpoint. Consumes RNG
        // every step (batch picks + noise draws) — unlike nw, this descent is
        // NOT RNG-silent; it sits in the program's stream like any training.
        shared_ptr<TrainedNet> dsm_p, dsm_q;
        function<float(TrainedNet&, const Mat&)> dsm_train;
        if (f->dsm) {
            if (!tp) err("estimator=dsm needs the target's samples (reverseKL(Dataset))");
            if (record) err("record=true is not supported under estimator=dsm — the replay machinery "
                            "stores nw bandwidths; re-simulate forward instead");
            int dj = (int)X.num_row();
            if ((int)tp->num_row() != dj)
                err("dsm: the target joint has " + to_string((int)tp->num_row()) + " rows but the ensemble has " +
                    to_string(dj) + " — the score nets live on the SAME joint space");
            if (f->dsm_spec->dims.front() != dj || f->dsm_spec->dims.back() != dj)
                err("dsm: expected network dims " + to_string(dj) + " -> ... -> " + to_string(dj) +
                    " (joint point in, epsilon out)");
            float sg = f->dsm_sigma;
            // capture by VALUE: this lambda outlives the enclosing block (the
            // step loop calls it for the warm start), so reference captures
            // of block locals would dangle
            auto dsm_step = [f = f, dj, sg](TrainedNet& tn, const Mat& D) {
                int b = min(f->dsm_batch, (int)D.num_col());
                uniform_int_distribution<int> pick(0, (int)D.num_col() - 1);
                Mat xb("xb", dj, b);
                for (int c = 0; c < b; c++) {
                    int j = pick(global_rand_gen);
                    for (int r = 0; r < dj; r++) xb.elem(r, c) = D.elem(r, j);
                }
                Mat eps = Mat::randn(dj, b);
                Mat xt = xb + eps * sg;
                LossLayer<float> LL(b, eps);
                tn.net.push_front(&LL);
                float loss = item(forward(tn.net, xt));
                backprop(tn.net, xt);
                tn.net.pop_front();
                return loss;
            };
            dsm_p = build_net(*f->dsm_spec, f->dsm_batch, false);
            dsm_q = build_net(*f->dsm_spec, f->dsm_batch, false);
            set_lr(*dsm_p, f->dsm_trainlr); set_lr(*dsm_q, f->dsm_trainlr);
            printf("[dsm] score nets: sigma %g, pretrain %d steps (p on %d target draws, q on %d particles), "
                   "warm %d/flow step\n", (double)sg, f->dsm_pre, (int)tp->num_col(), (int)X.num_col(), f->dsm_warm);
            float lp = 0, lq = 0;
            for (int i = 0; i < f->dsm_pre; i++) { lp = dsm_step(*dsm_p, *tp); lq = dsm_step(*dsm_q, X); }
            printf("[dsm] pretrain done: p-loss %.4f, q-loss %.4f\n", lp, lq);
            // the step loop below owns the warm-start; stash the trainer
            dsm_train = dsm_step;
        }
        // fresh simulation; record history if asked
        if (record) { hist = make_shared<vector<Mat>>(); hist_h = make_shared<vector<float>>(); }
        for (int i = 0; i < steps; i++) {
            if (record) { hist->push_back(X); hist_h->push_back(svgd_bandwidth(tp ? pool_cols(X, *tp) : X, f->nw_norm)); }
            if (f->dsm) {
                for (int w = 0; w < f->dsm_warm; w++) dsm_train(*dsm_q, X);
                Mat Ep = dsm_p->eval(X), Eq = dsm_q->eval(X);
                Mat V = (Eq - Ep) * (1.0f / f->dsm_sigma);      // (s_p - s_q): s = -eps/sigma
                int n = (int)X.num_col();
                int rlo = f->free_lo >= 0 ? f->free_lo : 0;
                int rhi = f->free_lo >= 0 ? f->free_hi : (int)X.num_row();
                if (proj_instant) {
                    Mat Ef = rows_of(X, rlo, rhi);
                    Mat W = instant_project(Ef, rows_of(V, rlo, rhi));
                    for (int r = rlo; r < rhi; r++)
                        for (int c = 0; c < n; c++)
                            X.elem(r, c) = X.elem(r, c) + lr * W.elem(r - rlo, c);
                } else
                for (int r = rlo; r < rhi; r++)
                    for (int c = 0; c < n; c++)
                        X.elem(r, c) = X.elem(r, c) + lr * V.elem(r, c);
            }
            else if (f->div->name == "w2") X = X + w2_displacement(X, *tp, f->div->w2_eps) * lr;
            else if (f->obs) {
                // KL decomposition (2026-07, SBI line): the N-observation KL
                // splits by Bayes + log-likelihood additivity (WGPathV::obs);
                // the additivity is assembled at the log-WEIGHT level (see the
                // precompute above), so each step is a PLAIN descent onto the
                // likelihood-weighted prior KDE — one attraction, one
                // repulsion, one shared bandwidth:
                //     v = ∇log p̂_post − ∇log q̂,   p̂_post = Σ_j W_j k_w(·, w_j).
                // Both sides smoothed once by the same kernel, so the fixpoint
                // is the weighted posterior itself (no product-of-smoothed-
                // factors collapse), and the far-field slopes cancel exactly.
                float hw = svgd_bandwidth(pool_cols(X, obsWm), f->nw_norm);
                Mat V = kde_score_at(obsWm, hw, X, f->nw_norm, &obsW)   // ∇log p̂_post
                      - kde_score_at(X, hw, X, f->nw_norm);             // −∇log q̂
                X = X + V * lr;
            }
            else if (f->free_lo >= 0 && tp) {
                // conditional descent (from=(y|x)), samples target: the
                // NW-ratio assembly with the likelihood-role bandwidth —
                // frozen pinned-row weights pinW select the library's slice
                // at the simulator's own noise scale, then attraction and
                // repulsion are plain free-row KDE scores under ONE pooled
                // bandwidth (free rows only — the pinned spread must not
                // inflate it). The free rows of the joint score difference
                // ARE the conditional score difference (WGPathV::free_lo).
                Mat Xf  = rows_of(X, f->free_lo, f->free_hi);
                float h = svgd_bandwidth(pool_cols(Xf, pinWl), f->nw_norm);
                Mat V = kde_score_at(pinWl, h, Xf, f->nw_norm, nullptr, &pinW)  // ∇log Σ_l W_lc k_h(·, w_l)
                      - ((int)pinU.num_col() > 0
                             ? kde_score_at(Xf, h, Xf, f->nw_norm, nullptr, &pinU) // −∇log q̂(·|y_c)
                             : kde_score_at(Xf, h, Xf, f->nw_norm));               // −∇log q̂ (one slice)
                if (proj_instant) V = instant_project(Xf, V);
                int n = (int)X.num_col();
                for (int r = f->free_lo; r < f->free_hi; r++)
                    for (int c = 0; c < n; c++)
                        X.elem(r, c) = X.elem(r, c) + lr * V.elem(r - f->free_lo, c);
            }
            else if (f->free_lo >= 0) {
                // conditional descent (from=(y|x)), SCORE target: pinning
                // needs no window at all — score_dist evaluates the exact
                // joint score at the pinned queries, so the joint-space
                // smoothed form is already the conditional field. Only the
                // free rows move.
                float h = svgd_bandwidth(X, f->nw_norm);
                Mat V = svgd_field_at(X, h, td, tp, X, f->nw_norm);
                int n = (int)X.num_col();
                if (proj_instant) {
                    Mat Ef = rows_of(X, f->free_lo, f->free_hi);
                    Mat W = instant_project(Ef, rows_of(V, f->free_lo, f->free_hi));
                    for (int r = f->free_lo; r < f->free_hi; r++)
                        for (int c = 0; c < n; c++)
                            X.elem(r, c) = X.elem(r, c) + lr * W.elem(r - f->free_lo, c);
                } else
                for (int r = f->free_lo; r < f->free_hi; r++)
                    for (int c = 0; c < n; c++)
                        X.elem(r, c) = X.elem(r, c) + lr * V.elem(r, c);
            }
            else svgd_step(X, td, tp, lr, f->nw_norm);
            if (traj && ((i + 1) % max(1, steps / 12) == 0 || i == steps - 1)) traj->push_back(X);
        }
        return X;
    }
    if (f->kind == FieldV::ROTFM) {
        // skew-linear field: integrate on the group — each step is an exact
        // rotation (Euler would inflate norms, Omega x ⊥ x), and the inverse
        // integration is exact too (transposes, in reverse order).
        float ds = 1.0f / steps;
        for (int i = 0; i < steps; i++) {
            float s = inverse ? 1.0f - (float)(i + 1) / steps : (float)i / steps;
            int k = (int)(s * (float)(f->rot_knots.size() - 1) + 0.5f);
            k = max(0, min((int)f->rot_knots.size() - 1, k));
            Mat R = so_exp(f->rot_knots[k] * (inverse ? -ds : ds));
            X = R * X;
            if (traj && ((i + 1) % max(1, steps / 12) == 0 || i == steps - 1)) traj->push_back(X);
        }
        return X;
    }
    float ds = 1.0f / steps;
    for (int i = 0; i < steps; i++) {
        float s = inverse ? 1.0f - (float)i / steps : (float)i / steps;
        auto V = f->vel(X, s);
        X = inverse ? X - V * ds : X + V * ds;
        if (traj && ((i + 1) % max(1, steps / 12) == 0 || i == steps - 1)) traj->push_back(X);
    }
    return X;
}

// ─────────────────────────────────────────────────────── plotting ───
static void dump_json_pts(FILE* fp, const Mat& X, bool d3 = false) {
    // d3 (decided at the plot site, where the Dataset is visible): dump
    // [x,y,z] triples — the playground renders them as a drag-rotatable
    // orthographic projection. Everything else keeps the rows-0,1 view.
    d3 = d3 && X.num_row() == 3;
    fprintf(fp, "[");
    for (size_t i = 0; i < X.num_col(); i++) {
        if (d3) fprintf(fp, "%s[%.3f,%.3f,%.3f]", i ? "," : "", X.elem(0, i), X.elem(1, i), X.elem(2, i));
        else    fprintf(fp, "%s[%.3f,%.3f]", i ? "," : "", X.elem(0, i), X.elem(1, i));
    }
    fprintf(fp, "]");
}
static FILE* dump_file() {
    static FILE* fp = nullptr; static bool tried = false;
    if (!tried) { tried = true; const char* p = getenv("LIU_DUMP"); if (p) fp = fopen(p, "a"); }
    return fp;
}

struct Series { string label; const Mat* X; };

// fixed orthographic view for terminal 3-D scatters (azimuth 0.7, elevation
// 0.5 — chosen so a torus reads as a torus); the playground gets the raw
// triples and rotates interactively.
static const float VIEW_AZ = 0.7f, VIEW_EL = 0.5f;
static Mat proj3(const Mat& X) {
    float ca = cosf(VIEW_AZ), sa = sinf(VIEW_AZ), ce = cosf(VIEW_EL), se = sinf(VIEW_EL);
    int n = (int)X.num_col();
    Mat P("p3", 2, n);
    for (int i = 0; i < n; i++) {
        float x = X.elem(0, i), y = X.elem(1, i), z = X.elem(2, i);
        P.elem(0, i) = -sa * x + ca * y;
        P.elem(1, i) = -se * ca * x - se * sa * y + ce * z;
    }
    return P;
}

static void ascii_scatter2(const vector<Series>& ss, const string& title) {
    const int W = 66, H = 24;
    float xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9;
    for (auto& s : ss)
        for (size_t i = 0; i < s.X->num_col(); i++) {
            xmin = min(xmin, s.X->elem(0, i)); xmax = max(xmax, s.X->elem(0, i));
            ymin = min(ymin, s.X->elem(1, i)); ymax = max(ymax, s.X->elem(1, i));
        }
    float px = (xmax - xmin) * 0.05f + 1e-3f, py = (ymax - ymin) * 0.05f + 1e-3f;
    xmin -= px; xmax += px; ymin -= py; ymax += py;
    vector<string> grid(H, string(W, ' '));
    const char* marks = ".o#%@";
    for (size_t k = 0; k < ss.size(); k++) {
        char m = marks[k % 5];
        const Mat& X = *ss[k].X;
        for (size_t i = 0; i < X.num_col(); i++) {
            int cx = (int)((X.elem(0, i) - xmin) / (xmax - xmin) * (W - 1));
            int cy = (int)((X.elem(1, i) - ymin) / (ymax - ymin) * (H - 1));
            if (cx >= 0 && cx < W && cy >= 0 && cy < H) grid[H - 1 - cy][cx] = m;
        }
    }
    printf("┌─ plot: %s\n", title.c_str());
    for (auto& row : grid) printf("│%s│\n", row.c_str());
    printf("└");
    for (size_t k = 0; k < ss.size(); k++) printf("  [%c] %s", marks[k % 5], ss[k].label.c_str());
    printf("   x∈[%.2f,%.2f] y∈[%.2f,%.2f]\n\n", xmin, xmax, ymin, ymax);
}

// 3-D switch, decided by the CALLER at the plot site: a figure goes 3-D
// only when every series is an eligible 3-D CLOUD — exactly 3 rows that
// form one coordinate block (Dataset::blk 0 or 3). A 3-row kernel joint
// (1-D output + 2-D parameter) is a bundle, not a point cloud, and keeps
// the classic rows-0,1 view; so do 2-row and >=4-row datasets.
static void ascii_scatter(const vector<Series>& ss, const string& title, bool d3 = false) {
    for (auto& s : ss) d3 = d3 && s.X->num_row() == 3;
    if (!d3 || ss.empty()) { ascii_scatter2(ss, title); return; }
    vector<Mat> P; P.reserve(ss.size());
    for (auto& s : ss) P.push_back(proj3(*s.X));
    vector<Series> ps;
    for (size_t k = 0; k < ss.size(); k++) ps.push_back({ss[k].label, &P[k]});
    ascii_scatter2(ps, title + "  [3d: az 0.7, el 0.5]");
}

// signal view (plot_signal): columns in INDEX ORDER — each coordinate row of
// a Dataset becomes one line over i = 0..n-1. The view is a WINDOW of the
// first SIGNAL_SHOW samples (a zoom, not a decimation — decimation would
// alias waveforms); the footer states the total honestly. Terminal rendering
// downsamples like an oscilloscope: per character column, the min–max band
// of the samples it covers (a smooth signal draws a line, an i.i.d. cloud
// honestly draws a band — order is meaningless there and the plot says so).
static const int SIGNAL_SHOW = 200;
struct SignalSeries { string label; const Mat* X; int row; };
static void ascii_signal(const vector<SignalSeries>& ss, const string& title) {
    const int W = 66, H = 24;
    float ymin = 1e9, ymax = -1e9; size_t nmax = 0; int nshow = 0;
    for (auto& s : ss) {
        nmax = max(nmax, s.X->num_col());
        int ns = min((int)s.X->num_col(), SIGNAL_SHOW);
        nshow = max(nshow, ns);
        for (int i = 0; i < ns; i++) {
            float v = s.X->elem(s.row, i);
            ymin = min(ymin, v); ymax = max(ymax, v);
        }
    }
    float py = (ymax - ymin) * 0.05f + 1e-3f; ymin -= py; ymax += py;
    vector<string> grid(H, string(W, ' '));
    const char* marks = ".o#%@";
    for (size_t k = 0; k < ss.size(); k++) {
        char m = marks[k % 5];
        const Mat& X = *ss[k].X; int n = min((int)X.num_col(), SIGNAL_SHOW), r = ss[k].row;
        for (int cx = 0; cx < W; cx++) {
            int i0 = (int)((long)cx * n / W), i1 = max(i0 + 1, (int)((long)(cx + 1) * n / W));
            float vmin = 1e9, vmax = -1e9;
            for (int i = i0; i < i1 && i < n; i++) { float v = X.elem(r, i); vmin = min(vmin, v); vmax = max(vmax, v); }
            int c0 = (int)((vmin - ymin) / (ymax - ymin) * (H - 1));
            int c1 = (int)((vmax - ymin) / (ymax - ymin) * (H - 1));
            for (int cy = max(0, c0); cy <= min(H - 1, c1); cy++) grid[H - 1 - cy][cx] = m;
        }
    }
    printf("┌─ signal: %s\n", title.c_str());
    for (auto& row : grid) printf("│%s│\n", row.c_str());
    printf("└");
    for (size_t k = 0; k < ss.size(); k++) printf("  [%c] %s", marks[k % 5], ss[k].label.c_str());
    if ((int)nmax > nshow) printf("   i∈[0,%d) of %d y∈[%.2f,%.2f]\n\n", nshow, (int)nmax, ymin, ymax);
    else                   printf("   i∈[0,%d) y∈[%.2f,%.2f]\n\n", nshow, ymin, ymax);
}

// ─────────────────────────────────────────────────────── evaluator ──
struct Interp {
    map<string, Value> env;
    bool seeded = false;

    Value eval(const ExprP& e) {
        switch (e->k) {
        case Expr::Num: { Value v; v.k = Value::Num; v.num = e->num; return v; }
        case Expr::Vec: { Value v; v.k = Value::Vec; v.vec = e->vec; return v; }
        case Expr::Ident: {
            if (e->id == "t") { Value v; v.k = Value::TFun; v.tf = [](float x) { return x; }; v.tfdesc = "t"; return v; }
            // x1/x2 are NOT reserved: user bindings shadow the coordinate
            // symbols, which take effect only when the name is unbound —
            // so x0/x1 stay available for conditional-kernel programs while
            // unnormalized(...) keeps its x1/x2 notation.
            auto it = env.find(e->id);
            if (it != env.end()) return it->second;
            if (e->id == "x1") { Value v; v.k = Value::XYFun; v.xy = [](float a, float) { return a; }; v.xydesc = "x1"; return v; }
            if (e->id == "x2") { Value v; v.k = Value::XYFun; v.xy = [](float, float b) { return b; }; v.xydesc = "x2"; return v; }
            Value v; v.k = Value::Symbol; v.sym = e->id; return v;
        }
        case Expr::Dims: {
            if (e->id != "mlp") err("unknown network skeleton: " + e->id);
            Value v; v.k = Value::Net; v.net = make_shared<NetSpec>();
            v.net->dims = e->dims; return v;
        }
        case Expr::Bin:  return eval_bin(e);
        case Expr::Mix:  return eval_mix(e);
        case Expr::Call: return eval_call(e);
        case Expr::Sample: return eval_sample(e);
        case Expr::Push: return eval_push(e);
        case Expr::Transport: return eval_transport(e);
        case Expr::KernelDef: {
            // programmable Markov kernel (spec 10.10): package the body;
            // free identifiers resolve from the program environment at
            // INSTANTIATION time (program text is fixed, so this is still a
            // pure function of the source — the reproducibility contract
            // rides on statement order as everywhere else).
            auto K = make_shared<KernelV>();
            K->prog = true; K->kd_param = e->kd_param; K->kd_body = e->kd_body;
            string b;
            for (auto& [nm, ex] : e->kd_body) b += (b.empty() ? "" : ";") + nm + "=" + expr_canon(ex);
            K->canon = canon_cap("kernel(" + e->kd_param + "){" + b + "}");
            Value r; r.k = Value::Kernel; r.kernel = K; return r;
        }
        }
        err("internal error: unhandled expression node");
    }

    // —— RV lifting & arithmetic ————————————————————————————————
    static Value lift(const Value& v) {              // Dist/Dataset → anonymous RV
        Value r; r.k = Value::RV; r.rv = make_shared<RVal>();
        RTerm tm; tm.coeff = [](float) { return 1.0f; }; tm.cdesc = "1";
        tm.src = fresh_src(); tm.anon = true;
        if (v.k == Value::DistV) tm.dist = v.dist;
        else if (v.k == Value::Data) tm.data = v.data;
        else err("only a Distribution or a Dataset can be lifted to a random variable");
        r.rv->terms.push_back(move(tm));
        return r;
    }
    static bool liftable(const Value& v) { return v.k == Value::DistV || v.k == Value::Data; }

    static Value rv_add(const Value& A, const Value& B, float sgn) {
        Value r; r.k = Value::RV; r.rv = make_shared<RVal>();
        r.rv->terms = A.rv->terms;
        for (auto tm : B.rv->terms) {
            if (sgn < 0) { TF c = tm.coeff; tm.coeff = [c](float x) { return -c(x); }; tm.cdesc = "-(" + tm.cdesc + ")"; }
            bool merged = false;
            for (auto& ex : r.rv->terms) {
                if (ex.src == tm.src) {              // same named draw: coefficients add
                    if (ex.blo != tm.blo || ex.bhi != tm.bhi || ex.clo != tm.clo || ex.chi != tm.chi) {
                        if (ex.peer && tm.peer && ex.clo == tm.clo && ex.chi == tm.chi)
                            continue;                // couple: peer blocks of ONE paired draw are
                                                     // separate terms by design (spec 10.3)
                        err("two different coordinate blocks (or conditionings) of one joint draw cannot merge into "
                            "one formula term — interpolate the y-blocks and attach conditions with | (spec 10.8)");
                    }
                    TF a = ex.coeff, b = tm.coeff;
                    ex.coeff = [a, b](float x) { return a(x) + b(x); };
                    ex.cdesc = ex.cdesc + "+" + tm.cdesc;
                    merged = true; break;
                }
                if (ex.origin() == tm.origin() && ex.anon && tm.anon)
                    err(tm.srcname() + " appears twice in this path formula, but a distribution carries no draw identity — "
                        "do the two occurrences denote the same random variable, or two independent copies? "
                        "Disambiguate with rv(): bind x = rv(...) and reuse x for the same draw, or declare two rv(...) for independent copies.");
            }
            if (!merged) r.rv->terms.push_back(move(tm));
        }
        return r;
    }

    // —— given: kernel construction / evaluation (spec 10.8) ——————
    // —— draw-level evaluation (spec 10.10): programmable kernel bodies ——
    // The body of kernel(w){...} is a straight-line program defined
    // POINTWISE in the parameter and executed vectorized: names denote
    // blocks of draws (columns = draws), arithmetic is elementwise with
    // 1-row broadcast (W·x shapes), distributions auto-lift to FRESH
    // draws (same rule as t-formulas; a repeated canon is a teaching
    // error), and trained transports apply per draw via #. This is a
    // second, scoped semantic level — none of its idioms leak out
    // (the evaluator exists only under kernel_body_eval).
    struct DV { bool scalar = true; float s = 0; Mat m; DV() : m("dv", 0, 0) {} };

    DV deval(const ExprP& e, map<string, Mat>& denv, map<string, float>& senv,
             int n, set<string>& lifted) {
        auto lift = [&](const shared_ptr<Dist>& D) -> DV {
            if (!D->samplable)
                err("kernel body: " + D->canon + " admits no exact sampler — a simulator draws, it does not weigh");
            if (!lifted.insert(D->canon).second)
                err("kernel body: " + D->canon + " appears twice — two mentions would be two independent draws "
                    "read as one. Name the draw once (e = " + D->canon + ") and reuse e (draw identity, spec 10.10).");
            DV v; v.scalar = false; v.m = sample_dist(*D, n); return v;
        };
        switch (e->k) {
        case Expr::Num: { DV v; v.s = (float)e->num; return v; }
        case Expr::Ident: {
            auto di = denv.find(e->id);
            if (di != denv.end()) { DV v; v.scalar = false; v.m = di->second; return v; }
            auto si = senv.find(e->id);
            if (si != senv.end()) { DV v; v.s = si->second; return v; }
            auto oi = env.find(e->id);
            if (oi == env.end())
                err("kernel body: '" + e->id + "' is not defined — neither a body binding, the parameter, nor a program name");
            const Value& ov = oi->second;
            if (ov.k == Value::DistV) return lift(ov.dist);
            if (ov.k == Value::Num) { DV v; v.s = (float)ov.num; return v; }
            if (ov.k == Value::Map)
                err("kernel body: a Map is applied, not read — write " + e->id + " # z");
            if (ov.k == Value::Data) {
                // single-draw Datasets enter as CONSTANT blocks (stage 2):
                // one column, broadcast to every draw — how a computed point
                // like w0 = inv(T) # y0 gets into a simulator. Multi-column
                // data stays outside (feed it through the parameter slot).
                if (ov.data->X.num_col() == 1) {
                    DV v; v.scalar = false;
                    int d = (int)ov.data->X.num_row();
                    v.m = Mat("dv", d, n);
                    for (int r = 0; r < d; r++)
                        for (int c = 0; c < n; c++) v.m.elem(r, c) = ov.data->X.elem(r, 0);
                    return v;
                }
                err("kernel body: a multi-column Dataset has no draw-level meaning — a single-column Dataset "
                    "enters as a constant block (a computed point, e.g. w0 = inv(T) # y0 ~ 1); for data-driven "
                    "parameters use the parameter slot ((K | X) resamples X's columns)");
            }
            if (ov.k == Value::Kernel)
                err("kernel body: kernels do not nest in v1 — inline the inner body");
            err("kernel body: '" + e->id + "' has no draw-level meaning (bodies know draws, scalars, "
                "distributions, and trained maps via #)");
        }
        case Expr::Call: {
            if ((e->id == "sin" || e->id == "cos" || e->id == "exp" || e->id == "log" || e->id == "sqrt")
                && e->args.size() == 1 && e->args[0].kw.empty()) {
                DV a = deval(e->args[0].e, denv, senv, n, lifted);
                auto f = [&](float x) -> float {
                    return e->id == "sin" ? sinf(x) : e->id == "cos" ? cosf(x)
                         : e->id == "exp" ? expf(x) : e->id == "log" ? logf(x) : sqrtf(x);
                };
                if (a.scalar) { a.s = f(a.s); return a; }
                for (size_t r = 0; r < a.m.num_row(); r++)
                    for (size_t c = 0; c < a.m.num_col(); c++) a.m.elem(r, c) = f(a.m.elem(r, c));
                return a;
            }
            // dot(a, b): per-draw inner product — a 1-row block of column-wise
            // dots. The reduction that projections are made of (spec 10.10,
            // stage 2): perp of e against u = e - u*(dot(u,e)/dot(u,u)).
            if (e->id == "dot" && e->args.size() == 2 && e->args[0].kw.empty() && e->args[1].kw.empty()) {
                DV a = deval(e->args[0].e, denv, senv, n, lifted);
                DV b = deval(e->args[1].e, denv, senv, n, lifted);
                if (a.scalar || b.scalar)
                    err("kernel body: dot(a, b) reduces two draw BLOCKS to per-draw inner products — for scalars just multiply");
                if (a.m.num_row() != b.m.num_row())
                    err("kernel body: dot(a, b) needs blocks of equal rows (" + to_string((int)a.m.num_row()) +
                        " vs " + to_string((int)b.m.num_row()) + ")");
                DV v; v.scalar = false; v.m = Mat("dv", 1, n);
                for (int c = 0; c < n; c++) {
                    float s = 0;
                    for (size_t r = 0; r < a.m.num_row(); r++) s += a.m.elem(r, c) * b.m.elem(r, c);
                    v.m.elem(0, c) = s;
                }
                return v;
            }
            // jvp(T, z, v): the trained map's Jacobian at z applied to
            // direction v, per draw — central finite differences through two
            // map applications ((T(z+hv) - T(z-hv)) / 2h, h = 1e-3; the same
            // discipline as the path-coefficient derivative in train_field).
            // The manifold tangent T(x,W) = J_G(w0+Wx)·W is this verb
            // (Khoo, Liu & Beaumont 2026, Def. 1). Exact for linear maps
            // (rotations) up to float rounding.
            if (e->id == "jvp" && e->args.size() == 3) {
                for (auto& a : e->args) if (!a.kw.empty())
                    err("kernel body: jvp takes three positional arguments — jvp(T, z, v)");
                Value mv = eval(e->args[0].e);
                if (mv.k != Value::Map)
                    err("kernel body: the first argument of jvp must be a Map (a trained transport)");
                if (mv.map->f->kind == FieldV::WG)
                    err("kernel body: a descent map re-simulates its own gradient flow — it has no pointwise "
                        "Jacobian to probe. jvp takes trained transports (declared-path flows, rotations, "
                        "amortized generators).");
                if (mv.map->f->cond_dim > 0)
                    err("kernel body: jvp of a conditional map needs its condition slot filled at measure level — v1 takes unconditional transports");
                DV z = deval(e->args[1].e, denv, senv, n, lifted);
                DV dir = deval(e->args[2].e, denv, senv, n, lifted);
                if (z.scalar || dir.scalar) err("kernel body: jvp probes a draw block along a draw block — scalars have no direction");
                if (z.m.num_row() != dir.m.num_row())
                    err("kernel body: jvp(T, z, v): z and v need equal rows (" + to_string((int)z.m.num_row()) +
                        " vs " + to_string((int)dir.m.num_row()) + ")");
                const float h = 1e-3f;
                int d = (int)z.m.num_row();
                Mat P("jp", d, n), Q("jq", d, n);
                for (int r = 0; r < d; r++)
                    for (int c = 0; c < n; c++) {
                        P.elem(r, c) = z.m.elem(r, c) + h * dir.m.elem(r, c);
                        Q.elem(r, c) = z.m.elem(r, c) - h * dir.m.elem(r, c);
                    }
                Mat Yp = mv.map->apply(P), Yq = mv.map->apply(Q);
                DV v; v.scalar = false; v.m = Mat("dv", d, n);
                for (int r = 0; r < d; r++)
                    for (int c = 0; c < n; c++)
                        v.m.elem(r, c) = (Yp.elem(r, c) - Yq.elem(r, c)) / (2.0f * h);
                return v;
            }
            // rows(a, lo, hi): slice a draw block's rows [lo, hi) — 0-based,
            // half-open, LITERAL bounds (the checker sees every dimension
            // statically, same discipline as body-for's bounds). This is the
            // stage-3 primitive that unlocks MATRIX-valued parameters: a
            // d*m-vector parameter is m columns stacked, and W·x for x in R^m
            // is written column by column with the existing 1-row broadcast —
            //   z = w0 + rows(w,0,3)*rows(x,0,1) + rows(w,3,6)*rows(x,1,2).
            if (e->id == "rows" && e->args.size() == 3) {
                for (auto& a : e->args) if (!a.kw.empty())
                    err("kernel body: rows takes three positional arguments — rows(block, lo, hi)");
                if (e->args[1].e->k != Expr::Num || e->args[2].e->k != Expr::Num ||
                    e->args[1].e->num != (long)e->args[1].e->num || e->args[2].e->num != (long)e->args[2].e->num)
                    err("kernel body: the bounds of rows(block, lo, hi) must be literal integers — the "
                        "checker sees every block dimension statically (0-based, half-open: rows(w, 0, 3) "
                        "is the first three rows)");
                int lo = (int)e->args[1].e->num, hi = (int)e->args[2].e->num;
                DV a = deval(e->args[0].e, denv, senv, n, lifted);
                if (a.scalar)
                    err("kernel body: rows slices a draw BLOCK — a scalar has no rows");
                int d = (int)a.m.num_row();
                if (lo < 0 || hi <= lo || hi > d)
                    err("kernel body: rows(block, " + to_string(lo) + ", " + to_string(hi) + ") is out of "
                        "range for a " + to_string(d) + "-row block (0-based, half-open: 0 <= lo < hi <= " +
                        to_string(d) + ")");
                DV v; v.scalar = false; v.m = Mat("dv", hi - lo, n);
                for (int r = lo; r < hi; r++)
                    for (int c = 0; c < n; c++) v.m.elem(r - lo, c) = a.m.elem(r, c);
                return v;
            }
            Value ov = eval(e);          // constructor calls evaluate at measure level, then lift
            if (ov.k == Value::DistV) return lift(ov.dist);
            if (ov.k == Value::Num) { DV v; v.s = (float)ov.num; return v; }
            err("kernel body: " + e->id + "(...) has no draw-level meaning — bodies use scalar math "
                "(sin/cos/exp/log/sqrt), dot(a,b), jvp(T,z,v), rows(a,lo,hi), distribution constructors "
                "(auto-lifted to fresh draws), and trained maps via #");
        }
        case Expr::Bin: {
            const string& op = e->id;
            if (op == "|")
                err("kernel body: | is measure-level conditioning; a body is already pointwise in its parameter (spec 10.10)");
            if (op != "+" && op != "-" && op != "*" && op != "/")
                err("kernel body: operator '" + op + "' has no draw-level meaning");
            DV a = deval(e->a, denv, senv, n, lifted), b = deval(e->b, denv, senv, n, lifted);
            auto ap = [&](float x, float y) -> float {
                return op == "+" ? x + y : op == "-" ? x - y : op == "*" ? x * y : x / y;
            };
            if (a.scalar && b.scalar) { DV v; v.s = ap(a.s, b.s); return v; }
            DV v; v.scalar = false;
            if (a.scalar || b.scalar) {
                const Mat& M = a.scalar ? b.m : a.m; float s = a.scalar ? a.s : b.s;
                v.m = Mat("dv", (int)M.num_row(), (int)M.num_col());
                for (size_t r = 0; r < M.num_row(); r++)
                    for (size_t c = 0; c < M.num_col(); c++)
                        v.m.elem(r, c) = a.scalar ? ap(s, M.elem(r, c)) : ap(M.elem(r, c), s);
                return v;
            }
            int ra = (int)a.m.num_row(), rb = (int)b.m.num_row(), d = max(ra, rb);
            if (ra != rb && ra != 1 && rb != 1)
                err("kernel body: blocks of " + to_string(ra) + " and " + to_string(rb) +
                    " rows do not align — draw-level arithmetic is elementwise, with 1-row blocks broadcast");
            v.m = Mat("dv", d, n);
            for (int r = 0; r < d; r++)
                for (int c = 0; c < n; c++)
                    v.m.elem(r, c) = ap(a.m.elem(ra == 1 ? 0 : r, c), b.m.elem(rb == 1 ? 0 : r, c));
            return v;
        }
        case Expr::Push: {
            Value mv = eval(e->a);
            if (mv.k != Value::Map)
                err("kernel body: the left side of # must be a Map (a trained transport)");
            if (mv.map->f->kind == FieldV::WG)
                err("kernel body: a descent map re-simulates its own gradient flow on whatever it is given — "
                    "it is not a pointwise function of a draw. Bodies take trained transports "
                    "(declared-path flows, rotations, amortized generators).");
            if (mv.map->f->cond_dim > 0)
                err("kernel body: conditional maps need their condition slot filled at measure level (spec 10.8) — "
                    "v1 bodies apply unconditional transports only");
            DV z = deval(e->b, denv, senv, n, lifted);
            if (z.scalar) err("kernel body: # applies a map to a draw block, not to a scalar");
            DV v; v.scalar = false; v.m = mv.map->apply(z.m); return v;
        }
        case Expr::DrawFor: {
            // draw-level for-as-expression (spec 10.10): macro expansion;
            // value = row-stack of each copy's FINAL binding — the
            // N-observation block ⊕ᵢ yᵢ that turns a single-observation
            // conditional into a product likelihood. The lift scope resets
            // per copy (fresh draws each round: iid, literally); bindings
            // rebind as usual and persist; the index is a per-copy scalar.
            bool idx_shadow_s = senv.count(e->id), idx_shadow_d = denv.count(e->id);
            float saved_s = idx_shadow_s ? senv[e->id] : 0;
            Mat saved_d = idx_shadow_d ? denv.at(e->id) : Mat("dv", 0, 0);
            vector<Mat> parts;
            for (int i = e->df_lo; i <= e->df_hi; i++) {
                senv[e->id] = (float)i; denv.erase(e->id);
                set<string> lifted_copy;                    // fresh draws per round
                DV last;
                for (auto& [nm, ex] : e->kd_body) {
                    last = deval(ex, denv, senv, n, lifted_copy);
                    if (last.scalar) { senv[nm] = last.s; denv.erase(nm); }
                    else { denv.insert_or_assign(nm, last.m); senv.erase(nm); }
                }
                if (last.scalar)
                    err("body for: the final binding of the loop body is a scalar — the loop's value stacks "
                        "draw BLOCKS (one per round)");
                parts.push_back(last.m);
            }
            senv.erase(e->id);
            if (idx_shadow_s) senv[e->id] = saved_s;
            if (idx_shadow_d) denv.insert_or_assign(e->id, saved_d);
            vector<MatrixView<float>> views;
            for (auto& p : parts) views.push_back(p);
            DV v; v.scalar = false; v.m = vstack<float>(views);
            return v;
        }
        case Expr::Sample:
            err("kernel body: ~ has no meaning here — the body defines ONE draw at a time; vectorization over n "
                "is the interpreter's job at instantiation, (K | prior) ~ n (spec 10.10)");
        case Expr::Mix:
            err("kernel body: `or` mixes MEASURES; a body computes a draw — put the mixture in the parameter's "
                "prior or inside a constructor");
        case Expr::Vec:
            err("kernel body: a bare vector literal has no draw-level meaning — vectors live inside constructors "
                "(gaussian([m1, m2], s))");
        case Expr::KernelDef:
            err("kernel body: kernels do not nest in v1");
        default:
            err("kernel body: this expression form has no draw-level meaning");
        }
    }

    Mat kernel_body_eval(const KernelV& K, const Mat& cond) {
        int n = (int)cond.num_col();
        map<string, Mat> denv; map<string, float> senv; set<string> lifted;
        denv.emplace(K.kd_param, cond);
        DV last;
        for (auto& [nm, ex] : K.kd_body) {
            last = deval(ex, denv, senv, n, lifted);
            if (last.scalar) { senv[nm] = last.s; denv.erase(nm); }
            else denv.insert_or_assign(nm, last.m);
        }
        if (last.scalar)
            err("kernel body: the final binding is a scalar — the last value is the kernel's OUTPUT y, a draw block");
        return last.m;
    }

    // path formulas are built from BARE blocks (2026-07): a conditioned term
    // entering the arithmetic is the retired inline spelling t*(y|x) + ... —
    // the formula lives at the draw level, conditioning at the law level.
    void no_conditioned_term(const Value& v) {
        for (auto& tm : v.rv->terms)
            if (tm.clo >= 0)
                err("a conditioned block cannot enter a path formula: the formula interpolates DRAWS, "
                    "and conditioning is an operation on LAWS. Build the formula from bare blocks and "
                    "attach the index where the law is taken — prob(xt | d), or prob(xt | d0, d1) for "
                    "several (2026-07; spec 10.8)");
    }

    Value eval_given(Value A, Value B) {
        // law-gate conditioning (2026-07): FORMULA | index-block. The formula
        // interpolates DRAWS; conditioning is an operation on LAWS — so the
        // index is attached where the law is taken, prob(xt | d), not inside
        // the arithmetic. The index must be a sibling block (same joint draw)
        // of the term it conditions; extra indices chain (prob(xt | d0, d1))
        // and must follow term order, because condition slots are assembled
        // in term order all the way through training and evaluation (10.8).
        if (A.k == Value::RV) {
            if (B.k != Value::RV || B.rv->terms.size() != 1 || B.rv->terms[0].blo < 0)
                err("the right side of | must be a coordinate block of a joint draw ((y, x) = rv(joint)) — for a mixture of measures use `or`: w1*A or w2*B");
            const RTerm& xt = B.rv->terms[0];
            auto rv = make_shared<RVal>(*A.rv);
            int hit = -1;
            for (size_t k = 0; k < rv->terms.size(); k++)
                if (rv->terms[k].src == xt.src) { hit = (int)k; break; }
            if (hit < 0)
                err("draw identity: this index shares no joint draw with any term of the formula, so "
                    "conditioning on it cannot bite (the law would be unchanged). Destructure once, "
                    "(y, x) = rv(joint), build the formula from y, and condition at the law gate: "
                    "prob(formula | x) (spec 10.8)");
            if (rv->terms[hit].clo >= 0) err("this block already carries a conditioner");
            for (size_t k = hit + 1; k < rv->terms.size(); k++)
                if (rv->terms[k].clo >= 0)
                    err("condition slots are assembled in the formula's term order — attach the indices "
                        "in that order (the k-th declared index conditions the k-th conditioned term, "
                        "and | [..] fills slots in the same order; spec 10.8)");
            rv->terms[hit].clo = xt.blo; rv->terms[hit].chi = xt.bhi;
            Value r; r.k = Value::RV; r.rv = rv;
            return r;
        }
        // kernel evaluation: fill the open condition slots (all slots share one
        // value in v1). Paired needs only joint samples; decoupled needs the
        // joint's conditional sampler — checked inside sample_cond.
        if (A.k == Value::Kernel) {
            const KernelV& K0 = *A.kernel;
            if (K0.inst != 0) err("this kernel is already instantiated — | fills the condition slot once");
            auto K = make_shared<KernelV>(K0);
            if (B.k == Value::RV && B.rv->terms.size() == 1 && B.rv->terms[0].blo >= 0) {
                if (K0.prog)
                    err("paired evaluation reuses a source joint's own draw, but a programmable kernel has no "
                        "source joint — its joint is what instantiation constructs: (K | prior) ~ n (spec 10.10)");
                const RTerm& xt = B.rv->terms[0];
                if (xt.src != K0.src)
                    err("this block belongs to a different draw than the kernel's source pair — paired evaluation "
                        "reuses the SAME draw (draw identity); for decoupled evaluation pass a point [z], a "
                        "Distribution, or a Dataset (spec 10.8)");
                K->inst = 1; K->canon += " | " + xt.srcname();
            } else if (B.k == Value::Vec) {
                if (!K0.prog && (int)B.vec.size() != K0.joint->dx)
                    err("the condition point has dimension " + to_string(B.vec.size()) +
                        " but the kernel's condition block has dimension " + to_string(K0.joint->dx));
                K->inst = 2;
                ostringstream o; o << K0.canon << " | [";
                for (size_t i = 0; i < B.vec.size(); i++) { K->zfix.push_back((float)B.vec[i]); o << (i ? "," : "") << B.vec[i]; }
                o << "]"; K->canon = o.str();
            } else if (B.k == Value::DistV) {
                if (!K0.prog && B.dist->dim != K0.joint->dx) err("condition-measure dimension mismatch");
                if (K0.prog && !B.dist->samplable)
                    err("a programmable kernel's condition slot draws parameters — " + B.dist->canon +
                        " admits no exact sampler");
                K->inst = 3; K->zdist = B.dist; K->canon += " | " + B.dist->canon;
            } else if (B.k == Value::Data) {
                if (!K0.prog && (int)B.data->X.num_row() != K0.joint->dx) err("condition-dataset dimension mismatch");
                K->inst = 4; K->zdata = B.data; K->canon += " | " + B.data->prov;
            } else err("|: the condition slot takes a paired block, a point [z], a Distribution, or a Dataset");
            Value r; r.k = Value::Kernel; r.kernel = K; return r;
        }
        // N-observation conditioning (2026-07, KL decomposition): (q | Obs) —
        // q a plain Dataset (the moving parameter ensemble) and Obs a Dataset
        // (or frozen kernel-joint block) whose COLUMNS are the observations
        // Y_1..Y_N. Unlike (y|x), the observations are FOREIGN to the ensemble:
        // no draw is shared. The value is a conditioned ensemble for descent —
        // it carries no law of its own (see WGPathV::obs for the theorem).
        if (A.k == Value::Data) {
            shared_ptr<Dataset> od; int olo = 0, ohi = 0;
            if (B.k == Value::Data) { od = B.data; ohi = (int)B.data->X.num_row(); }
            else if (B.k == Value::RV && B.rv->terms.size() == 1 &&
                     B.rv->terms[0].data && B.rv->terms[0].blo >= 0) {
                od = B.rv->terms[0].data; olo = B.rv->terms[0].blo; ohi = B.rv->terms[0].bhi;
            } else
                err("(q | Obs) conditions an ensemble on an OBSERVATION SET — the right side must be a "
                    "Dataset of observation columns, or a frozen block of a kernel joint: "
                    "(Ys, Ws) = (K | z) ~ n, then (q | Ys) (spec 10.10)");
            Value r; r.k = Value::RV; r.rv = make_shared<RVal>();
            RTerm tm; tm.coeff = [](float) { return 1.0f; }; tm.cdesc = "1";
            tm.src = fresh_src(); tm.anon = false;
            tm.data = A.data; tm.obsd = od; tm.olo = olo; tm.ohi = ohi;
            r.rv->terms.push_back(move(tm));
            return r;
        }
        err("| applies to a random-variable block (formula construction: y|x), a kernel (evaluation), or a Dataset ensemble (observation conditioning: q | Obs) — spec 10.8/10.10. For a mixture of measures use `or`: w1*A or w2*B");
    }

    Value eval_bin(const ExprP& e) {
        Value A = eval(e->a), B = eval(e->b);
        const string& op = e->id;
        if (op == "|") return eval_given(A, B);
        auto isT = [](const Value& v) { return v.k == Value::TFun; };
        auto isN = [](const Value& v) { return v.k == Value::Num; };
        auto asT = [](const Value& v) {
            if (v.k == Value::TFun) return v.tf;
            float c = (float)v.num; return TF([c](float) { return c; });
        };
        auto tdesc = [](const Value& v) {
            return v.k == Value::TFun ? v.tfdesc : ([&] { ostringstream o; o << v.num; return o.str(); })();
        };
        // Num ∘ Num
        if (isN(A) && isN(B)) {
            Value r; r.k = Value::Num;
            r.num = op == "+" ? A.num + B.num : op == "-" ? A.num - B.num
                  : op == "*" ? A.num * B.num : A.num / B.num;
            return r;
        }
        // t-expressions
        if ((isT(A) || isN(A)) && (isT(B) || isN(B)) && (isT(A) || isT(B))) {
            TF a = asT(A), b = asT(B);
            Value r; r.k = Value::TFun;
            if (op == "+") r.tf = [a, b](float x) { return a(x) + b(x); };
            else if (op == "-") r.tf = [a, b](float x) { return a(x) - b(x); };
            else if (op == "*") r.tf = [a, b](float x) { return a(x) * b(x); };
            else r.tf = [a, b](float x) { return a(x) / b(x); };
            r.tfdesc = "(" + tdesc(A) + op + tdesc(B) + ")";
            return r;
        }
        // coordinate expressions (log-densities)
        auto isXY = [](const Value& v) { return v.k == Value::XYFun; };
        if ((isXY(A) || isN(A)) && (isXY(B) || isN(B)) && (isXY(A) || isXY(B))) {
            auto asXY = [](const Value& v) {
                if (v.k == Value::XYFun) return v.xy;
                float c = (float)v.num; return function<float(float,float)>([c](float, float) { return c; });
            };
            auto xdesc = [](const Value& v) {
                return v.k == Value::XYFun ? v.xydesc : ([&] { ostringstream o; o << v.num; return o.str(); })();
            };
            auto a = asXY(A), b = asXY(B);
            Value r; r.k = Value::XYFun;
            if (op == "+") r.xy = [a, b](float u, float w) { return a(u, w) + b(u, w); };
            else if (op == "-") r.xy = [a, b](float u, float w) { return a(u, w) - b(u, w); };
            else if (op == "*") r.xy = [a, b](float u, float w) { return a(u, w) * b(u, w); };
            else r.xy = [a, b](float u, float w) { return a(u, w) / b(u, w); };
            r.xydesc = "(" + xdesc(A) + op + xdesc(B) + ")";
            return r;
        }
        // weight * Dist (mixture weights)
        if (op == "*" && isN(A) && B.k == Value::DistV) {
            auto D = make_shared<Dist>(*B.dist);
            for (auto& c : D->comps) c.w *= (float)A.num;
            ostringstream o; o << A.num << "*" << B.dist->canon;
            D->canon = o.str();
            Value r; r.k = Value::DistV; r.dist = D; return r;
        }
        // coeff(t) * RV-able  → RV term scaling
        if (op == "*" && (isT(A) || isN(A)) && (liftable(B) || B.k == Value::RV)) {
            Value rb = B.k == Value::RV ? B : lift(B);
            no_conditioned_term(rb);
            TF c = asT(A); string cd = tdesc(A);
            Value r; r.k = Value::RV; r.rv = make_shared<RVal>();
            for (auto tm : rb.rv->terms) {
                TF old = tm.coeff;
                tm.coeff = [c, old](float x) { return c(x) * old(x); };
                tm.cdesc = tm.cdesc == "1" ? cd : cd + "*" + tm.cdesc;
                r.rv->terms.push_back(move(tm));
            }
            return r;
        }
        if (op == "*" && (isT(B) || isN(B)) && (liftable(A) || A.k == Value::RV))
            { swap(A, B); ExprP e2 = make_shared<Expr>(*e); e2->a = e->b; e2->b = e->a; return eval_bin(e2); }
        // RV ± RV (with lifting)
        if ((op == "+" || op == "-") &&
            (A.k == Value::RV || liftable(A)) && (B.k == Value::RV || liftable(B))) {
            Value ra = A.k == Value::RV ? A : lift(A);
            Value rb = B.k == Value::RV ? B : lift(B);
            no_conditioned_term(ra); no_conditioned_term(rb);
            return rv_add(ra, rb, op == "+" ? 1.0f : -1.0f);
        }
        // —— field algebra (spec 10.3.1): Fields are a vector space ——
        auto isField = [](const Value& v) { return v.k == Value::Field; };
        if (op == "*" && ((isN(A) && isField(B)) || (isN(B) && isField(A)))) {
            if (isN(B)) swap(A, B);
            return field_lincomb(field_terms(B.field, (float)A.num));
        }
        if (op == "/" && isField(A) && isN(B)) {
            if (B.num == 0) err("division of a field by zero");
            return field_lincomb(field_terms(A.field, (float)(1.0 / B.num)));
        }
        if ((op == "+" || op == "-") && isField(A) && isField(B)) {
            auto ts = field_terms(A.field, 1.0f);
            for (auto& t : field_terms(B.field, op == "-" ? -1.0f : 1.0f)) ts.push_back(t);
            return field_lincomb(ts);
        }
        if ((op == "+" || op == "-") && isN(A) && A.num == 0 && isField(B))   // unary minus: 0 - v
            return field_lincomb(field_terms(B.field, op == "-" ? -1.0f : 1.0f));
        err("unsupported operation '" + op + "' for these operand types (hint: a path formula is a sum of coefficient(t) * random-variable terms)" +
            ((env.count("x1") || env.count("x2"))
                 ? string(" — note: x1/x2 are bound in this program, shadowing the coordinate symbols of unnormalized(...)")
                 : ""));
    }

    Value eval_mix(const ExprP& e) {
        auto D = make_shared<Dist>();
        ostringstream cn;
        bool first = true;
        for (auto& pe : e->parts) {
            Value v = eval(pe);
            if (v.k != Value::DistV) err("the components of a mixture `or` must be distributions, optionally weighted, e.g. 0.7*gaussian(...) or 0.3*gaussian(...)");
            for (auto c : v.dist->comps) D->comps.push_back(c);
            D->dim = v.dist->dim;
            if (!first) cn << " or ";
            cn << v.dist->canon;
            first = false;
        }
        D->has_score = all_of(D->comps.begin(), D->comps.end(),
                              [](const Dist::Comp& c) { return c.kind == "gaussian"; });
        D->canon = cn.str();
        Value v; v.k = Value::DistV; v.dist = D; return v;
    }

    // —— distributions ————————————————————————————————————————————
    Value dist_call(const string& name, const vector<Arg>& args) {
        auto D = make_shared<Dist>();
        Dist::Comp c; c.kind = name;
        ostringstream cn; cn << name << "(";
        auto num_at = [&](size_t i, float dflt) {
            if (args.size() > i) { Value v = eval(args[i].e); if (v.k != Value::Num) err(name + ": : expected a numeric argument"); return (float)v.num; }
            return dflt;
        };
        if (name == "gaussian") {
            if (args.empty()) err("gaussian(mean[, std]) requires a mean vector");
            Value m = eval(args[0].e);
            if (m.k != Value::Vec) err("the mean of gaussian must be a vector, e.g. [0, 0]");
            for (double x : m.vec) c.mean.push_back((float)x);
            c.s1 = num_at(1, 1.0f);
            D->dim = (int)c.mean.size(); D->has_score = true;
            cn << "["; for (size_t i = 0; i < m.vec.size(); i++) cn << (i ? "," : "") << m.vec[i]; cn << "]," << c.s1;
        } else if (name == "uniform") {
            Value lo = eval(args.at(0).e), hi = eval(args.at(1).e);
            if (lo.k != Value::Vec || hi.k != Value::Vec) err("uniform(lo, hi) requires two vectors");
            for (size_t i = 0; i < lo.vec.size(); i++) c.mean.push_back((float)((lo.vec[i] + hi.vec[i]) / 2));
            c.s1 = (float)((hi.vec[0] - lo.vec[0]) / 2);
            D->dim = (int)c.mean.size();
            cn << "..";
        } else if (name == "mixed_sources") {
            // ICA toy (SICA line): two INDEPENDENT non-Gaussian sources —
            // s1 ~ uniform(-1.5, 1.5), s2 ~ bimodal Gaussian mixture — pushed
            // through the linear mixture x = [[1, m], [m, 1]] s. A pair-joint
            // over (x, s): rows [0,2) = the observed mixture, rows [2,4) = the
            // true sources, one draw. Destructure (X, S) = rv(mixed_sources(m)) ~ n
            // to keep the ground truth for judging recovery; the demo task is
            // still to recover the product structure from X alone.
            c.mean = {0, 0};
            c.kind = "mixed";
            c.s1 = num_at(0, 0.7f);              // mixing coefficient m
            D->dim = 4; D->dy = 2; D->dx = 2;
            D->pair_blocks = true;               // x and s are peer coordinates of ONE draw
            cn << c.s1;
        } else if (name == "mixed_ar") {
            // ICA toy, GAUSSIAN time-series version — the identifiability
            // wall made runnable: two stationary Gaussian AR(1) sources with
            // DIFFERENT autocorrelations (rho = 0.9 / -0.6), same mixing and
            // pair-joint layout as mixed_signals. Every time-marginal is
            // Gaussian, so INSTANTANEOUS independence-seeking provably cannot
            // identify the rotation (the cloud carries no signal beyond
            // covariance); the separation signal lives entirely in the LAGGED
            // structure — window() + decouple(block=) + rotation(block=)
            // recover it (SOBI's regime, full-distribution machinery).
            c.mean = {0, 0};
            c.kind = "arjoint";
            c.s1 = num_at(0, 0.7f);              // mixing coefficient m
            D->dim = 4; D->dy = 2; D->dx = 2;
            D->pair_blocks = true;
            cn << c.s1;
        } else if (name == "mixed_nl") {
            // ICA toy, NONLINEAR-MIXING version: the mixed_ar AR pair
            // through an instantaneous invertible nonlinear mixture (two
            // residual GELU layers; see the "nljoint" branch of sample_dist).
            // The knob a is the residual gain — 0 = identity mixing,
            // ~0.5 = the invertibility edge. This toy is the reason the
            // instantaneous demixer class exists: linear members (whiten +
            // any rotation) cannot contain the inverse warp.
            c.mean = {0, 0};
            c.kind = "nljoint";
            c.s1 = num_at(0, 0.4f);              // residual gain a
            D->dim = 4; D->dy = 2; D->dx = 2;
            D->pair_blocks = true;
            cn << c.s1;
        } else if (name == "mixed_signals") {
            // ICA toy, TIME-SERIES version: same pair-joint shape as
            // mixed_sources ([0,2) = observed mixture, [2,4) = true sources,
            // x = [[1,m],[m,1]] s), but one draw of n columns is ONE
            // trajectory — see the "signals" branch of sample_dist for the
            // waveforms and the ergodicity note. plot_signal shows the
            // channels as waveforms; instantaneous methods only ever see the
            // marginal cloud, which the incommensurate periods keep honest.
            c.mean = {0, 0};
            c.kind = "signals";
            c.s1 = num_at(0, 0.7f);              // mixing coefficient m
            D->dim = 4; D->dy = 2; D->dx = 2;
            D->pair_blocks = true;
            cn << c.s1;
        } else if (name == "linear_gaussian" || name == "sine_gaussian") {
            // toy JOINTS over (y, x): x ~ N(0,1); linear: y = slope*x + noise*e;
            // sine: y = sin(freq*x) + noise*e. Carry block structure and a
            // closed-form conditional sampler (spec 10.8).
            c.mean = {0, 0};
            c.kind = name == "linear_gaussian" ? "linjoint" : "sinejoint";
            c.s1 = num_at(0, name == "linear_gaussian" ? 0.5f : 2.0f);
            c.s2 = num_at(1, name == "linear_gaussian" ? 0.3f : 0.2f);
            D->dim = 2; D->dy = 1; D->dx = 1; D->has_cond_sampler = true;
            cn << c.s1 << "," << c.s2;
        } else {
            c.mean = {0, 0};
            if (name == "moons")  { c.s1 = num_at(0, 0.1f); cn << c.s1; }
            else if (name == "ring")   { c.s1 = num_at(0, 2.0f); c.s2 = num_at(1, 0.1f); cn << c.s1 << "," << c.s2; }
            else if (name == "spiral") { c.s1 = num_at(0, 2.0f); c.s2 = num_at(1, 0.1f); cn << c.s1 << "," << c.s2; }
            else if (name == "torus") {
                // torus(R, r, noise): a 2-D manifold in R^3 — ring's big
                // sibling. s1 = major radius, s2 = minor; the noise rides in
                // mean[0] (Comp has two scalar slots; documented hack).
                c.s1 = num_at(0, 1.0f); c.s2 = num_at(1, 0.4f);
                c.mean = {num_at(2, 0.05f)};
                D->dim = 3;
                cn << c.s1 << "," << c.s2 << "," << c.mean[0];
            }
            else err("unknown distribution: " + name);
        }
        cn << ")";
        D->comps.push_back(c);
        D->canon = cn.str();
        Value v; v.k = Value::DistV; v.dist = D; return v;
    }

    // —— calls ————————————————————————————————————————————————————
    Value arg_kw(const vector<Arg>& args, const string& kw) {
        for (auto& a : args) if (a.kw == kw) return eval(a.e);
        Value v; v.k = Value::Num; v.num = NAN; return v;
    }

    Value eval_call(const ExprP& e) {
        const string& f = e->id;
        if (f == "gaussian" || f == "uniform" || f == "moons" || f == "ring" || f == "spiral" ||
            f == "torus" || f == "linear_gaussian" || f == "sine_gaussian" || f == "mixed_sources" ||
            f == "mixed_signals" || f == "mixed_ar" || f == "mixed_nl")
            return dist_call(f, e->args);

        // unary math on t-expressions / numbers
        if (f == "sqrt" || f == "exp" || f == "log" || f == "sin" || f == "cos") {
            Value a = eval(e->args.at(0).e);
            auto fn = f == "sqrt" ? (float(*)(float))sqrtf : f == "exp" ? expf
                    : f == "log" ? logf : f == "sin" ? sinf : cosf;
            if (a.k == Value::Num) { Value r; r.k = Value::Num; r.num = fn((float)a.num); return r; }
            if (a.k == Value::TFun) {
                TF g = a.tf; Value r; r.k = Value::TFun;
                r.tf = [g, fn](float x) { return fn(g(x)); };
                r.tfdesc = f + "(" + a.tfdesc + ")";
                return r;
            }
            if (a.k == Value::XYFun) {
                auto g = a.xy; Value r; r.k = Value::XYFun;
                r.xy = [g, fn](float u, float w) { return fn(g(u, w)); };
                r.xydesc = f + "(" + a.xydesc + ")";
                return r;
            }
            err(f + "  applies only to numbers, t-expressions, or coordinate (x1/x2) expressions");
        }

        if (f == "unnormalized") {
            Value a = eval(e->args.at(0).e);
            if (a.k != Value::XYFun)
                err(string("unnormalized(L) takes a log-density expression in the coordinate symbols x1, x2 "
                    "(defined up to an additive constant)") +
                    (env.count("x1") || env.count("x2")
                         ? " — note: x1/x2 are bound to values in this program, shadowing the coordinate symbols; rename those bindings"
                         : ""));
            auto D = make_shared<Dist>();
            D->dim = 2; D->has_score = true; D->samplable = false;
            D->logp = a.xy;
            D->canon = "unnormalized(" + a.xydesc + ")";
            Dist::Comp c; c.kind = "unnorm"; c.mean = {0, 0}; D->comps.push_back(c);
            Value r; r.k = Value::DistV; r.dist = D; return r;
        }
        if (f == "rv") {
            Value a = eval(e->args.at(0).e);
            if (a.k == Value::RV)
                err("this is already a random variable — identity is minted exactly once: rv() for a measure, "
                    "the destructuring bind for a joint. Re-wrapping would have to mean either nothing (it has "
                    "an identity) or a fresh draw (silently breaking any coupling it belongs to), so both are "
                    "refused. For an independent copy, rv(...) the underlying measure again; for its law, "
                    "prob(x); to use the pairing, put both blocks in one formula.");
            if (!liftable(a)) err("rv(D) takes a Distribution or a Dataset");
            Value r = lift(a);
            r.rv->terms[0].anon = false;      // named draw: reuse = same draw
            return r;
        }
        if (f == "prob") {
            Value a = eval(e->args.at(0).e);
            if (a.k != Value::RV)
                err("prob(.) takes the law of a random-variable expression; its argument should be an interpolation formula such as t*data + (1-t)*noise");
            // law-gate indices (2026-07): prob(xt | d0, d1, ...) — the commas
            // make the extra indices call arguments; attach each in turn
            // (sibling-draw and term-order checks live in eval_given)
            for (size_t i = 1; i < e->args.size(); i++)
                a = eval_given(a, eval(e->args[i].e));
            // t-free single draw: the law is a MEASURE, not a curve of measures —
            // prob is the one gate from the RV world back to the samplable world,
            // and Law(X) of an unscaled draw is representable exactly (spec 10.3:
            // for a couple block this is the marginal; coupling never moves it).
            if (a.rv->terms.size() == 1 && a.rv->terms[0].obsd)
                err("(q | Obs) is a conditioned ENSEMBLE, not a random variable — it has no law of its own "
                    "until the descent runs. Condition the path: descent(reverseKL(P), from=(q | Obs)), "
                    "then push the same conditioned ensemble through its flow (spec 10.10).");
            if (a.rv->terms.size() == 1 && a.rv->terms[0].cdesc == "1" && a.rv->terms[0].clo < 0) {
                const RTerm& tm = a.rv->terms[0];
                if (tm.dist) {
                    if (tm.dist->pair_blocks && tm.blo >= 0 &&
                        (tm.dist->cpl_a || tm.dist->cpl_ax)) {   // couple block: the stored marginal
                        Value r;
                        if (tm.blo == 0) {
                            if (tm.dist->cpl_a) { r.k = Value::DistV; r.dist = tm.dist->cpl_a; }
                            else { r.k = Value::Data; r.data = tm.dist->cpl_ax; }
                        } else if (tm.dist->cpl_map) {           // via=paired: marginal = T # base (lazy)
                            auto pd = make_shared<PushedDist>();
                            pd->map = tm.dist->cpl_map; pd->base = tm.dist->cpl_b;
                            pd->canon = pd->map->desc + "#" + pd->base->canon;
                            r.k = Value::Pushed; r.pushed = pd;
                        } else if (tm.dist->cpl_b) { r.k = Value::DistV; r.dist = tm.dist->cpl_b; }
                        else { r.k = Value::Data; r.data = tm.dist->cpl_bx; }
                        return r;
                    }
                    if (tm.blo >= 0)
                        err("the marginal of " + tm.srcname() + " has no closed form in this prototype (a constructed "
                            "joint's block is a compound) — sample the joint and read the block off the pair");
                    Value r; r.k = Value::DistV; r.dist = tm.dist; return r;
                }
                if (tm.blo < 0) { Value r; r.k = Value::Data; r.data = tm.data; return r; }
                Mat B("blk", tm.bhi - tm.blo, (int)tm.data->X.num_col());
                for (int rr = tm.blo; rr < tm.bhi; rr++)
                    for (size_t c = 0; c < tm.data->X.num_col(); c++)
                        B.elem(rr - tm.blo, c) = tm.data->X.elem(rr, c);
                Value r; r.k = Value::Data;
                r.data = make_shared<Dataset>(move(B), tm.srcname());
                return r;
            }
            auto pv = make_shared<PathV>(); pv->rv = a.rv;
            ostringstream cn;
            for (size_t i = 0; i < a.rv->terms.size(); i++) {
                const auto& tm = a.rv->terms[i];
                cn << (i ? " + " : "") << tm.cdesc << "*" << tm.srcname();
                if (tm.clo >= 0) cn << " | x";
            }
            pv->canon = "prob(" + cn.str() + ")";
            Value r; r.k = Value::Path; r.path = pv; return r;
        }
        if (f == "descent") {
            Value d0 = eval(e->args.at(0).e);
            if (d0.k != Value::Div) err("the first argument of descent(D, from=q0, time=, metric=w2) must be a Divergence");
            auto wp = make_shared<WGPathV>(); wp->div = d0.div;
            for (auto& a : e->args) {
                if (a.kw == "from") {
                    Value q = eval(a.e);
                    if (q.k == Value::DistV)     { wp->from_d = q.dist; wp->from_prov = q.dist->canon; }
                    else if (q.k == Value::Data) { wp->from_x = q.data; wp->from_prov = q.data->prov; }
                    else if (q.k == Value::RV && q.rv->terms.size() == 1 && q.rv->terms[0].obsd) {
                        // from=(q | Obs) — N-observation conditioning (2026-07,
                        // KL decomposition): the whole ensemble moves in
                        // parameter space; the observations enter through the
                        // signed sum of fields (see WGPathV::obs).
                        const RTerm& tm = q.rv->terms[0];
                        wp->from_x = tm.data;
                        wp->obs = tm.obsd; wp->obs_lo = tm.olo; wp->obs_hi = tm.ohi;
                        string on = tm.obsd->prov +
                                    (tm.olo == 0 && tm.ohi == (int)tm.obsd->X.num_row() ? ""
                                     : tm.olo == 0 ? ":y" : ":x");
                        wp->from_prov = "(" + tm.data->prov + "|" + on + ")";
                    }
                    else if (q.k == Value::RV && q.rv->terms.size() == 1 && q.rv->terms[0].clo >= 0) {
                        // from=(y | x) — conditional descent (2026-07, SBI line):
                        // the ensemble is the frozen JOINT; the y-block moves,
                        // the conditioning block is pinned. Same `|` as
                        // everywhere else: conditioning = pinning rows of one
                        // draw. See WGPathV::free_lo for the theorem.
                        const RTerm& tm = q.rv->terms[0];
                        if (!tm.data)
                            err("descent moves a concrete ensemble — freeze the joint first: "
                                "(x0, y0) = rv(couple(...)) ~ n, then from=(y0 | x0)");
                        int d = (int)tm.data->X.num_row();
                        bool tiled = (tm.bhi - tm.blo) + (tm.chi - tm.clo) == d &&
                                     min(tm.blo, tm.clo) == 0 && max(tm.bhi, tm.chi) == d &&
                                     (tm.bhi == tm.clo || tm.chi == tm.blo);
                        if (!tiled)
                            err("from=(y | x): the moving block and the pinned block must tile the whole joint — "
                                "rows neither moved nor pinned would ride every kernel weight as silent ghosts");
                        wp->from_x = tm.data;
                        wp->free_lo = tm.blo; wp->free_hi = tm.bhi;
                        wp->from_prov = "(" + tm.srcname() + "|x)";
                    }
                    else err("from= of descent must be a Distribution, a Dataset, or a conditioned block of a "
                             "frozen joint, (y0 | x0) — the initial measure of the path");
                }
                if (a.kw == "time") wp->T = (float)eval(a.e).num;
                if (a.kw == "metric") {
                    Value m = eval(a.e);
                    if (m.k != Value::Symbol) err("metric= expects a geometry name (w2 / stein / fisher_rao)");
                    if (m.sym == "fisher_rao")
                        err("metric=fisher_rao (birth-death dynamics) is not implemented in this prototype");
                    if (m.sym != "w2" && m.sym != "stein") err("unknown geometry: " + m.sym);
                    wp->metric = m.sym;
                }
                // family= names the manifold the PATH moves on (2026-07, ICA
                // line): free (default) = all of W2 space; rotation = the
                // constrained steepest descent on the orbit {R # q0 : R in
                // SO(d)}. The constraint is a property of the CURVE — the
                // projected field and flow's Lie-group step both follow from
                // it as theorems (metric picks the ambient geometry, family
                // picks the submanifold; the two axes compose).
                if (a.kw == "family") {
                    if (a.e->k == Expr::Call && a.e->id == "rotation") {
                        // rotation(block=L): the rotation acts on CHANNELS and
                        // identically at each of the L lags (R ⊗ I_L) — the
                        // process-ICA constraint for window()ed trajectories.
                        wp->rotation = true;
                        for (auto& ra : a.e->args) {
                            if (ra.kw != "block")
                                err("family=rotation takes block= only (the lag count L of window()ed data; "
                                    "the rotation is then R ⊗ I_L — one channel rotation shared across lags)");
                            wp->rot_block = (int)eval(ra.e).num;
                        }
                        if (wp->rot_block < 1)
                            err("family=rotation(block=L): L must be a positive lag count");
                    } else {
                        Value m = eval(a.e);
                        if (m.k != Value::Symbol || (m.sym != "rotation" && m.sym != "free"))
                            err("family= names the manifold the descent path moves on: free (default — all of W2 "
                                "space) or rotation (the curve of rotations of from=; its velocity is the so(d) "
                                "projection of the free descent field — the linear-ICA search space. Whiten the "
                                "ensemble first: rotations preserve covariance). On window()ed data write "
                                "rotation(block=L): channel rotations shared across the L lags." +
                                (m.k == Value::Symbol ? " Unknown family: " + m.sym : string(" family= expects a name.")));
                        wp->rotation = m.sym == "rotation";
                    }
                }
            }
            if (wp->from_prov.empty())
                err("descent requires from=: a gradient flow is an initial-value problem, so the curve is determined jointly by the divergence and the initial measure");
            if (wp->metric == "stein" && d0.div->name != "reverseKL")
                err("metric=stein kernelizes a SCORE-driven descent (SVGD is reverse KL in the kernelized Stein "
                    "geometry); the " + d0.div->name + " descent is defined here in W2 — omit metric=.");
            if (!d0.div->moving_prov.empty() && d0.div->moving_prov != wp->from_prov)
                err("the divergence's second slot (" + d0.div->moving_prov + ") disagrees with from= (" + wp->from_prov +
                    ") — the moving argument is the initial measure; specify it in one place only");
            if (wp->rotation && wp->from_x) {
                const Mat& X = wp->from_x->X;
                int d = (int)X.num_row(), n = (int)X.num_col(), L = wp->rot_block;
                if (d % L != 0)
                    err("family=rotation(block=" + to_string(L) + ") needs the dimension (" + to_string(d) +
                        ") to be a multiple of the lag count L — window(X, L) produces channels*L rows");
                // reachability note at the declaration site: rotations (R ⊗ I_L)
                // preserve the LAG-POOLED channel covariance, so if that is not
                // ~identity the orbit cannot reach a whitened target. from= is a
                // Dataset here, so the check is deterministic and free. (For
                // block=1 this is the plain covariance check; Dist starts are
                // checked on the concrete ensemble at apply.)
                int C = d / L;
                vector<float> mu(d, 0);
                for (int r = 0; r < d; r++) { double s = 0; for (int i = 0; i < n; i++) s += X.elem(r, i); mu[r] = (float)(s / n); }
                float dev = 0;
                for (int a2 = 0; a2 < C; a2++) for (int b2 = 0; b2 < C; b2++) {
                    double s = 0;
                    for (int l = 0; l < L; l++)
                        for (int i = 0; i < n; i++)
                            s += (double)(X.elem(a2 * L + l, i) - mu[a2 * L + l]) * (X.elem(b2 * L + l, i) - mu[b2 * L + l]);
                    dev = max(dev, fabsf((float)(s / ((double)n * L)) - (a2 == b2 ? 1.0f : 0.0f)));
                }
                if (dev > 0.25f)
                    printf("note: family=rotation confines this path to the rotation orbit of from=, whose "
                           "lag-pooled channel covariance is not white (max |cov − I| = %.2f); rotations "
                           "preserve it, so whiten(...) the channels first if the target is a "
                           "whitened/decoupled law.\n", dev);
            }
            // from=(y|x) guards (2026-07, SBI line): conditioning as row
            // pinning. The theorem that makes this a CONDITIONAL descent and
            // not a hack: log p(y|x) = log p(y,x) − log p(x), and the second
            // term dies under ∇_y — so the free rows of the JOINT score
            // difference are exactly the conditional score difference. That
            // cancellation is a property of the LOG (reverseKL); mmd's
            // witness and w2's displacement do not factor this way.
            if (wp->free_lo >= 0) {
                if (d0.div->name != "reverseKL")
                    err("from=(y | x) pins the observed block and descends the CONDITIONAL law, which is a "
                        "theorem about the log: ∇_y log p(y|x) = ∇_y log p(y,x) (log p(x) dies under ∇_y). "
                        "The " + d0.div->name + " field does not factor this way — its restriction to the "
                        "free rows is a different (unconditional) object. Use reverseKL, or drop the |.");
                if (wp->rotation)
                    err("from=(y | x) pins rows while family=rotation moves every coordinate through one "
                        "orthogonal matrix — a map frozen on a block and rotating the rest is not an element "
                        "of SO(d). Pick one: conditioning (|) or the constrained orbit (family=rotation).");
            }
            // from=(q | Obs) guards (2026-07, KL decomposition): the split
            //   KL[q‖p(·|Y_1..N)] = Σ_i KL[q‖p(·|Y_i)] − (N−1)·KL[q‖prior] + c
            // is Bayes plus log-likelihood ADDITIVITY — a property of the log,
            // and of a target that is a joint sample library whose observed
            // rows can be pinned. Anything else gets a teaching error.
            if (wp->obs) {
                if (d0.div->name != "reverseKL")
                    err("(q | Obs) splits the N-observation target by log-likelihood additivity — "
                        "log p(Y_1..N|w) = Σ_i log p(Y_i|w) — which is a property of the LOG. The " +
                        d0.div->name + " objective does not decompose observation-by-observation. "
                        "Use reverseKL, or condition on nothing.");
                if (wp->rotation)
                    err("(q | Obs) descends a posterior over parameters while family=rotation constrains the "
                        "curve to a rotation orbit — the posterior of a simulator is not a rotation of its "
                        "prior. Pick one.");
                if (!d0.div->target_x)
                    err("(q | Obs) pins observation rows of a JOINT SAMPLE LIBRARY — the divergence target "
                        "must be Dataset samples of the joint (y; w), e.g. P = (K | prior) ~ n. A closed-form "
                        "target has no simulator rows to pin.");
                int dY = wp->obs_hi - wp->obs_lo;
                int dw = (int)wp->from_x->X.num_row();
                int dj = (int)d0.div->target_x->X.num_row();
                if (dY + dw != dj)
                    err("block mismatch: each observation has " + to_string(dY) + " rows and the moving ensemble "
                        "has " + to_string(dw) + ", but the target joint has " + to_string(dj) + " rows — the "
                        "library's layout is (y; w), observed rows FIRST, and the two blocks must tile it "
                        "(spec 10.8/10.10).");
                if (d0.div->target_x->blk > 0 && d0.div->target_x->blk != dY)
                    err("the target joint's observed block has " + to_string(d0.div->target_x->blk) +
                        " rows but each observation has " + to_string(dY) +
                        " — the pinned rows must be exactly the library's y-block.");
            }
            wp->canon = "descent(" + d0.div->canon + ",from=" + wp->from_prov +
                        (wp->metric == "w2" ? "" : ",metric=" + wp->metric) +
                        (wp->rotation ? (wp->rot_block > 1 ? ",family=rotation(block=" + to_string(wp->rot_block) + ")"
                                                           : ",family=rotation") : "") + ")";
            Value r; r.k = Value::WGPath; r.wgpath = wp; return r;
        }
        if (f == "field") {
            Value p0 = eval(e->args.at(0).e);
            if (p0.k == Value::Div)
                err("a divergence induces a path, not a field (spec 10.2.1): construct descent(D, from=q0) first, then extract with field(...)");
            if (p0.k != Value::Path && p0.k != Value::WGPath)
                err("the first argument of field must be a path: prob(...) (declared) or descent(...) (gradient flow)");
            // —— WG path: the field IS given (steepest descent); estimator = nw ——
            if (p0.k == Value::WGPath) {
                auto& divd = *p0.wgpath->div;
                string est = "";
                for (auto& a : e->args)
                    if (a.kw == "estimator" && a.e->k == Expr::Call) {
                        est = a.e->id;
                        for (auto& ra : a.e->args)
                            if (ra.kw == "normalize")             // kernel= accepted; rbf is the only kernel
                                err("normalize= is gone (2026-07): the form of the field is a THEOREM of the "
                                    "divergence and the geometry, not a switch — reverseKL under metric=w2 is the "
                                    "normalized NW estimate of \u2207log(p/q); under metric=stein it is the exact "
                                    "(unnormalized) SVGD update; mmd is the witness gradient \u2207(p\u0302\u2212q\u0302). "
                                    "Pick the divergence and the metric; the normalization follows.");
                    }
                if (est == "locallinear" || est == "ratio_then_grad")
                    err("estimator=" + est + "  is not implemented in this prototype — use nw(kernel=rbf).");
                // —— estimator=dsm (2026-07): warm-started neural score descent ——
                // Both scores of the reverseKL field estimated by fixed-sigma
                // denoising score matching: the target joint's net trained once,
                // the moving cloud's net warm-started (a few SGD steps per flow
                // step). Same sigma both sides — the neural mirror of the
                // shared-bandwidth rule (the fixpoint is p_sigma = q_sigma).
                // sigma replaces the kernel bandwidth; it does not disappear.
                shared_ptr<NetSpec> dsm_spec; float dsm_sigma = 0.1f, dsm_trainlr = 1e-3f;
                int dsm_warm = 16, dsm_pre = 1500, dsm_batch = 128;
                if (est == "dsm") {
                    if (divd.name != "reverseKL")
                        err("estimator=dsm estimates the two scores of the reverseKL field — a " + divd.name +
                            " descent has no score difference to estimate");
                    if (!divd.target_x)
                        err("estimator=dsm learns the target score from SAMPLES — give reverseKL a Dataset target "
                            "(a score target needs no net on the p-side: use nw, whose p-term evaluates the exact score)");
                    if (p0.wgpath->obs)
                        err("estimator=dsm does not support the N-observation decomposition yet — use nw "
                            "(the log-weight assembly is kernel-specific; the neural field-level sum is future work)");
                    if (p0.wgpath->rotation)
                        err("estimator=dsm with family=rotation is not supported — the so(d) projection rides on nw");
                    if (p0.wgpath->metric == "stein")
                        err("the dsm score difference IS the W2 velocity ∇log(p/q) — metric=stein's kernelized "
                            "form is nw's; use metric=w2 (the default)");
                    for (auto& a : e->args)
                        if (a.kw == "estimator")
                            for (auto& ra : a.e->args) {
                                if (ra.kw.empty()) {
                                    Value nv = eval(ra.e);
                                    if (nv.k != Value::Net) err("dsm takes an mlp: estimator=dsm(mlp(d -> ... -> d), sigma=...)");
                                    dsm_spec = nv.net;
                                } else if (ra.kw == "sigma")   dsm_sigma  = (float)eval(ra.e).num;
                                else if (ra.kw == "warm")      dsm_warm   = (int)eval(ra.e).num;
                                else if (ra.kw == "steps")     dsm_pre    = (int)eval(ra.e).num;
                                else if (ra.kw == "trainlr")   dsm_trainlr = (float)eval(ra.e).num;
                                else if (ra.kw == "batch")     dsm_batch  = (int)eval(ra.e).num;
                                else err("dsm: unknown argument " + ra.kw + "=  (takes a net, sigma=, warm=, steps=, trainlr=, batch=)");
                            }
                    if (!dsm_spec) err("estimator=dsm needs a net: estimator=dsm(mlp(d -> ... -> d), sigma=...)");
                    if (dsm_sigma <= 0) err("dsm: sigma must be positive (it is the smoothing scale — the bandwidth's successor, not its abolition)");
                }
                // dispatch (2026-07): divergence × metric decides the field's form
                if (divd.name == "w2") {
                    if (est == "nw")
                        err("the w2 descent field is the barycentric displacement of the entropic plan — no kernel "
                            "regression is involved. Omit estimator=, or write estimator=sinkhorn for emphasis.");
                    if (!est.empty() && est != "sinkhorn")
                        err("the w2 descent admits estimator=sinkhorn only (the plan solver)");
                } else if (est != "nw" && est != "dsm")
                    err("field over a " + divd.name + " descent requires estimator=nw(kernel=rbf) or estimator=dsm(mlp(...))");
                auto& div = *p0.wgpath->div;
                if (div.name == "reverseKL" && div.target_d && !div.target_d->has_score)
                    err("nw needs either the target's score \u2207log p or its samples, and " + div.target_d->canon +
                        " provides neither directly (no analytic score). It does have a sampler \u2014 descend onto its samples: "
                        "reverseKL(" + div.target_d->canon + " ~ n). The p-term of the field then becomes the KDE score of "
                        "the samples (same kernel on both sides, so the smoothing biases cancel where p\u0302 = q\u0302).");
                for (auto& a : e->args)
                    if (a.kw == "family")
                        err("family= declares the manifold the PATH moves on, so it belongs to descent, not to "
                            "the field extractor: a constrained curve is a different curve (an IVP on the orbit "
                            "{R # from : R in SO(d)}), and its field is the so(d) projection as a THEOREM — write "
                            "descent(D, from=, family=rotation); the estimator stays nw (it estimates the ambient "
                            "field; the projection is exact linear algebra, not estimation).");
                // family=rotation travels with the path (2026-07, ICA line):
                // the constrained curve's velocity IS the so(d) projection of
                // the free descent field — the extractor just inherits it.
                auto fv = make_shared<FieldV>(); fv->kind = FieldV::WG;
                fv->div = p0.wgpath->div; fv->from_prov = p0.wgpath->from_prov;
                fv->wg_from_d = p0.wgpath->from_d; fv->wg_from_x = p0.wgpath->from_x;
                fv->T = p0.wgpath->T;
                fv->nw_norm = div.name == "reverseKL" && p0.wgpath->metric == "w2";
                fv->rotation = p0.wgpath->rotation;
                fv->rot_block = p0.wgpath->rot_block;
                fv->free_lo = p0.wgpath->free_lo;
                fv->free_hi = p0.wgpath->free_hi;
                fv->obs = p0.wgpath->obs;
                fv->obs_lo = p0.wgpath->obs_lo;
                fv->obs_hi = p0.wgpath->obs_hi;
                if (est == "dsm") {
                    fv->dsm = true; fv->dsm_spec = dsm_spec; fv->dsm_sigma = dsm_sigma;
                    fv->dsm_warm = dsm_warm; fv->dsm_pre = dsm_pre;
                    fv->dsm_trainlr = dsm_trainlr; fv->dsm_batch = dsm_batch;
                }
                fv->desc = "field(" + p0.wgpath->canon +
                           (div.name == "w2" ? ",sinkhorn)" : est == "dsm" ? ",dsm)" : ",nw)");
                Value r; r.k = Value::Field; r.field = fv; return r;
            }
            // —— declared path: regress the conditional-expectation field ——
            // regress(net, base=v) freezes a pretrained field and regresses
            // only the correction (spec 10.3.2); the result is the genuine
            // field of THIS path, represented as 1*[base] + 1*[residual].
            // regress(rotation) restricts the HYPOTHESIS CLASS to skew-linear
            // fields {x -> Omega(t) x} — a declared path cannot be re-curved
            // (contrast family= on descent, which changes the IVP), but its
            // velocity can be approximated within a class, and for this class
            // the least squares is CLOSED FORM (no net, no SGD; the map is
            // exactly a rotation and inv is free).
            shared_ptr<NetSpec> spec;
            shared_ptr<NetSpec> rot_net;
            shared_ptr<FieldV> base;
            bool rot_est = false; int rot_blk = 1;
            int steps = 6000; float lr = 1e-3f; int batch = 128;
            for (auto& a : e->args) {
                if (a.kw == "estimator") {
                    if (a.e->k != Expr::Call || a.e->id != "regress")
                        err("field over a declared path supports estimator=regress(mlp(...)) or regress(rotation)");
                    for (auto& ra : a.e->args) {
                        if (ra.kw.empty()) {
                            if (ra.e->k == Expr::Call && ra.e->id == "rotation") {
                                // rotation(block=L): the class tightens to the
                                // block-Kronecker rotations {(Omega (x) I_L) x}
                                // — process-level constrained transport.
                                // rotation(net=mlp): the neural member — the
                                // conditional field is trained by the net, the
                                // map is its per-slice so(d) projection
                                rot_est = true;
                                for (auto& rra : ra.e->args) {
                                    if (rra.kw == "block") rot_blk = (int)eval(rra.e).num;
                                    else if (rra.kw == "net") {
                                        Value nv2 = eval(rra.e);
                                        if (nv2.k != Value::Net)
                                            err("rotation(net=...) takes an mlp — the conditional field estimator "
                                                "whose element-Jacobian skew becomes the rotation");
                                        rot_net = nv2.net;
                                    } else
                                        err("regress(rotation(...)) takes block= (lag count, closed form) or "
                                            "net= (neural conditional field, so(d)-projected)");
                                }
                                if (rot_blk < 1) err("regress(rotation(block=L)): L must be a positive lag count");
                                if (rot_net && rot_blk > 1)
                                    err("rotation(net=...) does not combine with block= — the net's contexts "
                                        "already carry the lags");
                                continue;
                            }
                            Value nv = eval(ra.e);
                            if (nv.k == Value::Symbol && nv.sym == "rotation") { rot_est = true; continue; }
                            if (nv.k != Value::Net) err("regress takes an mlp or the hypothesis-class name rotation");
                            spec = nv.net;
                        } else if (ra.kw == "base") {
                            Value bv = eval(ra.e);
                            if (bv.k != Value::Field)
                                err("base= expects a Field (a pretrained field whose correction this regression fits)");
                            base = bv.field;
                        } else err("regress: unknown argument " + ra.kw + "=  (takes a net and optionally base=)");
                    }
                }
                if (a.kw == "steps") steps = (int)eval(a.e).num;
                if (a.kw == "lr")    lr = (float)eval(a.e).num;
                if (a.kw == "batch") batch = (int)eval(a.e).num;
            }
            if (!spec && !rot_est) err("field requires estimator=regress(mlp(...)) or regress(rotation)");
            if (spec && rot_est) err("regress takes ONE hypothesis class: an mlp or rotation, not both");
            string sp;
            {   // the declared path's t=0 marginal: the terms surviving at t=0
                ostringstream o; bool first = true;
                for (auto& tm : p0.path->rv->terms) {
                    float c0 = tm.coeff(0.0f);
                    if (fabs(c0) < 1e-6f) continue;
                    if (!first) o << " + ";
                    if (fabs(c0 - 1.0f) > 1e-6f) o << c0 << "*";
                    o << tm.srcname();
                    first = false;
                }
                sp = o.str();
            }
            int cond_total = 0;
            for (auto& tm : p0.path->rv->terms) cond_total += tm.cdim();
            if (base && cond_total > 0)
                err("base= with conditional paths is not yet supported");
            if (rot_est) {
                if (base) err("regress(rotation) admits no base= — the skew-linear class is closed form, "
                              "there is no residual to train");
                // conditional paths welcome (2026-07, SICA line): the indices
                // join the closed-form fit as covariates; the field keeps only
                // the skew x-part — self-contained (see fit_rot_field)
                if (cond_total > 0 && rot_blk > 1)
                    err("regress(rotation(block=L)) does not combine with a conditional path — the "
                        "conditional fit works at the element level, and its contexts already carry "
                        "the lags; use plain regress(rotation)");
                int d = p0.path->rv->terms.at(0).dim();
                if (d % rot_blk != 0)
                    err("regress(rotation(block=" + to_string(rot_blk) + ")) needs the path dimension (" +
                        to_string(d) + ") to be a multiple of the lag count L — window(X, L) produces channels*L rows");
                int rbatch = batch == 128 ? 384 : batch;      // default 384 per slice unless batch= given
                auto fv = make_shared<FieldV>(); fv->kind = FieldV::ROTFM;
                if (rot_net) {
                    // neural conditional-rotation (2026-07, SICA line): the
                    // contexts enter the NET (nonlinear), not as linear
                    // features; the trained field is then projected onto
                    // so(d) per slice (average element-Jacobian, skew part —
                    // see project_rot_net). The returned field is the same
                    // self-contained skew-linear object as the closed form.
                    if (rot_net->dims.front() != d + cond_total || rot_net->dims.back() != d)
                        err("rotation(net=...): expected network dims " + to_string(d + cond_total) +
                            " -> ... -> " + to_string(d) + " (element dim + condition rows in, element dim out)");
                    auto tn = train_field(*p0.path->rv, *rot_net, steps, batch, lr);
                    fv->rot_knots = project_rot_net(*p0.path->rv, *tn, 33, rbatch);
                } else
                    fv->rot_knots = fit_rot_field(*p0.path->rv, 33, rbatch, rot_blk);
                fv->dim = d; fv->ydim = d;
                fv->start_prov = sp;
                fv->desc = "field(" + p0.path->canon + ",regress(rotation" +
                           (rot_net ? "(net))" : rot_blk > 1 ? "(block=" + to_string(rot_blk) + "))" : "))");
                Value r; r.k = Value::Field; r.field = fv; return r;
            }
            if (base) {
                if (base->kind == FieldV::WG)
                    err("base= requires a self-contained field — a descent field is re-estimated from the "
                        "evolving ensemble each step and has no standalone evaluation (spec 10.3.1). On the "
                        "descent side, combine upstream at the divergence level: descent(a*D1 + b*D2).");
                int d = p0.path->rv->terms.at(0).dim();
                if (base->dim != d)
                    err("base= field has dimension " + to_string(base->dim) + " but this path lives in dimension " + to_string(d));
                if (!base->start_prov.empty() && base->start_prov != sp)
                    err("the base field transports " + base->start_prov + " but this path starts from " + sp +
                        ": a residual over a foreign base is not the field of this path — declare both "
                        "interpolation formulas from the same source (spec 10.3.2).");
            }
            auto tn = train_field(*p0.path->rv, *spec, steps, batch, lr, base);
            auto fv = make_shared<FieldV>(); fv->kind = FieldV::FM; fv->net = tn;
            fv->dim = tn->d_out;
            fv->ydim = tn->d_out; fv->cond_dim = cond_total;
            if (base) {
                // Δ is a correction, not the transport field of any path: no
                // start_prov of its own; the combination inherits the base's.
                fv->desc = "regress-residual(" + p0.path->canon + ")";
                return field_lincomb({{1.0f, base}, {1.0f, fv}});
            }
            fv->desc = "field(" + p0.path->canon + ")";
            fv->start_prov = sp;
            Value r; r.k = Value::Field; r.field = fv; return r;
        }

        if (f == "flow") {
            if (e->args.empty()) err("flow(v, steps=) requires a field");
            Value v0 = eval(e->args[0].e);
            if (v0.k == Value::WGPath || v0.k == Value::Path)
                err("flow integrates a field, not a path — extract the field with field(...) first");
            if (v0.k != Value::Field) err("the first argument of flow must be a Field");
            auto m = make_shared<MapV>(); m->f = v0.field;
            Value st = arg_kw(e->args, "steps"), lr = arg_kw(e->args, "lr");
            Value rc = arg_kw(e->args, "record");
            m->steps = isnan(st.num) ? 50 : (int)st.num;
            if (!isnan(lr.num)) m->lr = (float)lr.num;
            if (rc.k == Value::Symbol && rc.sym == "true") m->record = true;
            if (v0.field->kind == FieldV::WG && v0.field->free_lo >= 0 && m->record)
                err("record= replays the whole ensemble's trajectory, but a conditional descent (from=(y|x)) "
                    "moves only the free rows of a conditioned ensemble — the recorded history would replay a "
                    "joint that was never a joint. Not supported; re-simulate from the pinned start instead "
                    "(descents are cheap).");
            if (v0.field->kind == FieldV::WG && v0.field->obs && m->record)
                err("record= freezes per-step ensembles for pointwise NW replay, but the N-observation field "
                    "(from=(q|Obs)) drives its attraction through likelihood-weighted library points "
                    "(spec 10.10), which the replay machinery does not reconstruct. Not supported; "
                    "re-simulate from the conditioned start instead (descents are cheap).");
            Value pj = arg_kw(e->args, "project");
            if (pj.k == Value::Symbol) {
                if (pj.sym != "instant")
                    err("project= takes instant — Rao-Blackwell each step's conditional field onto the "
                        "elements, so every step is one shared instantaneous map (the identifiability "
                        "theorem's class; SICA line)");
                if (v0.field->kind != FieldV::WG || v0.field->free_lo < 0 || !v0.field->wg_from_x)
                    err("project=instant averages the pinned contexts out of a CONDITIONAL descent field — "
                        "it needs from=(E | C), a frozen conditioned ensemble (contexts are what get "
                        "Rao-Blackwelled away; an unconditional field is already context-free)");
                if (v0.field->obs)
                    err("project=instant is for the per-particle-pinned mode from=(E | C); the N-observation "
                        "engine (q | Obs) moves one shared ensemble and has no per-column contexts to average");
                if (v0.field->rotation)
                    err("family=rotation already constrains the map (linear instantaneous); project=instant "
                        "is the nonlinear-instantaneous alternative — pick one");
                m->proj_instant = true;
            }
            // —— amortized flow (spec 10.6): into= replaces the particle
            //    simulator with an optimizer; same field, third representation
            //    (simulate / record / amortize). WG fields only: self-contained
            //    fields are already one network, and their flow needs no
            //    training (one-step distillation of trained flows is a
            //    different construction — consistency models — not this slot).
            for (auto& a : e->args) {
                if (a.kw != "into") continue;
                Value nv = eval(a.e);
                if (nv.k != Value::Net) err("into= expects an mlp — the one-step generator that will carry the flow");
                if (v0.field->kind != FieldV::WG)
                    err("into= amortizes a descent simulation, and " + v0.field->desc +
                        " is already self-contained — its flow costs no training to apply. "
                        "(Distilling a trained flow into one step is a different construction, not implemented.)");
                if (m->record)
                    err("record=true and into= are two different ways to buy back a descent map — memory records the "
                        "particle trajectory, into= amortizes it into weights (which keeps no trajectory to record). Pick one.");
                if (v0.field->rotation)
                    err("into= amortizes a free-space descent into a one-step generator net; a rotation flow is "
                        "already finite-dimensional — its map is one d×d orthogonal matrix, nothing to amortize.");
                if (v0.field->free_lo >= 0) {
                    // Level 1 of the instantaneous demixer (SICA line): a
                    // conditional descent whose ensemble carries WINDOW
                    // provenance amortizes into an instantaneous net — the
                    // outer iteration loop dissolves into the optimizer
                    // (train_amortized_demixer). Everything else keeps the
                    // original exclusion.
                    auto& fx = v0.field->wg_from_x;
                    if (!fx || !fx->wsrc || fx->wlen < 2)
                        err("into= trains a generator of the FULL ensemble law, but a conditional descent (from=(y|x)) "
                            "holds the observed rows fixed — the object being transported is a conditional slice, not "
                            "a joint. Amortizing over observations is the regression route: the conditional path "
                            "formula of sbi_npse.liu. (Exception: an ensemble built by lagsplit(window(Z, L), L) "
                            "carries its trajectory provenance, and into= then trains the amortized INSTANTANEOUS "
                            "demixer — contexts in the objective, the net a function of the elements alone.)");
                    if (!v0.field->dsm)
                        err("the amortized demixer re-estimates the conditional field at every optimizer step from "
                            "the net's own output; the nw estimator measurably cannot carry the conditional score "
                            "on the window joint (ledger: flat at the floor) — use estimator=dsm(...)");
                    if (v0.field->free_lo != 0 || v0.field->free_hi != fx->blk)
                        err("the amortized demixer moves the ELEMENT block of a lagsplit ensemble — from= must be "
                            "(E | C) with E the element rows [0, d)");
                    if (!v0.field->div->target_x ||
                        v0.field->div->target_x->prov.rfind("lagsplit(decouple(window(", 0) != 0)
                        err("the amortized demixer rebuilds its target from the net's own output at every step — "
                            "the declared target must be the per-channel permuted product, "
                            "lagsplit(decouple(window(Z, L), block=L) ~ N, L), so the rebuild has a recipe to follow "
                            "(canon strings are identity)");
                    int dch = fx->blk, djoint = dch * fx->wlen;
                    if (v0.field->dsm_spec->dims.front() != djoint || v0.field->dsm_spec->dims.back() != djoint)
                        err("dsm: expected network dims " + to_string(djoint) + " -> ... -> " + to_string(djoint) +
                            " (the score nets live on the window joint)");
                    if (nv.net->dims.front() != dch || nv.net->dims.back() != dch)
                        err("expected network dims " + to_string(dch) + " -> ... -> " + to_string(dch) +
                            " — the demixer is an instantaneous map of the " + to_string(dch) +
                            " channels (contexts enter the objective, never the net; that is the "
                            "identifiability theorem's class)");
                    Value bt = arg_kw(e->args, "batch"), tl = arg_kw(e->args, "trainlr");
                    int batch = isnan(bt.num) ? 256 : (int)bt.num;
                    float trainlr = isnan(tl.num) ? 1e-3f : (float)tl.num;
                    int tsteps = isnan(st.num) ? 2000 : (int)st.num;
                    auto tn = train_amortized_demixer(*v0.field, *nv.net, tsteps, batch, m->lr, trainlr);
                    auto fv = make_shared<FieldV>(); fv->kind = FieldV::AMORT; fv->residual = true;
                    fv->net = tn; fv->dim = dch; fv->from_prov = fx->wsrc->prov;
                    fv->desc = "amortized-instant(" + v0.field->desc + ")";
                    auto am = make_shared<MapV>(); am->f = fv; am->steps = 1;
                    am->invertible = false;
                    am->desc = "flow(" + v0.field->desc + ",into=mlp,steps=" + to_string(tsteps) + ")";
                    Value r; r.k = Value::Map; r.map = am; return r;
                }
                if (v0.field->obs)
                    err("into= amortizes ONE descent into weights, but the N-observation field (from=(q|Obs)) is "
                        "tied to its specific observation set — a net trained on it speaks for those observations "
                        "only. Amortizing over observations is the regression route: the conditional path formula "
                        "of sbi_npse.liu.");
                if (v0.field->wg_from_d && !v0.field->wg_from_d->samplable)
                    err("into= trains on latent batches drawn from from= (" + v0.field->from_prov +
                        "), which admits no exact sampler — start the descent from a samplable measure or a Dataset.");
                int d = v0.field->dim;
                if (nv.net->dims.front() != d || nv.net->dims.back() != d)
                    err("expected network dims " + to_string(d) + " -> ... -> " + to_string(d) +
                        " (a one-step generator maps latents to samples; no time input — the evolution lives in training, not in the net)");
                Value bt = arg_kw(e->args, "batch"), tl = arg_kw(e->args, "trainlr");
                int batch = isnan(bt.num) ? 128 : (int)bt.num;
                float trainlr = isnan(tl.num) ? 1e-3f : (float)tl.num;
                int tsteps = isnan(st.num) ? 2000 : (int)st.num;   // optimizer steps: default 2000, not flow's 50
                auto tn = train_amortized(*v0.field, *nv.net, tsteps, batch, m->lr, trainlr);
                auto fv = make_shared<FieldV>(); fv->kind = FieldV::AMORT;
                fv->net = tn; fv->dim = d; fv->from_prov = v0.field->from_prov;
                fv->desc = "amortized(" + v0.field->desc + ")";
                auto am = make_shared<MapV>(); am->f = fv; am->steps = 1;
                am->invertible = false;          // 一步生成器没有可倒放的轨迹(价目表新行)
                am->desc = "flow(" + v0.field->desc + ",into=mlp,steps=" + to_string(tsteps) + ")";
                Value r; r.k = Value::Map; r.map = am; return r;
            }
            if (v0.field->kind == FieldV::WG) {
                if (m->record && v0.field->rotation)
                    err("record=true stores per-step ensembles so the NW field can be replayed pointwise, but a "
                        "rotation flow needs none of that: the whole map is the product of its per-step rotations — "
                        "a single d×d orthogonal matrix. Storing that composite (and its free inverse, the transpose) "
                        "is not implemented in this prototype; apply the map forward without record=.");
                if (m->record && v0.field->div && v0.field->div->name == "w2")
                    err("record=true for a w2 descent is not implemented: the plan is re-solved each step and its "
                        "displacement fields are not yet stored for replay — the recorded-map machinery currently "
                        "reconstructs NW fields only (reverseKL / mmd descents).");
                m->invertible = m->record;       // 反演用内存买:录史后场逐点可重建
                m->endpoint_prov = make_shared<string>();
            }
            m->desc = "flow(" + v0.field->desc + ",steps=" + to_string(m->steps) +
                      (m->proj_instant ? ",project=instant" : "") + ")";
            Value r; r.k = Value::Map; r.map = m; return r;
        }
        if (f == "inv") {
            Value v0 = eval(e->args.at(0).e);
            if (v0.k != Value::Map) err("inv takes a Map");
            if (v0.map->f->kind == FieldV::AMORT)
                err("a one-step generator has no pointwise inverse: into= amortized the whole Euler chain into the "
                    "weights, so there is no trajectory to run backwards and the net need not be injective. If you "
                    "need the inverse, buy it another way: simulate the same descent with flow(v, record=true) "
                    "(memory), or train a self-contained field with reverse(descent, estimator=denoiser(net)).");
            if (!v0.map->invertible)
                err("this Map integrates a gradient-flow field; without a recorded history it admits no pointwise inverse. Three price tiers: "
                    "inv is free for self-contained fields; flow(..., record=true) makes inv legal by recording the per-step ensembles "
                    "(the NW field is then evaluable at arbitrary points — inversion bought with memory, conditioned by dissipation); "
                    "or buy a self-contained field with training: reverse(descent, estimator=denoiser(net)).");
            auto m = make_shared<MapV>(*v0.map);
            m->inverse = !m->inverse;
            m->desc = "inv(" + v0.map->desc + ")";
            Value r; r.k = Value::Map; r.map = m; return r;
        }
        if (f == "decouple") {
            // the dual of couple(..., via=independent), any dimension: take a
            // Dataset's empirical (block) marginals and return their PRODUCT
            // measure. block=L groups rows into blocks of L that stay JOINT —
            // on window()ed data this is process independence: each channel's
            // own dynamics survive, only cross-channel dependence is forgotten.
            Value t = eval(e->args.at(0).e);
            if (t.k == Value::DistV)
                err("decouple forgets the dependence of an EMPIRICAL joint — its argument is a Dataset. " +
                    t.dist->canon + (t.dist->samplable ? " has a sampler: decouple(" + t.dist->canon + " ~ n)."
                                                       : " has no exact sampler; produce samples upstream first."));
            if (t.k != Value::Data) err("decouple takes a Dataset (an empirical joint measure)");
            int blk = 1;
            for (auto& a : e->args)
                if (a.kw == "block") blk = (int)eval(a.e).num;
            int d = (int)t.data->X.num_row();
            if (blk < 1 || d % blk != 0)
                err("decouple(X, block=L) groups the " + to_string(d) + " rows into blocks of L that stay "
                    "jointly bootstrapped — L must divide the row count (got L = " + to_string(blk) +
                    "). For window()ed data, L is the window length.");
            auto dv = make_shared<Dist>();
            Dist::Comp c; c.kind = "decouple"; dv->comps.push_back(c);
            dv->dcp_src = t.data; dv->dcp_block = blk;
            dv->dim = d;
            dv->has_score = false; dv->samplable = true; dv->has_cond_sampler = false;
            dv->canon = "decouple(" + t.data->prov + (blk > 1 ? ",block=" + to_string(blk) : "") + ")";
            Value r; r.k = Value::DistV; r.dist = dv; return r;
        }
        if (f == "window") {
            // delay embedding: reshape a trajectory Dataset (d channels x n
            // time points, columns in time order) into the cloud of sliding
            // windows — (d*L) x (n-L+1), CHANNEL-MAJOR rows (row c*L + lag =
            // channel c at t+lag), so decouple(block=L) and
            // family=rotation(block=L) see one block per channel.
            // Deterministic, no RNG. Process-level statements about a time
            // series are cloud-level statements about its window embedding.
            Value t = eval(e->args.at(0).e);
            if (t.k != Value::Data)
                err("window(X, L) reshapes a trajectory Dataset into its sliding-window cloud — sample the "
                    "trajectory first (~ n draws ONE trajectory for time-series joints like mixed_signals)");
            if (e->args.size() < 2) err("window(X, L) needs the window length L");
            int L = (int)eval(e->args.at(1).e).num;
            const Mat& X = t.data->X;
            int d = (int)X.num_row(), n = (int)X.num_col();
            if (L < 1 || L > n)
                err("window length L = " + to_string(L) + " must lie in [1, " + to_string(n) +
                    "] (the trajectory has " + to_string(n) + " time points)");
            int m = n - L + 1;
            Mat W("win", d * L, m);
            for (int c = 0; c < d; c++)
                for (int l = 0; l < L; l++)
                    for (int i = 0; i < m; i++)
                        W.elem(c * L + l, i) = X.elem(c, i + l);
            Value r; r.k = Value::Data;
            r.data = make_shared<Dataset>(move(W), "window(" + t.data->prov + "," + to_string(L) + ")");
            r.data->wsrc = t.data; r.data->wlen = L;
            return r;
        }
        if (f == "unwindow") {
            // the inverse of window: take the lag-0 row of each channel block,
            // (d*L) x m -> d x m — the channel trajectories, read back off the
            // window cloud. After a window-space flow (rotation(block=L) maps
            // are (Omega x I_L), so windows stay consistent windows of ONE
            // rotated series) this recovers the per-channel signal without
            // plotting L shifted copies of it. Deterministic, no RNG.
            Value t = eval(e->args.at(0).e);
            if (t.k != Value::Data)
                err("unwindow(Z, L) reads channel trajectories back off a window-cloud Dataset — apply it to "
                    "the output of a window-space flow");
            if (e->args.size() < 2) err("unwindow(Z, L) needs the window length L used by window(X, L)");
            int L = (int)eval(e->args.at(1).e).num;
            const Mat& Z = t.data->X;
            int dL = (int)Z.num_row(), m = (int)Z.num_col();
            if (L < 1 || dL % L != 0)
                err("unwindow(Z, L): the row count " + to_string(dL) + " is not channels*L for L = " +
                    to_string(L) + " — use the same L as the window(X, L) that built this cloud");
            int d = dL / L;
            Mat Y("unwin", d, m);
            for (int c = 0; c < d; c++)
                for (int i = 0; i < m; i++) Y.elem(c, i) = Z.elem(c * L, i);
            Value r; r.k = Value::Data;
            r.data = make_shared<Dataset>(move(Y), "unwindow(" + t.data->prov + "," + to_string(L) + ")");
            return r;
        }
        if (f == "lagsplit") {
            // reorder a channel-major window cloud into [elements; contexts]
            // (2026-07, SICA line): the lag-0 row of each channel becomes the
            // ELEMENT block (rows [0, d)), the remaining lags stack lag-major
            // below (row d + (k-1)*d + i = channel i at lag k). The result
            // carries the block marker (Dataset::blk = d), so it destructures:
            // (E, C) = lagsplit(W, L) — element and context as formula-ready
            // block views of ONE table. This is the window layout Algorithm-1
            // style per-element conditioning needs (the formula moves E, the
            // law gate conditions on C); decouple(block=L) keeps operating on
            // the channel-major original. Pure row permutation, no RNG.
            Value t = eval(e->args.at(0).e);
            if (t.k != Value::Data)
                err("lagsplit(W, L) reorders a window-cloud Dataset (from window(X, L)) into "
                    "[elements; contexts] blocks — apply it to a Dataset");
            if (e->args.size() < 2) err("lagsplit(W, L) needs the window length L used by window(X, L)");
            int L = (int)eval(e->args.at(1).e).num;
            const Mat& Z = t.data->X;
            int dL = (int)Z.num_row(), m = (int)Z.num_col();
            if (L < 2 || dL % L != 0)
                err("lagsplit(W, L): the row count " + to_string(dL) + " is not channels*L for L = " +
                    to_string(L) + " (L must be at least 2 — with L = 1 there is no context to split off)");
            int d = dL / L;
            Mat Y("lagsplit", dL, m);
            for (int i = 0; i < d; i++)
                for (int c2 = 0; c2 < m; c2++) Y.elem(i, c2) = Z.elem(i * L, c2);
            for (int k = 1; k < L; k++)
                for (int i = 0; i < d; i++)
                    for (int c2 = 0; c2 < m; c2++)
                        Y.elem(d + (k - 1) * d + i, c2) = Z.elem(i * L + k, c2);
            Value r; r.k = Value::Data;
            r.data = make_shared<Dataset>(move(Y), "lagsplit(" + t.data->prov + "," + to_string(L) + ")");
            r.data->blk = d;
            r.data->wsrc = t.data->wsrc; r.data->wlen = t.data->wlen;
            return r;
        }
        if (f == "whiten") {
            // affine standardization: Z = Λ^{-1/2} Uᵀ (X − mean), so cov(Z) = I
            // exactly (PCA whitening). Deterministic — sample statistics only,
            // no RNG. The companion of family=rotation: rotations preserve the
            // covariance, so whitening first puts every candidate unmixing on
            // ONE orbit O(d)·Z and the rotation flow searches exactly that.
            Value t = eval(e->args.at(0).e);
            if (t.k == Value::DistV)
                err("whiten standardizes an EMPIRICAL ensemble — its mean and covariance are sample statistics, "
                    "not attributes of a law. Sample first: whiten(" + t.dist->canon + " ~ n).");
            if (t.k == Value::RV)
                err("whiten takes a Dataset; for a frozen joint block, take its law first: whiten(prob(x)).");
            if (t.k != Value::Data) err("whiten takes a Dataset (the ensemble to standardize)");
            const Mat& X = t.data->X;
            int d = (int)X.num_row(), n = (int)X.num_col();
            if (n < 2) err("whiten needs at least two samples to estimate a covariance");
            vector<float> mu(d, 0);
            for (int r = 0; r < d; r++) { double s = 0; for (int i = 0; i < n; i++) s += X.elem(r, i); mu[r] = (float)(s / n); }
            vector<vector<double>> C(d, vector<double>(d, 0));
            for (int i = 0; i < n; i++)
                for (int r = 0; r < d; r++) for (int s = r; s < d; s++)
                    C[r][s] += (double)(X.elem(r, i) - mu[r]) * (X.elem(s, i) - mu[s]);
            for (int r = 0; r < d; r++) for (int s = r; s < d; s++) { C[r][s] /= n; C[s][r] = C[r][s]; }
            // cyclic Jacobi eigendecomposition (deterministic; d is tiny)
            vector<vector<double>> V(d, vector<double>(d, 0));
            for (int r = 0; r < d; r++) V[r][r] = 1;
            for (int sweep = 0; sweep < 64; sweep++) {
                double off = 0;
                for (int p = 0; p < d; p++) for (int q = p + 1; q < d; q++) off += C[p][q] * C[p][q];
                if (off < 1e-18) break;
                for (int p = 0; p < d; p++) for (int q = p + 1; q < d; q++) {
                    if (fabs(C[p][q]) < 1e-15) continue;
                    double th = 0.5 * atan2(2 * C[p][q], C[q][q] - C[p][p]);
                    double cs = cos(th), sn = sin(th);
                    for (int k = 0; k < d; k++) {
                        double a = C[p][k], b = C[q][k];
                        C[p][k] = cs * a - sn * b; C[q][k] = sn * a + cs * b;
                    }
                    for (int k = 0; k < d; k++) {
                        double a = C[k][p], b = C[k][q];
                        C[k][p] = cs * a - sn * b; C[k][q] = sn * a + cs * b;
                        a = V[k][p]; b = V[k][q];
                        V[k][p] = cs * a - sn * b; V[k][q] = sn * a + cs * b;
                    }
                }
            }
            // canonical eigenvector signs (largest-|entry| positive): the output
            // must not depend on Jacobi's internal rotation choices
            for (int c = 0; c < d; c++) {
                int am = 0;
                for (int r = 1; r < d; r++) if (fabs(V[r][c]) > fabs(V[am][c])) am = r;
                if (V[am][c] < 0) for (int r = 0; r < d; r++) V[r][c] = -V[r][c];
            }
            for (int c = 0; c < d; c++)
                if (C[c][c] < 1e-10)
                    err("whiten: this ensemble is degenerate along a direction (an eigenvalue of the covariance "
                        "is ~0) — the whitening transform Λ^{-1/2} does not exist; the data lives on a lower-"
                        "dimensional subspace.");
            Mat Z("Z", d, n);
            for (int i = 0; i < n; i++)
                for (int c = 0; c < d; c++) {
                    double s = 0;
                    for (int r = 0; r < d; r++) s += V[r][c] * (X.elem(r, i) - mu[r]);
                    Z.elem(c, i) = (float)(s / sqrt(C[c][c]));
                }
            Value r; r.k = Value::Data;
            r.data = make_shared<Dataset>(move(Z), "whiten(" + t.data->prov + ")");
            return r;
        }
        if (f == "coupling")
            err("the verb is couple(A, B, via=...) — constructors in this language are verbs (descent, reverse, flow, couple)");
        if (f == "couple") {
            // couple (spec 10.3): a joint law over PAIRS whose marginals are A and
            // B; a draw is a batch paired within itself (minibatch coupling).
            auto dv = make_shared<Dist>();
            Dist::Comp c; c.kind = "couple"; dv->comps.push_back(c);
            int got = 0;
            string via = "independent"; float eps = 0.1f;
            string sides[2];
            for (auto& a : e->args) {
                if (a.kw == "via") {
                    if (a.e->k == Expr::Call && a.e->id == "sinkhorn") {
                        via = "sinkhorn";
                        if (!a.e->args.empty()) eps = (float)eval(a.e->args[0].e).num;
                    } else {
                        Value vv = eval(a.e);
                        if (vv.k != Value::Symbol || (vv.sym != "ot" && vv.sym != "independent" && vv.sym != "paired"))
                            err("couple: via= names the pairing plan — independent (keep the draw order), "
                                "ot (exact assignment within the batch), sinkhorn(eps) (entropic plan), or "
                                "paired (the second marginal is the FIRST one pushed through a map: reflow pairs)");
                        via = vv.sym;
                    }
                    continue;
                }
                if (!a.kw.empty()) err("couple: unknown argument " + a.kw + "=  (takes two measures and via=)");
                Value vv = eval(a.e);
                if (vv.k == Value::RV)
                    err("the pair identity lives on the couple: a coupling IS one joint draw, so its marginals "
                        "cannot carry their own rv() identity — write (x0, x1) = rv(couple(A, B, via=...)), "
                        "not couple(rv(A), ...).");
                shared_ptr<Dist> sd; shared_ptr<Dataset> sx; shared_ptr<MapV> smap; int dim_;
                string sname;
                if (vv.k == Value::DistV) {
                    if (!vv.dist->samplable)
                        err("couple: " + vv.dist->canon + " admits no exact sampler — a coupling draws from both "
                            "marginals (sample it into a Dataset first, or use via svgd upstream)");
                    if (vv.dist->dy > 0) err("couple: " + vv.dist->canon + " is itself a joint — couple takes plain marginals");
                    sd = vv.dist; dim_ = vv.dist->dim; sname = vv.dist->canon;
                } else if (vv.k == Value::Data) { sx = vv.data; dim_ = (int)vv.data->X.num_row(); sname = vv.data->prov; }
                else if (vv.k == Value::Pushed) {
                    if (got == 0) err("couple: a pushforward marginal goes in the SECOND slot — "
                                      "couple(base, T # base, via=paired) reads as \"pair each z with T(z)\"");
                    sd = vv.pushed->base; smap = vv.pushed->map;
                    dim_ = vv.pushed->base->dim; sname = vv.pushed->canon;
                }
                else err("couple: the two slots take a Distribution, a Dataset, or (second slot) a pushforward T # mu");
                if (got == 0) { dv->cpl_a = sd; dv->cpl_ax = sx; dv->dy = dim_; }
                else if (got == 1) { dv->cpl_b = sd; dv->cpl_bx = sx; dv->cpl_map = smap; dv->dx = dim_; }
                else err("couple takes exactly two measures");
                if (got <= 1) sides[got] = sname;
                got++;
            }
            if (got != 2) err("couple takes exactly two measures: couple(A, B, via=independent/ot/sinkhorn(eps)/paired)");
            if (via == "paired") {
                if (!dv->cpl_map)
                    err("via=paired pairs each z with its OWN image: the second marginal must be a pushforward "
                        "of the first — couple(base, T # base, via=paired) (reflow, spec 10.1)");
                string a = dv->cpl_a ? dv->cpl_a->canon : dv->cpl_ax->prov;
                if (!dv->cpl_b || dv->cpl_b->canon != a)
                    err("via=paired: the pushforward's base (" + (dv->cpl_b ? dv->cpl_b->canon : string("?")) +
                        ") is not the first marginal (" + a + ") — (z, T(z)) pairs are only defined when T is "
                        "applied to the SAME measure the z's are drawn from");
            } else if (dv->cpl_map)
                err("a pushforward marginal implies via=paired ((z, T(z)) pairs) — with " + via +
                    " the pushforward's draws would be independent of the first marginal's, which is just an "
                    "ordinary coupling: sample it into a Dataset first if that is what you mean");
            if (via != "independent" && via != "paired" && dv->dy != dv->dx)
                err("via=" + via + " pairs by distance, which needs both marginals in the SAME space — got dimensions " +
                    to_string(dv->dy) + " and " + to_string(dv->dx) + " (via=independent has no such constraint)");
            dv->cpl_via = via; dv->cpl_eps = eps;
            dv->dim = dv->dy + dv->dx;
            dv->pair_blocks = true; dv->has_score = false; dv->has_cond_sampler = false;
            dv->samplable = true;
            {
                ostringstream o; o << "couple(" << sides[0] << "," << sides[1] << ",via=" << via;
                if (via == "sinkhorn") o << "(" << eps << ")";
                o << ")";
                dv->canon = o.str();
            }
            Value r; r.k = Value::DistV; r.dist = dv; return r;
        }
        if (f == "reverseKL" || f == "forwardKL" || f == "mmd" || f == "w2") {
            if (f == "forwardKL")
                err("forwardKL needs an explicit density-ratio estimate (the h-transform of f-divergences, "
                    "Liu et al. 2024 Thm 2.1) — the locallinear estimator family is not in this prototype yet. "
                    "Available descents: reverseKL (score / KDE), mmd (witness gradient), w2 (entropic plan).");
            auto dv = make_shared<DivV>(); dv->name = f;
            Value t = eval(e->args.at(0).e);
            if (f == "mmd" || f == "w2") {
                // sample-comparison divergences: the target enters through its
                // samples only — a score is of no use to either functional
                if (t.k == Value::DistV)
                    err(f + " compares SAMPLES — its target must be a Dataset. " + t.dist->canon +
                        (t.dist->samplable ? " has a sampler: write " + f + "(" + t.dist->canon + " ~ n)."
                                           : " admits no exact sampler; produce samples upstream (via svgd) first."));
                if (t.k != Value::Data) err(f + " : the target slot must be a Dataset (an empirical measure)");
                dv->target_x = t.data; dv->canon = f + "(" + t.data->prov;
                if (f == "w2")
                    for (auto& a : e->args)
                        if (a.kw == "eps") { dv->w2_eps = (float)eval(a.e).num; dv->canon += ",eps=" + to_string(dv->w2_eps).substr(0, 5); }
            } else {
                if (t.k == Value::DistV)      { dv->target_d = t.dist; dv->canon = f + "(" + t.dist->canon; }
                else if (t.k == Value::Data)  { dv->target_x = t.data; dv->canon = f + "(" + t.data->prov; }
                else err(f + " : the target slot must be a Distribution or a Dataset");
            }
            if (e->args.size() > 1 && e->args[1].kw.empty()) {
                Value q = eval(e->args[1].e);
                if (q.k == Value::DistV)     dv->moving_prov = q.dist->canon;
                else if (q.k == Value::Data) dv->moving_prov = q.data->prov;
                else err(f + " : the moving slot must be a Distribution or a Dataset");
                dv->canon += "," + dv->moving_prov;
            }
            dv->canon += ")";
            Value r; r.k = Value::Div; r.div = dv; return r;
        }
        if (f == "WGpath")
            err("WGpath has been renamed to descent(D, from=q0, time=, metric=w2) — steepest descent is defined only relative to a geometry, "
                "so the metric is an explicit slot (default w2; under metric=stein the nw update is the exact gradient flow, under w2 a kernel-smoothed estimate of it)");
        if (f == "WGField" || f == "WGFlow")
            err(f + "  is retired (spec 10.2.1): a divergence induces a path, not a field. Write\n"
                "  qt = descent(D, from=q0, time=..., metric=w2)\n"
                "  v  = field(qt, estimator=nw(kernel=rbf))\n"
                "  X  = flow(v, steps=..., lr=...) # (samples of q0)");
        if (f == "reverse") {
            Value g0 = eval(e->args.at(0).e);
            if (g0.k == Value::Map)
                err("reverse applies to implicit paths (descent); the inverse of a Map is free — use inv(T).");
            if (g0.k == Value::Path)
                err("declared paths need no reverse: time reversal is the substitution t -> 1-t in the formula itself, "
                    "then field(prob(...)). reverse is reserved for implicit (descent) paths.");
            if (g0.k != Value::WGPath) err("the first argument of reverse must be a descent path (implicit)");
            shared_ptr<NetSpec> spec;
            int steps = 4000; float lr = 1e-3f; int batch = 128;
            for (auto& a : e->args) {
                if (a.kw == "estimator") {
                    if (a.e->k != Expr::Call || a.e->id != "denoiser")
                        err("reverse supports estimator=denoiser(net) only in this prototype");
                    Value nv = eval(a.e->args.at(0).e);
                    if (nv.k != Value::Net) err("denoiser(net) expects an mlp");
                    spec = nv.net;
                }
                if (a.kw == "steps") steps = (int)eval(a.e).num;
                if (a.kw == "lr")    lr = (float)eval(a.e).num;
            }
            if (!spec) err("reverse requires estimator=denoiser(mlp(...))");
            auto& wp = *g0.wgpath;
            auto& div = *wp.div;
            if (!div.target_d || div.target_d->comps.size() != 1 || div.target_d->comps[0].kind != "gaussian")
                err("the prototype's forward diffusion requires the target of reverseKL to be a single Gaussian (the stationary law of the reference process); "
                    "for general diffusion use a declared path: xt = sqrt(1 - t*t)*e + t*data");
            if (!wp.from_x)
                err("the initial measure of the forward path must be a Dataset (from=data) — it is what gets noised");
            int d = (int)wp.from_x->X.num_row();
            if (spec->dims.front() != d || spec->dims.back() != d)
                err("the denoiser network must have input/output dimension " + to_string(d));
            auto tn = train_dsm(dataset_sampler(wp.from_x->X), *spec, wp.T, steps, batch, lr);
            auto fv = make_shared<FieldV>(); fv->kind = FieldV::REV; fv->net = tn; fv->T = wp.T;
            fv->dim = tn->d_out;
            fv->desc = "reverse(" + wp.canon + ")";
            Value r; r.k = Value::Field; r.field = fv; return r;
        }
        if (f == "estimate" || f == "mirror" || f == "W2")
            err(f + "  is not implemented in this prototype (see docs/liu-spec.md, sec. 4.5)");
        Value v; v.k = Value::Symbol; v.sym = f; return v;
    }

    // —— transport (legacy sugar over declared paths) ————————————
    Sampler sampler_of(const Value& v, const string& what) {
        if (v.k == Value::DistV) { auto D = v.dist; return [D](int n) { return sample_dist(*D, n); }; }
        if (v.k == Value::Data)  return dataset_sampler(v.data->X);
        err("the " + what + " endpoint of transport must be samplable (a Distribution or a Dataset)");
    }
    static int dim_of(const Value& v) {
        if (v.k == Value::DistV) return v.dist->dim;
        if (v.k == Value::Data) return (int)v.data->X.num_row();
        return -1;
    }

    Value eval_transport(const ExprP& e) {
        Value A = eval(e->t_from), B = eval(e->t_to), N = eval(e->t_using);
        if (N.k != Value::Net) err("transport ... using expects a network skeleton (mlp)");
        int d = dim_of(A);
        if (d != dim_of(B)) err("the two endpoints of transport have different dimensions");
        if (N.net->dims.front() != d || N.net->dims.back() != d)
            err("expected input dim " + to_string(d) + ", got " + to_string(N.net->dims.front()));
        // desugar: field(prob(t*B + (1-t)*A), estimator=regress(net))
        RVal rv;
        RTerm t0; t0.coeff = [](float t) { return 1.0f - t; }; t0.cdesc = "(1-t)";
        t0.src = fresh_src();
        if (A.k == Value::DistV) t0.dist = A.dist; else t0.data = A.data;
        RTerm t1; t1.coeff = [](float t) { return t; }; t1.cdesc = "t";
        t1.src = fresh_src();
        if (B.k == Value::DistV) t1.dist = B.dist; else t1.data = B.data;
        rv.terms = {t0, t1};
        printf("[transport] (sugar) == field(prob(t*to + (1-t)*from), estimator=regress)\n");
        auto tn = train_field(rv, *N.net, e->t_steps, e->t_batch, (float)e->t_lr);
        auto fv = make_shared<FieldV>(); fv->kind = FieldV::FM; fv->net = tn;
        fv->desc = "transport";
        Value r; r.k = Value::Field; r.field = fv; return r;
    }

    // —— sampling ————————————————————————————————————————————————
    // Sample an instantiated kernel's joint (y; condition) as ONE frozen
    // Dataset — used by ~ and by the destructuring bind (y, w) = (K | z) ~ n,
    // which must not re-evaluate the instantiation (it may consume RNG).
    shared_ptr<Dataset> sample_kernel(const KernelV& K, int n) {
        if (K.inst == 0)
            err("a kernel is a family of distributions — instantiate its condition slot with | first "
                "(paired: | the block bound with it; decoupled: | [z], a Distribution, or a Dataset), then sample (spec 10.8)");
        if (K.prog) {
            // programmable kernel (spec 10.10): draw the parameters by
            // instantiation mode, run the body vectorized, return the
            // joint Dataset (y; parameter) — same layout as every
            // bundled joint (observed block first).
            Mat Xc("kx", 0, 0);
            if (K.inst == 2) {
                int dx = (int)K.zfix.size();
                Xc = Mat("kx", dx, n);
                for (int i = 0; i < n; i++)
                    for (int r = 0; r < dx; r++) Xc.elem(r, i) = K.zfix[r];
            } else if (K.inst == 3) {
                Xc = sample_dist(*K.zdist, n);
            } else {
                const Mat& Z = K.zdata->X;
                uniform_int_distribution<int> pick(0, (int)Z.num_col() - 1);
                Xc = Mat("kx", (int)Z.num_row(), n);
                for (int i = 0; i < n; i++) { int j = pick(global_rand_gen);
                    for (size_t r = 0; r < Z.num_row(); r++) Xc.elem(r, i) = Z.elem(r, j); }
            }
            Mat Y = kernel_body_eval(K, Xc);
            auto ds = make_shared<Dataset>(vstack<float>({Y, Xc}), K.canon);
            ds->blk = (int)Y.num_row();          // (y; w) split — destructurable joint
            return ds;
        }
        const Dist& J = *K.joint;
        int dy = J.dy, dx = J.dx;
        Mat Y("ky", dy, n), Xc("kx", dx, n);
        if (K.inst == 1) {                       // paired: joint samples suffice
            Mat Jm = sample_dist(J, n);
            Y = rows_of(Jm, 0, dy); Xc = rows_of(Jm, dy, dy + dx);
        } else if (K.inst == 2) {                // fixed point z
            for (int i = 0; i < n; i++) for (int r = 0; r < dx; r++) Xc.elem(r, i) = K.zfix[r];
            Y = sample_cond(J, Xc);
        } else if (K.inst == 3) {                // z ~ measure
            Xc = sample_dist(*K.zdist, n);
            Y = sample_cond(J, Xc);
        } else {                                 // z from a dataset (with replacement)
            const Mat& Z = K.zdata->X;
            uniform_int_distribution<int> pick(0, (int)Z.num_col() - 1);
            for (int i = 0; i < n; i++) { int j = pick(global_rand_gen);
                for (int r = 0; r < dx; r++) Xc.elem(r, i) = Z.elem(r, j); }
            Y = sample_cond(J, Xc);
        }
        int slots = K.map->f->cond_dim / dx;     // v1: all slots share one value
        Mat C = Xc;
        if (slots > 1) {
            vector<MatrixView<float>> crep;
            for (int s2 = 0; s2 < slots; s2++) crep.push_back(Xc);
            C = vstack<float>(crep);
        }
        vector<Mat> ytr;
        Mat Yout = K.map->apply_cond(Y, C, &ytr);
        auto ds = make_shared<Dataset>(vstack<float>({Yout, Xc}), K.canon);
        ds->blk = dy;                            // (y; z) split — destructurable joint
        for (auto& fr : ytr) ds->traj.push_back(vstack<float>({fr, Xc}));
        return ds;
    }

    Value eval_sample(const ExprP& e) {
        Value base = eval(e->a);
        int n = e->n;
        if (n <= 0 || n > 100000) err("sampling budget exceeded: a Dataset holds at most 100k samples");
        if (e->via) {
            if (e->via->k != Expr::Call || e->via->id != "svgd")
                err("via supports svgd(kernel=rbf, steps=, lr=) only in this prototype");
            const Dist* td = nullptr; const Mat* tp = nullptr;
            int tdim = 2; string tcanon;
            if (base.k == Value::DistV) {
                if (!base.dist->has_score)
                    err("svgd needs either the target's score \u2207log p or its samples, and " + base.dist->canon +
                        " has no analytic score. Use a Gaussian, a Gaussian mixture, or an unnormalized log-density — "
                        "or sample it into a Dataset first (data = " + base.dist->canon +
                        " ~ n; then data ~ m via svgd(...)): svgd descends onto the KDE of the samples.");
                td = base.dist.get(); tdim = base.dist->dim; tcanon = base.dist->canon;
            } else if (base.k == Value::Data) {
                tp = &base.data->X; tdim = (int)base.data->X.num_row(); tcanon = base.data->prov;
            } else
                err("the target of via svgd must be a Distribution (its score drives the field) or a Dataset "
                    "(the field degrades to the smoothed score difference of two KDEs)");
            int steps = 500; float lr = 0.5f;
            for (auto& a : e->via->args) {
                if (a.kw == "steps") steps = (int)eval(a.e).num;
                if (a.kw == "lr")    lr = (float)eval(a.e).num;
                if (a.kw == "normalize")
                    err("normalize= is gone (2026-07): svgd IS the reverse-KL descent in the kernelized Stein "
                        "geometry (unnormalized, exact there); the normalized NW field is the same divergence "
                        "under metric=w2 — write the mechanism level: "
                        "flow(field(descent(reverseKL(p), from=q0), estimator=nw(kernel=rbf))) # ...");
            }
            printf(td ? "[svgd] (sugar) == flow(field(descent(reverseKL(target), from=N(0,I), metric=stein), nw), steps=%d) # (N(0,I) ~ %d)\n"
                      : "[svgd] (sugar) == flow(field(descent(mmd(target), from=N(0,I)), nw), steps=%d) # (N(0,I) ~ %d)\n",
                   steps, n);
            Dist init; Dist::Comp c; c.kind = "gaussian";
            c.mean.assign(tdim, 0.0f); c.s1 = 1;
            init.comps.push_back(c); init.dim = tdim; init.has_score = true;
            Mat X = sample_dist(init, n);
            auto ds = make_shared<Dataset>(X, "svgd(" + tcanon + ")");
            ds->traj.push_back(X);
            for (int i = 0; i < steps; i++) {
                svgd_step(X, td, tp, lr, false);   // stein-KL (score) / mmd (samples): unnormalized by theorem
                if ((i + 1) % max(1, steps / 12) == 0) ds->traj.push_back(X);
            }
            ds->X = X;
            Value r; r.k = Value::Data; r.data = ds; return r;
        }
        if (base.k == Value::DistV) {
            if (!base.dist->samplable) err(base.dist->canon + "  admits no exact sampler — specify an inference algorithm with via");
            Mat X = sample_dist(*base.dist, n);
            Value r; r.k = Value::Data; r.data = make_shared<Dataset>(X, base.dist->canon); return r;
        }
        if (base.k == Value::Pushed) {
            Mat X0 = sample_dist(*base.pushed->base, n);
            auto ds = make_shared<Dataset>(X0, base.pushed->canon);
            ds->X = base.pushed->map->apply(X0, &ds->traj);
            Value r; r.k = Value::Data; r.data = ds; return r;
        }
        if (base.k == Value::Data) {
            const Mat& X = base.data->X;
            uniform_int_distribution<int> pick(0, (int)X.num_col() - 1);
            Mat out("sub", X.num_row(), n);
            for (int i = 0; i < n; i++) { int j = pick(global_rand_gen);
                for (size_t r = 0; r < X.num_row(); r++) out.elem(r, i) = X.elem(r, j); }
            auto ds = make_shared<Dataset>(out, base.data->prov);
            ds->traj = base.data->traj;
            ds->blk = base.data->blk;                // subsampling keeps the (y; w) split
            Value r; r.k = Value::Data; r.data = ds; return r;
        }
        if (base.k == Value::Kernel) {
            Value r; r.k = Value::Data; r.data = sample_kernel(*base.kernel, n); return r;
        }
        if (base.k == Value::Field)
            err("a field is not a distribution and cannot be sampled — integrate it into a Map with flow(v), then form the pushforward T # mu");
        if (base.k == Value::RV || base.k == Value::Path)
            err("random variables and probability paths are not sampled directly — a path is consumed by field(prob(xt)); "
                "to view samples at a given time, transport them there with flow");
        err("the left operand of ~ must be samplable (a Distribution or a pushforward)");
    }

    // —— pushforward ——————————————————————————————————————————————
    set<string> noted;               // honesty notes print once per program
    bool note_once(const string& k) { return noted.insert(k).second; }

    Value eval_push(const ExprP& e) {
        Value L = eval(e->a), R = eval(e->b);
        // conditional map: `T # (y | x)` transports a KERNEL — the
        // condition slots stay open until `given` fills them (spec 10.8)
        if (L.k == Value::Map && L.map->f->cond_dim > 0) {
            if (R.k != Value::RV || R.rv->terms.size() != 1 || R.rv->terms[0].clo < 0)
                err("a conditional map transports a kernel — push a conditioned block: T # (y|x) (spec 10.8)");
            // DATA-backed conditioned block (2026-07, SICA Algorithm 1): the
            // per-element refinement's APPLICATION. Each column's free block
            // integrates the conditional ODE under ITS OWN pinned context;
            // when the map was trained with several condition slots, the one
            // context tiles all of them — Theorem 3.3's substitution
            // (train with (C, C'), apply with C' := C), which is also the
            // kernel surface's v1 rule "all slots share one value". Eager,
            // returns the moved table in the input's [free; pinned] layout,
            // blk-marked so it destructures. Unlike descent pushes there is
            // no provenance pin: a trained conditional net is a total
            // function, and a foreign (element | context) table is a legal
            // evaluation — same freedom as unconditional trained maps.
            if (R.rv->terms[0].data) {
                const RTerm& tm = R.rv->terms[0];
                int dy = tm.bhi - tm.blo, dxc = tm.chi - tm.clo;
                if (L.map->f->ydim != dy)
                    err("this conditional map moves a " + to_string(L.map->f->ydim) +
                        "-row block but the pushed block has " + to_string(dy) + " rows");
                if (L.map->f->cond_dim % dxc != 0)
                    err("the pushed block's condition rows (" + to_string(dxc) + ") do not tile the map's "
                        "condition slots (" + to_string(L.map->f->cond_dim) + " rows)");
                Mat Y  = rows_of(tm.data->X, tm.blo, tm.bhi);
                Mat Xc = rows_of(tm.data->X, tm.clo, tm.chi);
                int slots = L.map->f->cond_dim / dxc;
                Mat C = Xc;
                if (slots > 1) {
                    vector<MatrixView<float>> crep;
                    for (int s2 = 0; s2 < slots; s2++) crep.push_back(Xc);
                    C = vstack<float>(crep);
                }
                vector<Mat> ytr;
                Mat Yout = L.map->apply_cond(Y, C, &ytr);
                auto ds = make_shared<Dataset>(vstack<float>({Yout, Xc}),
                                               L.map->desc + "#(" + tm.srcname() + "|c)");
                ds->blk = dy;
                for (auto& fr : ytr) ds->traj.push_back(vstack<float>({fr, Xc}));
                Value r; r.k = Value::Data; r.data = ds; return r;
            }
            if (!R.rv->terms[0].dist)
                err("a conditional map transports a kernel — push a conditioned block: T # (y|x) (spec 10.8)");
            const RTerm& tm = R.rv->terms[0];
            if (tm.dist->dy != tm.bhi - tm.blo || L.map->f->ydim != tm.dist->dy)
                err("the kernel's y-block dimension does not match the map's y-dimension");
            if (L.map->f->cond_dim % tm.dist->dx != 0)
                err("the kernel's condition block does not match the map's condition slots");
            auto K = make_shared<KernelV>();
            K->map = L.map; K->joint = tm.dist; K->src = tm.src;
            K->canon = L.map->desc + "#(" + tm.srcname() + "|x)";
            Value r; r.k = Value::Kernel; r.kernel = K; return r;
        }
        // conditional descent (2026-07, SBI line): T # (y | x) re-simulates the
        // conditional-KL flow on ITS OWN conditioned ensemble — the pinned
        // rows carry the observation into every kernel weight, so foreign
        // particles would answer a different question. Hard pin, no record
        // escape (the recorded joint would never have been a joint).
        if (L.k == Value::Map && L.map->f->kind == FieldV::WG && L.map->f->free_lo >= 0) {
            if (R.k != Value::RV || R.rv->terms.size() != 1 || R.rv->terms[0].clo < 0 || !R.rv->terms[0].data)
                err("this descent moves the free block of a conditioned ensemble — push the same conditioned "
                    "block it was declared on: T # (y0 | x0), with (x0, y0) the frozen joint of from=");
            const RTerm& tm = R.rv->terms[0];
            if (tm.data != L.map->f->wg_from_x || tm.blo != L.map->f->free_lo || tm.bhi != L.map->f->free_hi)
                err("provenance mismatch (hard error): a conditional descent is pinned to its own conditioned "
                    "ensemble (" + L.map->f->from_prov + ") — the pinned rows carry the observation, so foreign "
                    "particles would simulate a different conditional.");
            if (L.map->inverse)
                err("a conditional descent admits no inverse: record= is unsupported here (the recorded joint "
                    "would never have been a joint), and without a history a gradient-flow map has no pointwise "
                    "pullback — re-simulate forward from the conditioned start instead.");
            Mat X = tm.data->X;
            vector<Mat> traj;
            Mat Y = L.map->apply(X, &traj);
            auto ds = make_shared<Dataset>(move(Y), L.map->desc + "#" + L.map->f->from_prov);
            // when the FREE block leads the table ([moved; pinned] layout, e.g.
            // lagsplit windows), mark the split so the result destructures —
            // (En, Cn) = ... reads the refined elements back out (SICA line).
            // Pins-first ensembles (all SBI examples) keep blk = 0: unchanged.
            if (L.map->f->free_lo == 0) ds->blk = L.map->f->free_hi;
            for (auto& fr : traj) ds->traj.push_back(fr);
            Value r; r.k = Value::Data; r.data = ds; return r;
        }
        // N-observation conditional descent (2026-07, KL decomposition):
        // T # (q | Obs) re-simulates the signed-sum field on ITS OWN
        // conditioned ensemble — the observation set is baked into every one
        // of the N pinned query sets, so foreign particles or foreign
        // observations would answer a different question. Hard pin.
        if (L.k == Value::Map && L.map->f->kind == FieldV::WG && L.map->f->obs) {
            if (R.k != Value::RV || R.rv->terms.size() != 1 || !R.rv->terms[0].obsd)
                err("this descent moves an ensemble conditioned on an observation set — push the same "
                    "conditioned ensemble it was declared on: T # (q | Obs), with from=(q | Obs)");
            const RTerm& tm = R.rv->terms[0];
            if (tm.data != L.map->f->wg_from_x || tm.obsd != L.map->f->obs ||
                tm.olo != L.map->f->obs_lo || tm.ohi != L.map->f->obs_hi)
                err("provenance mismatch (hard error): an N-observation descent is pinned to its own ensemble "
                    "AND its own observation set (" + L.map->f->from_prov + ") — the observations enter every "
                    "kernel weight of every summand, so foreign particles or observations would simulate a "
                    "different posterior.");
            if (L.map->inverse)
                err("an N-observation descent admits no inverse: record= is unsupported here (the "
                    "likelihood-weighted attraction is not reconstructed by the replay machinery), and "
                    "without a history a gradient-flow map has no pointwise pullback — re-simulate forward "
                    "instead.");
            Mat X = tm.data->X;
            vector<Mat> traj;
            Mat Y = L.map->apply(X, &traj);
            // A ONE-observation set is the same object as a single pinned
            // observation, so return what the sibling mode (y0 | x0) returns:
            // the pinned JOINT — observation rows tiled over the moved
            // ensemble, (y; w) layout, destructurable (blk). plot then shows
            // the posterior as the slice line at the observation, on the
            // library cloud. N > 1 has no single row to tile: the parameter
            // ensemble comes back alone (its 1-D marginal is plot_signal's job).
            int olo = L.map->f->obs_lo, ohi = L.map->f->obs_hi;
            if ((int)L.map->f->obs->X.num_col() == 1) {
                const Mat& Ob = L.map->f->obs->X;
                int dY = ohi - olo;
                auto tile = [&](const Mat& W) {
                    int dw = (int)W.num_row(), n = (int)W.num_col();
                    Mat J("obsj", dY + dw, n);
                    for (int r2 = 0; r2 < dY; r2++)
                        for (int c = 0; c < n; c++) J.elem(r2, c) = Ob.elem(olo + r2, 0);
                    for (int r2 = 0; r2 < dw; r2++)
                        for (int c = 0; c < n; c++) J.elem(dY + r2, c) = W.elem(r2, c);
                    return J;
                };
                auto ds = make_shared<Dataset>(tile(Y), L.map->desc + "#" + L.map->f->from_prov);
                ds->blk = dY;
                for (auto& fr : traj) ds->traj.push_back(tile(fr));
                Value r; r.k = Value::Data; r.data = ds; return r;
            }
            auto ds = make_shared<Dataset>(move(Y), L.map->desc + "#" + L.map->f->from_prov);
            for (auto& fr : traj) ds->traj.push_back(fr);
            Value r; r.k = Value::Data; r.data = ds; return r;
        }
        if (L.k == Value::Map && R.k == Value::RV)
            err("this map is unconditional — it pushes a Distribution or a Dataset; kernels need a conditional map (spec 10.8)");
        if (L.k == Value::Map) {
            // WG-field maps: before recording, the map IS the simulation — the
            // field is re-estimated from the moving ensemble, so foreign
            // particles change the dynamics itself: hard pin to from= (§3.2).
            // AFTER record=true has frozen the per-step ensembles, the NW
            // readout is pointwise at arbitrary points (like a trained model
            // evaluated off its training set) and the replay is self-contained
            // in both directions — foreign measures are legal, with a printed
            // note stating exactly what the answer means.
            if (L.map->f->kind == FieldV::AMORT && !L.map->f->residual) {
                // one-step generators are hard-pinned: the net was trained on
                // latents of from= only — a foreign measure is an
                // out-of-distribution input, and there is no record= escape
                // (the weights ARE the record, and they only speak from=).
                // EXCEPTION: the residual instantaneous demixer (Level 1,
                // SICA) is a full function of R^d and applies freely —
                // being holdable and reusable is its point.
                string rp = R.k == Value::DistV ? R.dist->canon
                          : R.k == Value::Data ? R.data->prov : string();
                if (rp.empty()) err("the right operand of # must be a Distribution or a Dataset");
                if (rp != L.map->f->from_prov)
                    err("provenance mismatch (hard error): this one-step generator was drifted on latents from " +
                        L.map->f->from_prov + ", but you applied it to " + rp +
                        ". A net evaluated off its training measure is an out-of-distribution guess, and an amortized "
                        "map has no record=true escape — the weights are the record, and they only speak for from=.");
            }
            if (L.map->f->kind == FieldV::WG) {
                string rp = R.k == Value::DistV ? R.dist->canon
                          : R.k == Value::Data ? R.data->prov : string();
                if (rp.empty()) err("the right operand of # must be a Distribution or a Dataset");
                bool recorded = L.map->hist && !L.map->hist->empty();
                if (L.map->inverse) {
                    if (!recorded)
                        err("inversion needs a recorded history: apply this map (record=true) forward once first");
                    if (L.map->endpoint_prov && !L.map->endpoint_prov->empty() && rp != *L.map->endpoint_prov
                        && note_once("inv|" + rp + "|" + *L.map->endpoint_prov))
                        printf("[inv] note: %s is not the recorded endpoint (%s) — replaying the frozen map on foreign\n"
                               "      points; the pullback reads as a transport back to %s exactly insofar as the\n"
                               "      forward flow converged.\n",
                               rp.c_str(), L.map->endpoint_prov->c_str(), L.map->f->from_prov.c_str());
                } else if (rp != L.map->f->from_prov) {
                    if (recorded) {
                        if (note_once("fwd|" + rp + "|" + L.map->f->from_prov))
                            printf("[flow] note: %s is not the pinned start (%s) — replaying the frozen per-step fields\n"
                                   "      on foreign points (the recorded map is self-contained; this is NOT a\n"
                                   "      re-simulation of the gradient flow).\n",
                                   rp.c_str(), L.map->f->from_prov.c_str());
                    }
                    else
                        err("provenance mismatch (hard error): this gradient flow is the initial-value problem of " + L.map->f->div->canon +
                            " started at from=" + L.map->f->from_prov + ", but the object you applied it to is drawn from " +
                            rp + ". The trajectory of a Wasserstein gradient flow depends on its "
                            "initial measure — from a different start, the field is no longer the descent direction of this "
                            "divergence. (record=true freezes the per-step ensembles and lifts this pin: a recorded map "
                            "replays at arbitrary points.)");
                }
            }
            if (R.k == Value::DistV) {
                auto pd = make_shared<PushedDist>();
                pd->map = L.map; pd->base = R.dist;
                pd->canon = L.map->desc + "#" + R.dist->canon;
                Value r; r.k = Value::Pushed; r.pushed = pd; return r;
            }
            if (R.k == Value::Data) {
                if (L.map->f->kind == FieldV::WG)
                    printf("[flow/WG] %d particles, %d steps (nw/rbf)\n",
                           (int)R.data->X.num_col(), L.map->steps);
                auto ds = make_shared<Dataset>(R.data->X, L.map->desc + "#" + R.data->prov);
                ds->X = L.map->apply(R.data->X, &ds->traj);
                if (L.map->f->kind == FieldV::WG && L.map->record && !L.map->inverse &&
                    L.map->endpoint_prov && L.map->endpoint_prov->empty())
                    *L.map->endpoint_prov = ds->prov;
                Value r; r.k = Value::Data; r.data = ds; return r;
            }
            err("the right operand of # must be a Distribution or a Dataset");
        }
        if (L.k == Value::Field)
            err("v # mu is illegal: a field must first be integrated into a map. Write flow(" +
                (L.field ? L.field->desc : string("v")) + ", steps=...) # ... (spec 3.3: no sugar here, by design)");
        if (L.k == Value::WGPath)
            err("a path cannot push a measure directly — extract its field with field(...), then integrate with flow(...)");
        err("the left operand of # must be a Map");
    }

    // —— statements ———————————————————————————————————————————————
    void run(const vector<Stmt>& prog) {
        for (auto& s : prog) exec(s);
    }

    // one statement; bounded for (spec 10.1) recurses with the loop index on
    // the iter stack, so every event a body statement emits carries iter=[..k]
    void exec(const Stmt& s) {
            cur_line = s.line;
            if (s.k == Stmt::For) {
                if (!seeded) { global_rand_gen.seed(0); seeded = true; printf("(auto) seed 0\n\n"); }
                bool shadowed = env.count(s.name) > 0;
                Value saved; if (shadowed) saved = env[s.name];
                for (int k = s.for_lo; k <= s.for_hi; k++) {
                    Value kv; kv.k = Value::Num; kv.num = k;
                    env[s.name] = kv;                    // k: a constant of this copy of the body
                    iter_stack.push_back(k);
                    printf("—— iter %s = %d ——\n", s.name.c_str(), k);
                    for (auto& b : s.body) exec(b);
                    iter_stack.pop_back();
                    cur_line = s.line;
                }
                if (shadowed) env[s.name] = saved; else env.erase(s.name);
                return;
            }
            if (s.k == Stmt::Seed) {
                global_rand_gen.seed((unsigned)s.seed); seeded = true;
                printf("seed %ld\n\n", s.seed);
                return;
            }
            if (!seeded) { global_rand_gen.seed(0); seeded = true; printf("(auto) seed 0\n\n"); }
            if (s.k == Stmt::Bind) {
                if (s.name == "t")
                    err("t is a reserved symbol (the time of a path formula) and cannot be bound");
                env[s.name] = eval(s.e);
                return;
            }
            if (s.k == Stmt::BindPair) {             // (y, x) = one joint draw (spec 10.8)
                for (const string& nm : {s.name, s.name2})
                    if (nm == "t")
                        err("t is a reserved symbol (the time of a path formula) and cannot be bound");
                // eager mode (spec 10.3): (x0, x1) = rv(couple(...)) ~ n freezes ONE
                // coupling of n pairs; the two names become block views of the joint
                // Dataset (same src — training subsamples columns once per step,
                // so the pairing survives). Without ~, the pairing is re-drawn and
                // re-matched inside every training batch (minibatch coupling).
                shared_ptr<Dataset> J; int ndraw = 0;
                ExprP body = s.e;
                if (body->k == Expr::Sample && !body->via) { ndraw = body->n; body = body->a; }
                Value v = eval(body);
                // kernel-sampled joints (spec 10.10): the joint is a frozen
                // Dataset carrying its (y; w) row split (Dataset::blk), so the
                // two names become BLOCK VIEWS of one table — same machinery
                // as the frozen coupling below, provenance is the Dataset.
                if (v.k == Value::Kernel || (v.k == Value::Data && v.data->blk > 0)) {
                    shared_ptr<Dataset> KJ;
                    if (v.k == Value::Kernel) {
                        if (v.kernel->inst == 0)
                            err("a kernel is a family of distributions — instantiate its condition slot with | "
                                "first, then freeze draws: (y, w) = (K | z) ~ n (spec 10.8/10.10)");
                        if (ndraw <= 0)
                            err("(y, w) = (K | z) destructures FROZEN draws, and a kernel instantiation is a "
                                "law, not a table — write ~ n to fix how many joint draws the two names share");
                        if (ndraw > 100000) err("sampling budget exceeded: a Dataset holds at most 100k samples");
                        KJ = sample_kernel(*v.kernel, ndraw);
                    } else KJ = ndraw > 0 ? eval(s.e).data : v.data;
                    int dall = (int)KJ->X.num_row(), yb = KJ->blk;
                    long ksrc = fresh_src();
                    auto mkb = [&](int lo, int hi) {
                        Value r; r.k = Value::RV; r.rv = make_shared<RVal>();
                        RTerm tm; tm.coeff = [](float) { return 1.0f; }; tm.cdesc = "1";
                        tm.src = ksrc; tm.anon = false; tm.blo = lo; tm.bhi = hi;
                        tm.data = KJ;
                        r.rv->terms.push_back(move(tm)); return r;
                    };
                    env[s.name]  = mkb(0, yb);
                    env[s.name2] = mkb(yb, dall);
                    return;
                }
                long src; shared_ptr<Dist> D;
                if (v.k == Value::RV && v.rv->terms.size() == 1 && v.rv->terms[0].dist && v.rv->terms[0].blo < 0) {
                    D = v.rv->terms[0].dist; src = v.rv->terms[0].src;
                } else if (v.k == Value::DistV) { D = v.dist; src = fresh_src(); }
                else err("(y, x) = ... destructures ONE draw of a joint distribution — bind rv(linear_gaussian), rv(couple(...)), or a joint constructor");
                if (D->dy <= 0)
                    err(D->canon + " is not a joint distribution — only joints carry (y, x) coordinate blocks (linear_gaussian, sine_gaussian, couple)");
                if (ndraw > 0) {
                    if (!D->pair_blocks)
                        err("(y, x) = rv(joint) ~ n freezes a coupling (spec 10.3) and is defined for pair-joints "
                            "(couple(...), mixed_sources) only — conditional joints are consumed per-batch inside the training loop");
                    if (ndraw > 100000) err("sampling budget exceeded: a Dataset holds at most 100k samples");
                    J = make_shared<Dataset>(sample_dist(*D, ndraw),
                                             D->canon + "~" + to_string(ndraw));
                }
                auto mk = [&](int lo, int hi) {
                    Value r; r.k = Value::RV; r.rv = make_shared<RVal>();
                    RTerm tm; tm.coeff = [](float) { return 1.0f; }; tm.cdesc = "1";
                    tm.src = src; tm.anon = false; tm.blo = lo; tm.bhi = hi;
                    tm.peer = D->pair_blocks;
                    if (J) tm.data = J; else tm.dist = D;
                    r.rv->terms.push_back(move(tm)); return r;
                };
                env[s.name]  = mk(0, D->dy);
                env[s.name2] = mk(D->dy, D->dy + D->dx);
                return;
            }
            if (s.k == Stmt::PlotSignal) {
                // plot_signal x, y: draw each Dataset's columns in index order,
                // one line per coordinate row. A bare Distribution/pushforward
                // samples 500 i.i.d. points first (order is then draw order).
                vector<Value> keep; vector<SignalSeries> lines; string title;
                for (auto& pe : s.plots) {
                    Value v = eval(pe);
                    if (v.k == Value::Kernel && v.kernel->inst == 0)
                        err("a kernel is a family of distributions — instantiate its condition slot with | before plotting (spec 10.8)");
                    if (v.k == Value::DistV || v.k == Value::Pushed || v.k == Value::Kernel) {
                        ExprP se = make_shared<Expr>(); se->k = Expr::Sample; se->a = pe; se->n = 500;
                        v = eval(se);
                    }
                    if (v.k == Value::RV)
                        err("plot_signal draws Datasets — a random variable carries draw identity, not samples; "
                            "take its law first: plot_signal prob(x)");
                    if (v.k != Value::Data)
                        err("plot_signal draws Datasets (or samplable objects — a bare Distribution samples 500)");
                    keep.push_back(v);
                }
                for (auto& kv : keep) {
                    int d = (int)kv.data->X.num_row();
                    for (int r = 0; r < d; r++)
                        lines.push_back({kv.data->prov + (d > 1 ? "[" + to_string(r) + "]" : ""),
                                         &kv.data->X, r});
                    title += (title.empty() ? "" : " vs ") + kv.data->prov;
                }
                if (FILE* fp = dump_file()) {
                    fprintf(fp, "{\"type\":\"signal\",\"line\":%d,\"iter\":%s,\"series\":[",
                            cur_line, iter_json().c_str());
                    for (size_t k = 0; k < lines.size(); k++) {
                        const Mat& X = *lines[k].X; int n = (int)X.num_col();
                        int ns = min(n, SIGNAL_SHOW);                // first-200 window (zoom, not decimation)
                        fprintf(fp, "%s{\"label\":\"%s\",\"n\":%d,\"total\":%d,\"stride\":1,\"values\":[",
                                k ? "," : "", lines[k].label.c_str(), ns, n);
                        for (int i = 0; i < ns; i++)
                            fprintf(fp, "%s%.4g", i ? "," : "", (double)X.elem(lines[k].row, i));
                        fprintf(fp, "]}");
                    }
                    fprintf(fp, "]}\n"); fflush(fp);
                }
                ascii_signal(lines, title);
                return;
            }
            if (s.k == Stmt::Plot) {
                vector<Value> keep;
                string title;
                for (size_t i = 0; i < s.plots.size(); i++) {
                    Value v = eval(s.plots[i]);
                    if (v.k == Value::Kernel && v.kernel->inst == 0)
                        err("a kernel is a family of distributions — instantiate its condition slot with | before plotting (spec 10.8)");
                    if (v.k == Value::Pushed || v.k == Value::Kernel) {
                        ExprP se = make_shared<Expr>(); se->k = Expr::Sample; se->a = s.plots[i]; se->n = 500;
                        v = eval(se);
                    }
                    if (v.k != Value::Data) err("plot draws Datasets (or samplable objects)");
                    if ((int)v.data->X.num_row() < 2)
                        err("plot is the 2-D scatter view (rows 0 and 1 are the coordinates), and " +
                            v.data->prov + " has a single row — 1-D samples carry no second coordinate. "
                            "Draw them as a signal instead: plot_signal <name> (values over draw index; "
                            "the vertical band is the distribution), or plot a joint that carries them "
                            "as one of its blocks.");
                    keep.push_back(v);
                    // 3-D eligibility: a 3-row CLOUD (one coordinate block),
                    // not a 3-row kernel joint (1-D output + 2-D parameter)
                    auto d3ok = [](const shared_ptr<Dataset>& d) {
                        return (int)d->X.num_row() == 3 && (d->blk == 0 || d->blk == 3);
                    };
                    if (s.is_traj[i]) {
                        if (v.data->traj.size() < 2)
                            err("this dataset carries no trajectory (it is not the output of a flow)");
                        auto& T = v.data->traj;
                        bool d3t = d3ok(v.data);
                        if (FILE* fp = dump_file()) {
                            fprintf(fp, "{\"type\":\"traj\",\"line\":%d,\"iter\":%s,\"label\":\"%s\",\"frames\":[",
                                    cur_line, iter_json().c_str(), v.data->prov.c_str());
                            for (size_t k = 0; k < T.size(); k++) { if (k) fprintf(fp, ","); dump_json_pts(fp, T[k], d3t); }
                            fprintf(fp, "]}\n"); fflush(fp);
                        }
                        vector<Series> fr = {
                            {"t=0", &T.front()},
                            {"t=0.5", &T[T.size() / 2]},
                            {"t=1", &T.back()} };
                        ascii_scatter(fr, "trajectory of " + v.data->prov, d3t);
                    } else {
                        title += (title.empty() ? "" : " vs ") + v.data->prov;
                    }
                }
                vector<Series> flat;
                bool d3all = true;
                for (size_t i = 0; i < keep.size(); i++)
                    if (!s.is_traj[i]) {
                        flat.push_back({keep[i].data->prov, &keep[i].data->X});
                        d3all = d3all && (int)keep[i].data->X.num_row() == 3 &&
                                (keep[i].data->blk == 0 || keep[i].data->blk == 3);
                    }
                if (!flat.empty()) {
                    if (FILE* fp = dump_file()) {
                        fprintf(fp, "{\"type\":\"scatter\",\"line\":%d,\"iter\":%s,\"series\":[",
                                cur_line, iter_json().c_str());
                        for (size_t k = 0; k < flat.size(); k++) {
                            fprintf(fp, "%s{\"label\":\"%s\",\"pts\":", k ? "," : "", flat[k].label.c_str());
                            dump_json_pts(fp, *flat[k].X, d3all);
                            fprintf(fp, "}");
                        }
                        fprintf(fp, "]}\n"); fflush(fp);
                    }
                    ascii_scatter(flat, title, d3all);
                }
            }
    }
};

// ─────────────────────────────────────────── export (PyTorch) ───────
// `liu --export-pytorch prog.liu` prints a runnable PyTorch mirror of the
// program on stdout — the scale-out escape hatch. TEMPLATE-BASED, not a
// general transpiler. Covered pipelines: regress-FM/diffusion (path
// coefficients t / 1-t / sqrt(1-t·t); endpoints = dists, datasets, inline
// `dist ~ n`, or couple(A, B, via=ot/paired) pairs — per-batch Hungarian
// for ot, (z, T(z)) for paired), field algebra (a*v1 + b*v2 → linear
// combination of nets), flow / inv(·) / bounded `for` (macro expansion;
// rebindings get versioned python names — late binding must not leak a
// retrained net into an older map), and score-target SVGD in both
// geometries (metric=stein unnormalized / w2-default normalized, mirroring
// svgd_field_at and both svgd_bandwidth calibers), incl. the `via svgd`
// sugar (init N(0, I)). Everything else is REFUSED with a pointer — a
// wrong assembly must never be silently generated (the ledger's measured
// assembly decisions do not carry over by accident). Exported code is
// semantically equivalent, NOT bit-identical (different RNG streams and
// backend): the reproducibility contract stays on the Liu side.
struct PyExport {
    enum NK { DIST, DATA, FORMULA, PATH, FIELD, DESCENT, MAP, COUPLED };
    enum FK { REGRESS, NW, COMBO };
    enum MK { FMMAP, SVGDMAP };
    struct Comp { double w; vector<double> mean; double s; };
    struct Info {
        NK kind;
        string py;                            // python identifier of THIS binding
        int dim = 0;
        vector<pair<string, string>> terms;   // FORMULA: (coeff kind, block liu-name)
        string src, src2;                     // PATH→formula; FIELD→path/descent;
                                              // MAP→field; DESCENT: src=target, src2=from
        vector<int> dims;
        int steps = 0, batch = 128;
        double lr = 0;
        FK fk = REGRESS;
        MK mk = FMMAP;
        vector<Comp> comps;                   // DIST: score components
        vector<pair<double, string>> combo;   // FIELD COMBO: (coeff, field liu-name)
        bool has_traj = false;
        bool normalize = false;               // DESCENT: metric w2 (default) vs stein
        bool score_emitted = false;
        int side = 0;                         // COUPLED: 0 = first arg, 1 = second
        string pairfn;                        // COUPLED: python pair-drawing function
        string marginal;                      // COUPLED: python sampler expr "(n)"-callable
        string velfn;                         // MAP FMMAP: python velocity callable
    };
    map<string, Info> names;
    map<string, int> gen;
    string out;
    int plot_id = 0, anon_id = 0;
    bool used_traj = false, used_svgd = false, used_batch = false,
         used_mlp = false, used_ot = false;

    [[noreturn]] void refuse(int line, const string& what) {
        err("line " + to_string(line) + ": --export-pytorch covers the regress-FM/"
            "diffusion (incl. couple/guidance/inv/for) and score-target SVGD "
            "pipelines only — " + what + " has no export template yet. Templates "
            "are added one verified pipeline at a time so a wrong assembly is "
            "never silently generated; run this program under the interpreter, "
            "or ask for this pipeline's template.");
    }
    void emit(const string& s) { out += s; }

    // rebinding a liu name mints a fresh python identifier — python's late
    // binding must never let a retrained net leak into an older map (reflow)
    string bindPy(const string& name) {
        int g = ++gen[name];
        return g == 1 ? name : name + "__" + to_string(g);
    }
    Info& info(const string& name, int line) {
        auto it = names.find(name);
        if (it == names.end()) refuse(line, "the name `" + name + "`");
        return it->second;
    }

    static string pynum(double x) {
        char b[32]; snprintf(b, sizeof b, "%g", x); return b;
    }
    static string pyvec(const vector<double>& v) {
        string s = "[";
        for (size_t i = 0; i < v.size(); i++)
            s += (i ? ", " : "") + string("[") + pynum(v[i]) + "]";
        return s + "]";
    }
    static string coeffKind(ExprP c) {
        if (c->k == Expr::Ident && c->id == "t") return "t";
        if (c->k == Expr::Bin && c->id == "-" && c->a->k == Expr::Num && c->a->num == 1 &&
            c->b->k == Expr::Ident && c->b->id == "t") return "1mt";
        if (c->k == Expr::Call && c->id == "sqrt" && c->args.size() == 1) {
            ExprP i = c->args[0].e;
            if (i->k == Expr::Bin && i->id == "-" && i->a->k == Expr::Num && i->a->num == 1 &&
                i->b->k == Expr::Bin && i->b->id == "*" &&
                i->b->a->k == Expr::Ident && i->b->a->id == "t" &&
                i->b->b->k == Expr::Ident && i->b->b->id == "t") return "sq";
        }
        return "";
    }
    static string cPy(const string& k) {
        return k == "t" ? "t" : k == "1mt" ? "(1 - t)" : "torch.sqrt(1 - t*t)";
    }
    static string dcPy(const string& k) {
        return k == "t" ? "torch.ones_like(t)"
             : k == "1mt" ? "(-torch.ones_like(t))"
             : "(-t / torch.sqrt(1 - t*t))";   // t sampled in [0.02, 0.98]: no pole
    }

    Comp compOf(ExprP part, int line) {
        double w = 1; ExprP g = part;
        if (part->k == Expr::Bin && part->id == "*" && part->a->k == Expr::Num) {
            w = part->a->num; g = part->b;
        }
        if (g->k != Expr::Call || g->id != "gaussian" || g->args.size() != 2 ||
            g->args[0].e->k != Expr::Vec || g->args[1].e->k != Expr::Num)
            refuse(line, "this mixture component (template: w*gaussian([..], s))");
        return { w, g->args[0].e->vec, g->args[1].e->num };
    }

    void emitDistDef(const string& name, ExprP e, int line) {
        string py = bindPy(name);
        if (e->k == Expr::Mix) {
            vector<Comp> cs;
            for (auto& p : e->parts) cs.push_back(compOf(p, line));
            int d = (int)cs[0].mean.size();
            names[name] = { DIST, py, d };
            names[name].comps = cs;
            emit("def sample_" + py + "(n):   # line " + to_string(line) +
                 ": mixture of " + to_string(cs.size()) + " gaussians\n");
            string ws = "[", ms = "[", ss = "[";
            for (size_t i = 0; i < cs.size(); i++) {
                ws += (i ? ", " : "") + pynum(cs[i].w);
                ms += (i ? ", " : "") + pyvec(cs[i].mean);
                ss += (i ? ", " : "") + pynum(cs[i].s);
            }
            emit("    ws = torch.tensor(" + ws + "])\n"
                 "    means = torch.tensor(" + ms + "])       # k x d x 1\n"
                 "    stds = torch.tensor(" + ss + "])\n"
                 "    idx = torch.multinomial(ws / ws.sum(), n, replacement=True)\n"
                 "    return means[idx, :, 0].T + stds[idx] * torch.randn(" +
                 to_string(d) + ", n)\n\n");
            return;
        }
        if (e->id == "gaussian" && e->args.size() == 2 &&
            e->args[0].e->k == Expr::Vec && e->args[1].e->k == Expr::Num) {
            auto& mean = e->args[0].e->vec;
            names[name] = { DIST, py, (int)mean.size() };
            names[name].comps = { { 1.0, mean, e->args[1].e->num } };
            emit("def sample_" + py + "(n):   # line " + to_string(line) +
                 ": gaussian(mean, std)\n");
            emit("    return torch.tensor(" + pyvec(mean) + ") + " +
                 pynum(e->args[1].e->num) + " * torch.randn(" +
                 to_string(mean.size()) + ", n)\n\n");
        } else if (e->id == "moons" && e->args.size() == 1 && e->args[0].e->k == Expr::Num) {
            names[name] = { DIST, py, 2 };
            emit("def sample_" + py + "(n, noise=" + pynum(e->args[0].e->num) +
                 "):   # line " + to_string(line) + ": moons(noise), two half-circles\n");
            emit("    half = n // 2\n"
                 "    th1 = torch.rand(half) * torch.pi\n"
                 "    th2 = torch.rand(n - half) * torch.pi\n"
                 "    a = torch.stack([torch.cos(th1), torch.sin(th1)])\n"
                 "    b = torch.stack([1 - torch.cos(th2), 0.5 - torch.sin(th2)])\n"
                 "    X = torch.cat([a, b], dim=1)\n"
                 "    return X + noise * torch.randn_like(X)\n\n");
        } else if (e->id == "uniform" && e->args.size() == 2 &&
                   e->args[0].e->k == Expr::Vec && e->args[1].e->k == Expr::Vec) {
            auto &lo = e->args[0].e->vec, &hi = e->args[1].e->vec;
            names[name] = { DIST, py, (int)lo.size() };
            emit("def sample_" + py + "(n):   # line " + to_string(line) +
                 ": uniform(lo, hi)\n");
            emit("    lo = torch.tensor(" + pyvec(lo) + "); hi = torch.tensor(" +
                 pyvec(hi) + ")\n");
            emit("    return lo + (hi - lo) * torch.rand(" + to_string(lo.size()) + ", n)\n\n");
        } else {
            refuse(line, "distribution `" + e->id + "` (or this argument form)");
        }
    }

    void emitScore(const string& name, int line) {
        Info& d = info(name, line);
        if (d.score_emitted) return;
        if (d.kind != DIST || d.comps.empty())
            refuse(line, "a score for `" + name + "` (template: gaussian or gaussian "
                         "mixture targets; samples-only and unnormalized targets)");
        d.score_emitted = true;
        emit("def _logp_" + d.py + "(x):   # unnormalized log density "
             "(constants dropped — the score is unaffected)\n"
             "    lps = torch.stack([\n");
        for (auto& c : d.comps)
            emit("        math.log(" + pynum(c.w) + ") - ((x - torch.tensor(" +
                 pyvec(c.mean) + "))**2).sum(0) / (2 * " + pynum(c.s) + "**2) - " +
                 to_string(c.mean.size()) + " * math.log(" + pynum(c.s) + "),\n");
        emit("    ])\n"
             "    return torch.logsumexp(lps, dim=0)\n\n");
        emit("def score_" + d.py + "(x):   # ∇log p via autograd (exact here)\n"
             "    x = x.detach().requires_grad_(True)\n"
             "    g, = torch.autograd.grad(_logp_" + d.py + "(x).sum(), x)\n"
             "    return g\n\n");
    }

    // "how to draw n points from this measure", as a python call string
    string samplerOf(const string& name, const string& n, int line) {
        Info& i = info(name, line);
        if (i.kind == DIST) return "sample_" + i.py + "(" + n + ")";
        if (i.kind == DATA) { used_batch = true; return "_batch_from(" + i.py + ")(" + n + ")"; }
        refuse(line, "drawing from `" + name + "`");
    }

    // couple side: Ident dist/dataset, or Push{map # dist} (reflow's T # noise)
    string sideSampler(ExprP e, const string& n, int line) {
        if (e->k == Expr::Ident) return samplerOf(e->id, n, line);
        if (e->k == Expr::Push && e->a->k == Expr::Ident && e->b->k == Expr::Ident) {
            Info& m = info(e->a->id, line);
            if (m.kind != MAP) refuse(line, "this couple side");
            return "push_" + m.py + "(" + samplerOf(e->b->id, n, line) + ")";
        }
        refuse(line, "this couple side");
    }

    // (x0, x1) = [rv](couple(A, B, via=ot|paired)) [~ n]
    void bindCouple(const Stmt& s, ExprP cp, int eager_n, int line) {
        string via;
        ExprP A, B;
        for (auto& a : cp->args) {
            if (a.kw.empty() && !A) A = a.e;
            else if (a.kw.empty()) B = a.e;
            else if (a.kw == "via" && a.e->k == Expr::Ident) via = a.e->id;
            else refuse(line, "couple argument `" + a.kw + "`");
        }
        if (!A || !B || (via != "ot" && via != "paired"))
            refuse(line, "this couple form (template: via=ot / via=paired)");
        string py0 = bindPy(s.name), py1 = bindPy(s.name2);
        string fn = "_pair_" + py0 + "_" + py1;
        if (via == "ot") {
            used_ot = true;
            emit("def " + fn + "(B):   # line " + to_string(line) +
                 ": couple(via=ot) — exact OT matching within the batch (Hungarian)\n"
                 "    a = " + sideSampler(A, "B", line) + "\n"
                 "    b = " + sideSampler(B, "B", line) + "\n"
                 "    ri, ci = linear_sum_assignment((torch.cdist(a.T, b.T) ** 2).numpy())\n"
                 "    return a[:, ri], b[:, ci]\n\n");
        } else {
            // paired: side B must be side A's draw pushed through a map
            if (B->k != Expr::Push || B->b->k != Expr::Ident || A->k != Expr::Ident ||
                B->b->id != A->id || B->a->k != Expr::Ident)
                refuse(line, "this via=paired form (template: couple(mu, T # mu, via=paired))");
            Info& m = info(B->a->id, line);
            if (m.kind != MAP) refuse(line, "this via=paired form");
            emit("def " + fn + "(B):   # line " + to_string(line) +
                 ": couple(via=paired) — one draw z with its own image " +
                 B->a->id + "(z)\n"
                 "    z = " + samplerOf(A->id, "B", line) + "\n"
                 "    return z, push_" + m.py + "(z)\n\n");
        }
        auto marg = [&](ExprP side) {
            // Law(x0) is the side's own marginal — coupling never moves it
            if (side->k == Expr::Ident) return side->id;
            return string("");                 // pushed side: no named marginal
        };
        if (eager_n > 0) {
            emit(py0 + ", " + py1 + " = " + fn + "(" + to_string(eager_n) +
                 ")   # line " + to_string(line) + ": ONE frozen coupling of " +
                 to_string(eager_n) + " pairs\n\n");
            names[s.name] = { DATA, py0 };
            names[s.name2] = { DATA, py1 };
        } else {
            names[s.name] = { COUPLED, py0 };
            names[s.name].side = 0; names[s.name].pairfn = fn;
            names[s.name].marginal = marg(A);
            names[s.name2] = { COUPLED, py1 };
            names[s.name2].side = 1; names[s.name2].pairfn = fn;
            names[s.name2].marginal = marg(B);
        }
    }

    // field algebra: an expression whose leaves are Nums and FIELD names
    bool isFieldAlgebra(ExprP e) {
        if (e->k == Expr::Num) return true;
        if (e->k == Expr::Ident)
            return names.count(e->id) && names[e->id].kind == FIELD;
        if (e->k == Expr::Bin && (e->id == "+" || e->id == "-" || e->id == "*"))
            return isFieldAlgebra(e->a) && isFieldAlgebra(e->b);
        return false;
    }
    void comboWalk(ExprP e, double scale, vector<pair<double, string>>& acc, int line) {
        if (e->k == Expr::Ident) { acc.push_back({ scale, e->id }); return; }
        if (e->k == Expr::Bin && (e->id == "+" || e->id == "-")) {
            comboWalk(e->a, scale, acc, line);
            comboWalk(e->b, e->id == "+" ? scale : -scale, acc, line);
            return;
        }
        if (e->k == Expr::Bin && e->id == "*") {
            if (e->a->k == Expr::Num) { comboWalk(e->b, scale * e->a->num, acc, line); return; }
            if (e->b->k == Expr::Num) { comboWalk(e->a, scale * e->b->num, acc, line); return; }
        }
        refuse(line, "this field-algebra expression");
    }

    void bindFormula(const string& name, ExprP e, int line) {
        Info fm; fm.kind = FORMULA; fm.py = bindPy(name);
        for (ExprP term : { e->a, e->b }) {
            if (term->k != Expr::Bin || term->id != "*") refuse(line, "this path formula");
            ExprP c = term->a, blk = term->b;
            string k = coeffKind(c);
            if (k.empty() && coeffKind(blk) != "") { swap(c, blk); k = coeffKind(c); }
            if (k.empty()) refuse(line, "this path coefficient (template set: t, 1-t, sqrt(1-t*t))");
            string blkName;
            if (blk->k == Expr::Ident && names.count(blk->id)) blkName = blk->id;
            else if (blk->k == Expr::Sample && !blk->via && blk->a->k == Expr::Ident &&
                     names.count(blk->a->id) && names[blk->a->id].kind == DIST) {
                // inline frozen endpoint: t*(p_all ~ 2000) — materialize it
                blkName = "_ep" + to_string(++anon_id);
                string epy = bindPy(blkName);
                emit(epy + " = sample_" + names[blk->a->id].py + "(" + to_string(blk->n) +
                     ")   # line " + to_string(line) + ": inline frozen endpoint " +
                     blk->a->id + " ~ " + to_string(blk->n) + "\n\n");
                names[blkName] = { DATA, epy };
            } else refuse(line, "this path formula endpoint");
            fm.terms.push_back({ k, blkName });
        }
        if (fm.terms[0].second == fm.terms[1].second)
            err("line " + to_string(line) + ": both formula endpoints are `" +
                fm.terms[0].second + "` — the interpreter rejects this (no draw "
                "identity), and the export must not silently train on two "
                "independent draws of the same measure.");
        names[name] = fm;
    }

    void bindField(const string& name, ExprP e, int line) {
        Info f; f.kind = FIELD; f.py = bindPy(name); f.steps = 3500; f.lr = 2e-3;
        string srcName;
        for (auto& a : e->args) {
            if (a.kw.empty() && a.e->k == Expr::Ident && names.count(a.e->id))
                srcName = a.e->id;
            else if (a.kw.empty() && a.e->k == Expr::Call && a.e->id == "prob") {
                // inline field(prob(xt), ...) — resolve through the law gate
                if (a.e->args.size() == 1 && a.e->args[0].e->k == Expr::Ident)
                    srcName = a.e->args[0].e->id;
                else refuse(line, "this field source");
            }
            else if (a.kw == "estimator" && a.e->k == Expr::Call &&
                     a.e->id == "regress" && a.e->args.size() == 1 &&
                     a.e->args[0].e->k == Expr::Dims && a.e->args[0].e->id == "mlp")
                f.dims = a.e->args[0].e->dims;
            else if (a.kw == "estimator" && a.e->k == Expr::Call && a.e->id == "nw")
                f.fk = NW;   // kernel=rbf is the only kernel
            else if (a.kw == "steps" && a.e->k == Expr::Num) f.steps = (int)a.e->num;
            else if (a.kw == "lr" && a.e->k == Expr::Num) f.lr = a.e->num;
            else if (a.kw == "batch" && a.e->k == Expr::Num) f.batch = (int)a.e->num;
            else refuse(line, "field argument `" + a.kw + "`");
        }
        if (srcName.empty()) refuse(line, "this field form");
        Info& src = info(srcName, line);
        if (f.fk == REGRESS) {
            if (f.dims.empty()) refuse(line, "this field form (regress needs an mlp)");
            if (src.kind == PATH) f.src = src.src;
            else if (src.kind == FORMULA) f.src = srcName;
            else refuse(line, "this field source (regress needs prob(formula))");
            names[name] = f;
            emitFieldTraining(name, line);
        } else {
            if (src.kind != DESCENT) refuse(line, "this field form (nw needs a descent)");
            f.src = srcName;
            names[name] = f;
        }
    }

    void emitFieldTraining(const string& vName, int line) {
        used_mlp = true;
        Info& f = names[vName];
        Info& fm = names[f.src];
        auto& tA = fm.terms[0];
        auto& tB = fm.terms[1];
        Info &bA = info(tA.second, line), &bB = info(tB.second, line);
        string batchLines;
        if (bA.kind == COUPLED || bB.kind == COUPLED) {
            if (bA.kind != COUPLED || bB.kind != COUPLED || bA.pairfn != bB.pairfn)
                refuse(line, "this formula (endpoints must be the two sides of ONE couple)");
            // pair fn returns (side0, side1); map onto term order
            string lhs = (bA.side == 0) ? "xa, xb" : "xb, xa";
            batchLines = "        " + lhs + " = " + bA.pairfn + "(batch)\n";
        } else {
            batchLines = "        xa = " + samplerOf(tA.second, "batch", line) + "\n"
                         "        xb = " + samplerOf(tB.second, "batch", line) + "\n";
        }
        int print_every = f.steps > 7 ? f.steps / 7 : f.steps;
        string dimlist = "[";
        for (size_t i = 0; i < f.dims.size(); i++)
            dimlist += (i ? ", " : "") + to_string(f.dims[i]);
        dimlist += "]";
        emit("# line " + to_string(line) + ": " + vName +
             " = field(prob(" + f.src + "), estimator=regress(mlp), steps=" +
             to_string(f.steps) + ", lr=" + pynum(f.lr) + ")\n"
             "# regress E[dx/dt | x_t] on xt = " + cPy(tA.first) + "*" + tA.second +
             " + " + cPy(tB.first) + "*" + tB.second + "  (bench_torch.py op structure)\n");
        emit("def _train_" + f.py + "():\n"
             "    net = _mlp(" + dimlist + ")\n"
             "    opt = torch.optim.Adam(net.parameters(), lr=" + pynum(f.lr) + ", eps=1e-8)\n"
             "    steps, batch = " + to_string(f.steps) + ", " + to_string(f.batch) + "\n"
             "    for i in range(steps):\n"
             "        t = torch.rand(1, batch) * 0.96 + 0.02\n" +
             batchLines +
             "        xt = " + cPy(tA.first) + " * xa + " + cPy(tB.first) + " * xb\n"
             "        dxt = " + dcPy(tA.first) + " * xa + " + dcPy(tB.first) + " * xb\n"
             "        pred = net(torch.cat([xt, t], dim=0).T)\n"
             "        loss = torch.nn.functional.mse_loss(pred, dxt.T)\n"
             "        opt.zero_grad(); loss.backward(); opt.step()\n"
             "        if (i + 1) % " + to_string(print_every) + " == 0:\n"
             "            print(f'[" + vName + "] step {i+1}/{steps}  loss {loss.item():.4f}')\n"
             "    return net\n");
        emit("net_" + f.py + " = _train_" + f.py + "()\n\n");
    }

    void bindMap(const string& name, ExprP e, int line) {
        Info m; m.kind = MAP; m.py = bindPy(name); m.steps = 0; m.lr = -1;
        string fieldName;
        for (auto& a : e->args) {
            if (a.kw.empty() && a.e->k == Expr::Ident && names.count(a.e->id) &&
                names[a.e->id].kind == FIELD)
                fieldName = a.e->id;
            else if (a.kw.empty() && a.e->k == Expr::Call && a.e->id == "field") {
                fieldName = "_fld" + to_string(++anon_id);   // inline field(...)
                bindField(fieldName, a.e, line);
            }
            else if (a.kw == "steps" && a.e->k == Expr::Num) m.steps = (int)a.e->num;
            else if (a.kw == "lr" && a.e->k == Expr::Num) m.lr = a.e->num;
            else refuse(line, "flow argument `" + a.kw + "`");
        }
        if (fieldName.empty()) refuse(line, "this flow form");
        Info& f = names[fieldName];
        m.src = fieldName;
        if (f.fk == REGRESS || f.fk == COMBO) {
            m.mk = FMMAP;
            if (m.steps == 0) m.steps = 40;
            string vel;
            if (f.fk == REGRESS) vel = "net_" + f.py;
            else {
                vel = "_vel_" + m.py;
                string body;
                for (size_t i = 0; i < f.combo.size(); i++) {
                    Info& cf = info(f.combo[i].second, line);
                    if (cf.fk != REGRESS) refuse(line, "nested field algebra");
                    body += (i ? " + " : "") + pynum(f.combo[i].first) +
                            " * net_" + cf.py + "(inp)";
                }
                emit("def " + vel + "(inp):   # field algebra: " + fieldName + "\n"
                     "    return " + body + "\n\n");
            }
            m.velfn = vel;
            emit("@torch.no_grad()\n"
                 "def push_" + m.py + "(z, steps=" + to_string(m.steps) +
                 ", record=False):   # line " + to_string(line) +
                 ": flow(" + fieldName + ", steps=" + to_string(m.steps) + "), Euler\n"
                 "    zs = [z.clone()]\n"
                 "    ds = 1.0 / steps\n"
                 "    for i in range(steps):\n"
                 "        t = torch.full((1, z.shape[1]), i / steps)\n"
                 "        z = z + ds * " + vel + "(torch.cat([z, t], dim=0).T).T\n"
                 "        if record: zs.append(z.clone())\n"
                 "    return (z, zs) if record else z\n\n");
            emit("@torch.no_grad()\n"
                 "def push_" + m.py + "_inv(z, steps=" + to_string(m.steps) +
                 "):   # inv: reversed Euler (t = 1 - i/steps, z -= ds*v) — free\n"
                 "    ds = 1.0 / steps\n"
                 "    for i in range(steps):\n"
                 "        t = torch.full((1, z.shape[1]), 1 - i / steps)\n"
                 "        z = z - ds * " + vel + "(torch.cat([z, t], dim=0).T).T\n"
                 "    return z\n\n");
        } else {                       // NW field over a score-target descent
            m.mk = SVGDMAP;
            if (m.steps == 0 || m.lr < 0)
                refuse(line, "flow on a descent field without explicit steps= and lr=");
            used_svgd = true;
            Info& dsc = names[f.src];
            emitScore(dsc.src, line);
            string norm = dsc.normalize ? "True" : "False";
            emit("def push_" + m.py + "(z, record=False):   # line " + to_string(line) +
                 ": flow(" + fieldName + ", steps=" + to_string(m.steps) + ", lr=" +
                 pynum(m.lr) + ")\n"
                 "    # descent map (metric=" + (dsc.normalize ? "w2, normalized NW" : "stein, exact SVGD") +
                 "); hard-pinned to from=" + dsc.src2 + " in Liu\n"
                 "    return _svgd_run(z, score_" + names[dsc.src].py + ", " +
                 to_string(m.steps) + ", " + pynum(m.lr) + ", record, normalize=" +
                 norm + ")\n\n");
        }
        names[name] = m;
    }

    pair<string, string> seriesExpr(ExprP e, int line) {
        if (e->k == Expr::Ident && names.count(e->id) && names[e->id].kind == DATA)
            return { names[e->id].py, e->id };
        // prob(f0): the law of a frozen block is its empirical measure
        if (e->k == Expr::Call && e->id == "prob" && e->args.size() == 1 &&
            e->args[0].e->k == Expr::Ident) {
            Info& i = info(e->args[0].e->id, line);
            if (i.kind == DATA) return { i.py, "prob(" + e->args[0].e->id + ")" };
        }
        if (e->k == Expr::Sample && !e->via) {
            ExprP a = e->a;
            if (a->k == Expr::Ident && names.count(a->id) && names[a->id].kind == DIST)
                return { "sample_" + names[a->id].py + "(" + to_string(e->n) + ")",
                         a->id + " ~ " + to_string(e->n) };
            if (a->k == Expr::Ident && names.count(a->id) && names[a->id].kind == DATA) {
                used_batch = true;
                return { "_batch_from(" + names[a->id].py + ")(" + to_string(e->n) + ")",
                         a->id + " ~ " + to_string(e->n) };
            }
            // prob(x0) ~ n on a lazy coupling: the marginal — coupling never moves it
            if (a->k == Expr::Call && a->id == "prob" && a->args.size() == 1 &&
                a->args[0].e->k == Expr::Ident) {
                Info& i = info(a->args[0].e->id, line);
                if (i.kind == COUPLED && !i.marginal.empty())
                    return { samplerOf(i.marginal, to_string(e->n), line),
                             "prob(" + a->args[0].e->id + ") ~ " + to_string(e->n) };
                if (i.kind == DATA) {
                    used_batch = true;
                    return { "_batch_from(" + i.py + ")(" + to_string(e->n) + ")",
                             "prob(" + a->args[0].e->id + ") ~ " + to_string(e->n) };
                }
            }
            if (a->k == Expr::Push) return pushExpr(a, e->n, line);
        }
        refuse(line, "this plot series form");
    }

    string sourceExpr(ExprP b, int n, int line) {
        if (b->k == Expr::Ident && names.count(b->id))
            return samplerOf(b->id, to_string(n), line);
        refuse(line, "this pushforward source");
    }

    pair<string, string> pushExpr(ExprP p, int n, int line, bool record = false) {
        bool inv = false; ExprP mE = p->a;
        if (mE->k == Expr::Call && mE->id == "inv" && mE->args.size() == 1) {
            inv = true; mE = mE->args[0].e;
        }
        string mapName;
        if (mE->k == Expr::Ident && names.count(mE->id) && names[mE->id].kind == MAP)
            mapName = mE->id;
        else if (mE->k == Expr::Call && mE->id == "flow") {
            mapName = "_map" + to_string(++anon_id);   // inline flow(v, ...) # mu
            bindMap(mapName, mE, line);
        } else refuse(line, "this pushforward form");
        Info& m = names[mapName];
        if (inv && m.mk != FMMAP)
            refuse(line, "inv of a descent map (record= replay has no template; "
                         "inv is free only for trained FM flows)");
        string call = string("push_") + m.py + (inv ? "_inv" : "") + "(" +
                      sourceExpr(p->b, n, line) + (record ? ", record=True" : "") + ")";
        string label = (inv ? "inv(" + mapName + ")" : mapName) + " # " +
                       (p->b->k == Expr::Ident ? p->b->id : "?") + " ~ " + to_string(n);
        return { call, label };
    }

    void runStmts(const vector<Stmt>& prog) {
        for (auto& s : prog) {
            switch (s.k) {
            case Stmt::Seed:
                emit("torch.manual_seed(" + to_string(s.seed) + ")   # line " +
                     to_string(s.line) + "\n\n");
                break;
            case Stmt::For:
                for (int r = s.for_lo; r <= s.for_hi; r++) {
                    emit("# —— for round " + to_string(r) + " (line " +
                         to_string(s.line) + ") — macro expansion ——\n");
                    runStmts(s.body);
                }
                break;
            case Stmt::BindPair: {
                ExprP e = s.e;
                int eager_n = 0;
                if (e->k == Expr::Sample && !e->via) { eager_n = e->n; e = e->a; }
                if (e->k == Expr::Call && e->id == "rv" && e->args.size() == 1)
                    e = e->args[0].e;
                if (e->k == Expr::Call && e->id == "couple")
                    bindCouple(s, e, eager_n, s.line);
                else refuse(s.line, "this destructuring form (template: couple)");
                break;
            }
            case Stmt::Bind: {
                ExprP e = s.e;
                if (e->k == Expr::Mix ||
                    (e->k == Expr::Call && (e->id == "gaussian" || e->id == "moons" ||
                                            e->id == "uniform"))) {
                    emitDistDef(s.name, e, s.line);
                } else if (e->k == Expr::Sample && e->via) {
                    if (e->via->k != Expr::Call || e->via->id != "svgd" ||
                        e->a->k != Expr::Ident || !names.count(e->a->id))
                        refuse(s.line, "this via form");
                    Info& tgt = names[e->a->id];
                    int steps = 500; double lr = 0.5;
                    for (auto& a : e->via->args) {
                        if (a.kw == "steps" && a.e->k == Expr::Num) steps = (int)a.e->num;
                        else if (a.kw == "lr" && a.e->k == Expr::Num) lr = a.e->num;
                        else if (a.kw == "kernel") {}
                        else refuse(s.line, "via svgd argument `" + a.kw + "`");
                    }
                    used_svgd = true;
                    emitScore(e->a->id, s.line);
                    string py = bindPy(s.name);
                    emit("# line " + to_string(s.line) + ": " + s.name + " = " + e->a->id +
                         " ~ " + to_string(e->n) + " via svgd — sugar for descent from N(0, I), metric=stein\n");
                    emit(py + ", " + py + "_traj = _svgd_run(torch.randn(" +
                         to_string(tgt.dim) + ", " + to_string(e->n) + "), score_" +
                         tgt.py + ", " + to_string(steps) + ", " + pynum(lr) +
                         ", record=True)\n\n");
                    names[s.name] = { DATA, py, tgt.dim };
                    names[s.name].has_traj = true;
                } else if (e->k == Expr::Sample && !e->via && e->a->k == Expr::Call &&
                           e->a->id != "flow") {
                    string tmp = "_dist_" + s.name;
                    emitDistDef(tmp, e->a, s.line);
                    string py = bindPy(s.name);
                    names[s.name] = { DATA, py, names[tmp].dim };
                    emit(py + " = sample_" + names[tmp].py + "(" + to_string(e->n) +
                         ")   # line " + to_string(s.line) + ": frozen dataset\n\n");
                } else if (e->k == Expr::Sample && !e->via && e->a->k == Expr::Push) {
                    auto [expr, label] = pushExpr(e->a, e->n, s.line);
                    string py = bindPy(s.name);
                    emit(py + " = " + expr + "   # line " + to_string(s.line) +
                         ": " + label + "\n\n");
                    names[s.name] = { DATA, py };
                } else if (e->k == Expr::Push) {
                    ExprP mE = e->a;
                    string mapName;
                    if (mE->k == Expr::Call && mE->id == "flow") {
                        mapName = "_map_" + s.name;
                        bindMap(mapName, mE, s.line);
                    } else if (mE->k == Expr::Ident && names.count(mE->id) &&
                               names[mE->id].kind == MAP) {
                        mapName = mE->id;
                    } else refuse(s.line, "this pushforward binding");
                    if (e->b->k != Expr::Ident || !names.count(e->b->id) ||
                        names[e->b->id].kind != DATA)
                        refuse(s.line, "this pushforward source (binding form: # a frozen Dataset)");
                    string py = bindPy(s.name);
                    emit(py + ", " + py + "_traj = push_" + names[mapName].py + "(" +
                         names[e->b->id].py + ", record=True)   # line " +
                         to_string(s.line) + "\n\n");
                    names[s.name] = { DATA, py, names[e->b->id].dim };
                    names[s.name].has_traj = true;
                } else if (isFieldAlgebra(e) && e->k == Expr::Bin) {
                    Info f; f.kind = FIELD; f.fk = COMBO; f.py = bindPy(s.name);
                    comboWalk(e, 1.0, f.combo, s.line);
                    names[s.name] = f;
                } else if (e->k == Expr::Bin && e->id == "+") {
                    bindFormula(s.name, e, s.line);
                } else if (e->k == Expr::Call && e->id == "prob" && e->args.size() == 1 &&
                           e->args[0].e->k == Expr::Ident &&
                           names.count(e->args[0].e->id) &&
                           names[e->args[0].e->id].kind == FORMULA) {
                    names[s.name] = { PATH, bindPy(s.name) };
                    names[s.name].src = e->args[0].e->id;
                } else if (e->k == Expr::Call && e->id == "descent") {
                    Info d; d.kind = DESCENT; d.py = bindPy(s.name); d.normalize = true;
                    for (auto& a : e->args) {
                        if (a.kw.empty() && a.e->k == Expr::Call && a.e->id == "reverseKL" &&
                            a.e->args.size() == 1 && a.e->args[0].e->k == Expr::Ident)
                            d.src = a.e->args[0].e->id;
                        else if (a.kw == "from" && a.e->k == Expr::Ident) d.src2 = a.e->id;
                        else if (a.kw == "metric" && a.e->k == Expr::Ident &&
                                 a.e->id == "stein") d.normalize = false;
                        else if (a.kw == "metric" && a.e->k == Expr::Ident &&
                                 a.e->id == "w2") d.normalize = true;
                        else refuse(s.line, "descent argument `" + a.kw +
                                            "` (template: reverseKL score target, metric=stein/w2)");
                    }
                    if (d.src.empty() || d.src2.empty() || !names.count(d.src) ||
                        names[d.src].kind != DIST || names[d.src].comps.empty())
                        refuse(s.line, "this descent form (template: score targets — "
                                       "gaussian or gaussian mixture)");
                    names[s.name] = d;
                } else if (e->k == Expr::Call && e->id == "field") {
                    bindField(s.name, e, s.line);
                } else if (e->k == Expr::Call && e->id == "flow") {
                    bindMap(s.name, e, s.line);
                } else {
                    refuse(s.line, "this binding form");
                }
                break;
            }
            case Stmt::Plot: {
                emit("# line " + to_string(s.line) + ": plot\n");
                emit("plt.figure(figsize=(5, 5))\n");
                for (size_t i = 0; i < s.plots.size(); i++) {
                    ExprP pe = s.plots[i];
                    bool traj = s.is_traj[i];
                    if (traj && pe->k == Expr::Ident && names.count(pe->id) &&
                        names[pe->id].kind == DATA && names[pe->id].has_traj) {
                        used_traj = true;
                        emit("plot_traj(" + names[pe->id].py + "_traj)\n");
                        emit("plt.scatter(*" + names[pe->id].py + ", s=4, label='" +
                             pe->id + "')\n");
                        continue;
                    }
                    if (traj && pe->k == Expr::Sample && !pe->via && pe->a->k == Expr::Push) {
                        used_traj = true;
                        auto [expr, label] = pushExpr(pe->a, pe->n, s.line, /*record=*/true);
                        string v = "_s" + to_string(++plot_id);
                        emit(v + ", " + v + "_traj = " + expr + "\n");
                        emit("plot_traj(" + v + "_traj)\n");
                        emit("plt.scatter(*" + v + ", s=4, label='" + label + "')\n");
                        continue;
                    }
                    if (traj) refuse(s.line, "this trajectory form");
                    auto [expr, label] = seriesExpr(pe, s.line);
                    string v = "_s" + to_string(++plot_id);
                    emit(v + " = " + expr + "\n");
                    emit("plt.scatter(*" + v + ", s=4, alpha=0.6, label='" + label + "')\n");
                }
                emit("plt.legend(); plt.gca().set_aspect('equal')\n\n");
                break;
            }
            default:
                refuse(s.line, "this statement form");
            }
        }
    }

    void run(const vector<Stmt>& prog, const string& srcname) {
        emit("# Exported from " + srcname + " by `liu --export-pytorch`.\n"
             "# Semantically equivalent PyTorch mirror — NOT bit-identical to the\n"
             "# interpreter (different RNG streams/backend); the reproducibility\n"
             "# contract stays on the Liu side. Op structure follows the verified\n"
             "# bench mirror (bench/bench_torch.py) and svgd_field_at's assembly.\n"
             "import math\n"
             "import torch\n"
             "import matplotlib.pyplot as plt\n\n");
        runStmts(prog);
        emit("plt.show()\n");
        string helpers;
        if (used_ot)
            helpers += "from scipy.optimize import linear_sum_assignment  # couple(via=ot)\n\n";
        if (used_mlp)
            helpers +=
                "def _mlp(dims):   # Liu mlp: the input gains one time dimension\n"
                "    layers, ins = [], dims[0] + 1\n"
                "    for w in dims[1:-1]:\n"
                "        layers += [torch.nn.Linear(ins, w), torch.nn.ReLU()]; ins = w\n"
                "    layers += [torch.nn.Linear(ins, dims[-1])]\n"
                "    return torch.nn.Sequential(*layers)\n\n";
        if (used_batch)
            helpers +=
                "def _batch_from(X):   # frozen Dataset endpoint: minibatch its columns\n"
                "    return lambda B: X[:, torch.randint(0, X.shape[1], (B,))]\n\n";
        if (used_svgd)
            helpers +=
                "def _svgd_run(z, score_fn, steps, lr, record=False, normalize=False):\n"
                "    # reverseKL descent, mirroring svgd_field_at + svgd_bandwidth:\n"
                "    #  metric=stein (normalize=False): exact SVGD update, phi/n,\n"
                "    #    bandwidth med^2/log(n+1)  (Liu & Wang 2016)\n"
                "    #  metric=w2   (normalize=True): genuine NW — phi/kernel-mass,\n"
                "    #    bandwidth 2*med^2  (Gretton) — the consistent W2 velocity\n"
                "    zs = [z.clone()]\n"
                "    n = z.shape[1]\n"
                "    for _ in range(steps):\n"
                "        d2 = torch.cdist(z.T, z.T) ** 2\n"
                "        med = d2.flatten().median()\n"
                "        h = torch.clamp(2.0 * med if normalize else med / math.log(n + 1.0), min=1e-4)\n"
                "        K = torch.exp(-d2 / h)\n"
                "        num = score_fn(z) @ K + (z * K.sum(0) - z @ K) * (2.0 / h)\n"
                "        phi = num / K.sum(0) if normalize else num / n\n"
                "        z = z + lr * phi\n"
                "        if record: zs.append(z.clone())\n"
                "    return (z, zs) if record else z\n\n";
        if (used_traj)
            helpers +=
                "def plot_traj(zs, keep=60):   # trajectory: one line per particle\n"
                "    P = torch.stack(zs)                       # steps x d x n\n"
                "    for j in range(min(keep, P.shape[2])):\n"
                "        plt.plot(P[:, 0, j], P[:, 1, j], lw=0.5, alpha=0.4)\n\n";
        size_t at = out.find("import matplotlib.pyplot as plt\n\n");
        out.insert(at + strlen("import matplotlib.pyplot as plt\n\n"), helpers);
        fputs(out.c_str(), stdout);
    }
};

// ─────────────────────────────────────────────────────────── main ───
int compute() { return 0; }

#ifndef JUZHEN_NO_BLAS
extern "C" void openblas_set_num_threads(int);   // env is read in the library
                                                 // constructor, before main —
                                                 // only the API works here
#endif
int main(int argc, char** argv) {
    // The reproducibility contract (program text + seed + backend commit →
    // bit-identical output) must not depend on the host's core count:
    // multithreaded BLAS partitions reductions by thread count, and the
    // last-bit differences are amplified by training dynamics into different
    // outputs. Pin to one thread (also faster here: our matrices are small
    // enough that thread sync costs more than compute). An explicit
    // OPENBLAS_NUM_THREADS in the environment wins — that opt-out trades the
    // contract for parallelism, knowingly.
#ifndef JUZHEN_NO_BLAS
    if (!getenv("OPENBLAS_NUM_THREADS")) openblas_set_num_threads(1);
#endif
    setvbuf(stdout, nullptr, _IOLBF, 0);   // stream line-by-line even into pipes
    bool export_pt = argc >= 3 && string(argv[1]) == "--export-pytorch";
    const char* fname = export_pt ? argv[2] : (argc >= 2 ? argv[1] : nullptr);
    if (!fname) {
        printf("usage: liu <program.liu>\n"
               "       liu --export-pytorch <program.liu>   # PyTorch mirror on stdout\n");
        return 1;
    }
    ifstream fin(fname);
    if (!fin) { printf("cannot open %s\n", fname); return 1; }
    stringstream buf; buf << fin.rdbuf();
    if (export_pt) {
        try {
            Parser ps; ps.ts = lex(buf.str());
            auto prog = ps.program();
            PyExport().run(prog, fname);
            return 0;
        } catch (exception& e) {
            fprintf(stderr, "\n✗ export error:\n  %s\n", e.what());
            return 1;
        }
    }
    printf("Liu (\u6d41) v0.3+ prototype  ·  backend: Juzhen CPU\n");
    printf("═══════════════════════════════════════════\n");
    try {
        Parser ps; ps.ts = lex(buf.str());
        auto prog = ps.program();
        Interp in;
        in.run(prog);
    } catch (LiuError& e) {
        string site = site_text();       // lexer/parser messages carry their own "line N:"
        printf("\n\u2717 static check / type error:\n  %s%s\n",
               site.empty() ? "" : (site + ": ").c_str(), e.what());
        if (FILE* fp = dump_file()) {
            fprintf(fp, "{\"type\":\"error\",\"line\":%d,\"iter\":%s,\"text\":\"%s\"}\n",
                    cur_line, iter_json().c_str(), json_escape(e.what()).c_str());
            fflush(fp);
        }
        return 1;
    } catch (exception& e) {
        string site = site_text();
        printf("\n\u2717 runtime error: %s%s\n",
               site.empty() ? "" : (site + ": ").c_str(), e.what());
        if (FILE* fp = dump_file()) {
            fprintf(fp, "{\"type\":\"error\",\"line\":%d,\"iter\":%s,\"text\":\"%s\"}\n",
                    cur_line, iter_json().c_str(), json_escape(e.what()).c_str());
            fflush(fp);
        }
        return 1;
    }
    return 0;
}
