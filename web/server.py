#!/usr/bin/env python3
"""Liu playground server — stdlib only.

Runs Liu programs submitted from the browser and streams the interpreter's
output back as newline-delimited JSON. The language itself is the sandbox
(no I/O primitives, not Turing complete); this server adds the process-level
guards the prototype does not enforce: wall-clock timeout, address-space
limit, program-size limit, and a concurrency cap. Determinism (program text
+ seed decides the output bit for bit) makes whole-run caching sound.

Usage:  python3 web/server.py [--port 8080]   (run from the repository root)
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import resource
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIU = os.path.join(ROOT, "build_liu", "liu")


def _envint(name, default):
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# Limits are env-overridable so a public deployment can tighten them below the
# generous local-dev defaults (see Dockerfile) without editing this file.
MAX_PROGRAM_BYTES = _envint("LIU_MAX_PROGRAM_BYTES", 16 * 1024)
TIMEOUT_SECONDS = _envint("LIU_TIMEOUT_SECONDS", 90)
MEMORY_BYTES = _envint("LIU_MEMORY_BYTES", 8 * 1024 ** 3)
MAX_CONCURRENT = _envint("LIU_MAX_CONCURRENT", 4)
CACHE_MAX_ENTRIES = _envint("LIU_CACHE_MAX_ENTRIES", 128)

# Optional shared password. When LIU_PASSWORD is set, every request needs HTTP
# Basic Auth (any username, this password) — keeps internet scanners off /run
# on a public box. Unset (local dev) leaves the server open as before.
PASSWORD = os.environ.get("LIU_PASSWORD", "")

_sem = threading.Semaphore(MAX_CONCURRENT)
_cache = {}          # sha256(program) -> list[dict]  (the full event stream)
_cache_lock = threading.Lock()


def _limits():
    # best effort; e.g. RLIMIT_AS is unsupported on macOS
    for lim, val in ((resource.RLIMIT_AS, (MEMORY_BYTES,) * 2),
                     (resource.RLIMIT_CPU, (TIMEOUT_SECONDS + 30,) * 2)):
        try:
            resource.setrlimit(lim, val)
        except (ValueError, OSError):
            pass


def run_program(program: str):
    """Yield event dicts for one run: line / loss / plot / error / done.

    The interpreter streams structured events (loss points, plot data,
    errors — each tagged with its source line and iteration stack) into
    the LIU_DUMP file, flushed per row; we tail that file while the
    process runs so loss curves and plots reach the browser live, not at
    exit. Interpreter stdout is forwarded verbatim as `line` events.
    """
    with tempfile.TemporaryDirectory(prefix="liu_") as td:
        src = os.path.join(td, "program.liu")
        dump = os.path.join(td, "plots.jsonl")
        with open(src, "w") as f:
            f.write(program)

        dump_off = 0

        def drain():
            """New complete rows from the dump file since the last call."""
            nonlocal dump_off
            try:
                with open(dump, "rb") as f:
                    f.seek(dump_off)
                    chunk = f.read()
            except OSError:
                return
            end = chunk.rfind(b"\n")
            if end < 0:
                return
            dump_off += end + 1
            for row in chunk[:end].decode("utf-8", "replace").split("\n"):
                row = row.strip()
                if not row:
                    continue
                try:
                    ev = json.loads(row)
                except ValueError:
                    continue
                if ev.get("type") in ("loss", "error"):
                    yield ev
                else:                      # scatter / traj
                    yield {"type": "plot", "plot": ev}

        # single-threaded BLAS: keeps bit-for-bit determinism and avoids the
        # large per-thread address-space reservations that trip RLIMIT_AS on
        # many-core machines (a classic silent-death cause).
        env = dict(os.environ, LIU_DUMP=dump,
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        # the interpreter line-buffers its own stdout (setvbuf in main)
        proc = subprocess.Popen(
            [LIU, src],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env, cwd=ROOT, preexec_fn=_limits,
        )
        killer = threading.Timer(TIMEOUT_SECONDS, proc.kill)
        killer.start()
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line.startswith("profiler:"):
                    continue
                yield {"type": "line", "text": line}
                yield from drain()
            code = proc.wait()
        finally:
            killer.cancel()
        if code == -9:
            yield {"type": "line",
                   "text": f"✗ killed: exceeded the {TIMEOUT_SECONDS}s wall-clock limit"}
        yield from drain()
        yield {"type": "done", "code": code}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"      # stream until close; no chunked framing

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.address_string(), fmt % args))

    def _authed(self):
        """True if the request may proceed. When no password is configured the
        server is open; otherwise require HTTP Basic Auth (any user, matching
        password). On failure, send 401 and return False so the caller stops."""
        if not PASSWORD:
            return True
        hdr = self.headers.get("Authorization", "")
        if hdr.startswith("Basic "):
            try:
                supplied = base64.b64decode(hdr[6:]).decode("utf-8", "replace")
            except Exception:
                supplied = ""
            _, _, pw = supplied.partition(":")
            if hmac.compare_digest(pw, PASSWORD):
                return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Liu playground"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def _send_file(self, path, ctype):
        try:
            with open(os.path.join(ROOT, path), "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")   # never serve a stale editor
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._authed():
            return
        if self.path in ("/", "/index.html"):
            self._send_file("web/playground.html", "text/html; charset=utf-8")
        elif self.path == "/examples":
            self._send_examples()
        else:
            self.send_error(404)

    def _send_examples(self):
        """The example gallery's source of truth: examples/*.liu,
        each with its first comment line as the one-line description."""
        exdir = os.path.join(ROOT, "examples")
        items = []
        try:
            names = sorted(os.listdir(exdir))
        except OSError:
            names = []
        for fn in names:
            if not fn.endswith(".liu"):
                continue
            try:
                with open(os.path.join(exdir, fn), encoding="utf-8") as f:
                    code = f.read()
            except OSError:
                continue
            desc = ""
            for line in code.splitlines():
                line = line.strip()
                if line.startswith("//"):
                    desc = line.lstrip("/").strip()
                    break
                if line:
                    break
            items.append({"file": fn, "desc": desc, "code": code})
        body = json.dumps(items).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self._authed():
            return
        if self.path != "/run":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_PROGRAM_BYTES:
            self.send_error(413, "program too large")
            return
        program = self.rfile.read(n).decode("utf-8", "replace")

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(ev):
            self.wfile.write((json.dumps(ev) + "\n").encode())
            self.wfile.flush()

        key = hashlib.sha256(program.encode()).hexdigest()
        with _cache_lock:
            cached = _cache.get(key)
        if cached is not None:
            emit({"type": "line", "text": "(cached run — determinism makes this exact)"})
            for ev in cached:
                emit(ev)
            return

        if not _sem.acquire(blocking=False):
            emit({"type": "line", "text": "✗ server busy: too many concurrent runs, try again shortly"})
            emit({"type": "done", "code": 1})
            return
        events = []
        try:
            for ev in run_program(program):
                events.append(ev)
                emit(ev)
        except BrokenPipeError:
            return          # client went away; finish silently, don't cache partial
        except Exception as e:          # surface server-side failures to the page
            try:
                emit({"type": "line", "text": "\u2717 server error: %r" % e})
                emit({"type": "done", "code": 1})
            except BrokenPipeError:
                pass
            return
        finally:
            _sem.release()
        if events and events[-1].get("code") == 0:
            with _cache_lock:
                if len(_cache) >= CACHE_MAX_ENTRIES:
                    _cache.pop(next(iter(_cache)))
                _cache[key] = events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()
    if not os.path.exists(LIU):
        sys.exit("interpreter not found at %s — run ./interpreter/build.sh first" % LIU)
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    print("Liu playground on http://%s:%d  (interpreter: %s)" % (args.bind, args.port, LIU))
    srv.serve_forever()


if __name__ == "__main__":
    main()
