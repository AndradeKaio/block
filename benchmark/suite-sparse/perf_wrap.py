#!/usr/bin/env python3
"""
suite-sparse/perf_wrap.py — wrap a benchmark subprocess with `perf stat` +
`/usr/bin/time -v` to collect hardware counters and peak RSS.

Shared by benchmark_spgemm_cpu.py / benchmark_spmm_cpu.py / benchmark_spmv_cpu.py
behind their `--perf` flag.

Caveats (same for every domain that uses this):
  - perf stat measures the whole wrapped process's lifetime -- process
    startup, the full symbolic phase, and every internal run inside a single
    `--runs N` invocation -- not a single timed region. Every kernel here
    times N runs inside one process call, so the counters below are one
    aggregate reading for that whole call, not a per-run_id breakdown. Every
    CSV row emitted from one subprocess call carries the same aggregate
    reading.
  - On a `subprocess.run(..., timeout=...)` timeout, Python kills only the
    direct child (`time`), not the `perf`/target grandchildren it spawned --
    with --perf enabled a timeout can leave orphaned processes running
    briefly. Not handled here; keep an eye out during long --perf sweeps.
  - perf_event_open can be blocked entirely by container/seccomp policy
    (confirmed the case in this sandbox) regardless of perf_event_paranoid --
    in that case every event comes back as NaN. Check the raw stderr of a
    single manual run if a whole column is unexpectedly all-NaN.
"""

import glob
import os
import re
import shutil
import subprocess

PERF_EVENTS = [
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "branches",
    "branch-misses",
    "dTLB-loads",
    "dTLB-load-misses",
    "iTLB-loads",
    "iTLB-load-misses",
]


def _col(event: str) -> str:
    return event.replace("-", "_").replace("dTLB", "dtlb").replace("iTLB", "itlb").lower()


PERF_CSV_FIELDS = [_col(e) for e in PERF_EVENTS] + ["peak_rss_kb"]

_EVENT_COL = {e: _col(e) for e in PERF_EVENTS}


def _sanity_check(binary: str) -> bool:
    try:
        r = subprocess.run([binary, "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except OSError:
        return False


def _find_perf_bin() -> str:
    """Prefer $PRISMA_PERF_BIN, else the first working `perf` on PATH, else
    fall back to searching /usr/lib/linux-tools-*/perf -- needed because
    Ubuntu's /usr/bin/perf is a version-lookup shim that fails outright on a
    kernel it doesn't recognise (e.g. a custom/VM kernel flavor), even though
    a real perf binary is installed right next to it."""
    env = os.environ.get("PRISMA_PERF_BIN")
    if env:
        return env
    candidates = [shutil.which("perf")]
    candidates += sorted(glob.glob("/usr/lib/linux-tools-*/perf"), reverse=True)
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK) and _sanity_check(c):
            return c
    return "perf"


def _find_time_bin() -> str:
    return os.environ.get("PRISMA_TIME_BIN") or shutil.which("time") or "/usr/bin/time"


PERF_BIN = _find_perf_bin()
TIME_BIN = _find_time_bin()


def wrap_cmd(cmd: list[str], enable: bool) -> list[str]:
    """Prepend `time -v perf stat -x, -e ...` to cmd when enable is True.

    Both tools report to stderr only, so the wrapped binary's own stdout
    (where every domain's own JSON/text timing output lives) passes through
    untouched, and both `perf stat` and `time` propagate the wrapped
    command's exit status as their own -- existing returncode checks still
    work unchanged.
    """
    if not enable:
        return cmd
    return [
        TIME_BIN,
        "-v",
        PERF_BIN,
        "stat",
        "-x,",
        "-e",
        ",".join(PERF_EVENTS),
        "--",
    ] + cmd


_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
_NAN = float("nan")


def empty_metrics() -> dict:
    return {f: _NAN for f in PERF_CSV_FIELDS}


def parse(stderr_text: str) -> dict:
    """Parse combined `perf stat -x,` + `time -v` stderr into PERF_CSV_FIELDS.
    Missing/unsupported/uncounted individual events are left as NaN.

    If perf can't enable ANY event (e.g. perf_event_open blocked by
    container/seccomp policy, confirmed the case in this dev sandbox), it
    aborts before ever exec'ing the target -- no per-event CSV lines appear
    at all, and time -v then reports its own (perf's) tiny process footprint
    as "Maximum resident set size", which would masquerade as the target
    binary's memory use. Guard against that: only trust the RSS reading (or
    any counter) if we saw at least one real per-event CSV line, proving the
    target actually ran.
    """
    metrics = empty_metrics()
    if not stderr_text:
        return metrics

    saw_any_event_line = False
    for line in stderr_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) < 3:
            continue
        value_str = fields[0].strip()
        # A real perf -x, data line's first field is always a bare number or
        # a "<not counted>"/"<not supported>" placeholder. Anything else
        # (e.g. `time -v`'s "Command being timed: ...-e cycles,instructions,
        # ..." line, which also happens to comma-split into fields that
        # collide with our event names) is not actual perf output and must
        # be rejected before it's allowed to count as evidence perf ran.
        is_value_like = bool(re.fullmatch(r"[0-9.]+", value_str)) or (
            value_str.startswith("<") and value_str.endswith(">")
        )
        if not is_value_like:
            continue
        col = None
        for f in fields[1:]:
            col = _EVENT_COL.get(f.strip())
            if col:
                break
        if col is None:
            continue
        saw_any_event_line = True
        if "<" in value_str:
            continue  # "<not counted>" / "<not supported>"
        try:
            metrics[col] = float(value_str)
        except ValueError:
            continue

    if not saw_any_event_line:
        return metrics  # perf never actually ran the target -- RSS stays NaN too

    m = _RSS_RE.search(stderr_text)
    if m:
        metrics["peak_rss_kb"] = float(m.group(1))

    return metrics


def measure(cmd: list[str], timeout: int | None = None) -> dict:
    """Best-effort hardware-counter + peak-RSS measurement of cmd.

    Runs cmd a second time wrapped in `time -v perf stat`, entirely separate
    from the caller's own (unwrapped) timed invocation -- so a perf failure
    (missing permissions, no perf installed, timeout) never affects the
    actual benchmark result, it only leaves these metrics as NaN.
    """
    wrapped = wrap_cmd(cmd, True)
    try:
        r = subprocess.run(wrapped, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return empty_metrics()
    return parse(r.stderr)
