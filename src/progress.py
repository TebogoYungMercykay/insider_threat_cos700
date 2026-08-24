##
# @file src/progress.py
# @brief Shared terminal progress-reporting helpers.
#
# Every script in this project (script/dataset.py, train.py, and the library
# functions they call in src/preprocess.py, src/baseline.py, src/explain.py)
# uses this module so that a person watching the terminal always knows what
# is currently happening, instead of staring at a blank line during a
# multi-second-to-multi-minute operation. Three building blocks are provided:
#
#   - step / skip / ok / warn / err / kv / divider / header
#         Numbered-step / key-value console formatting, shared by
#         script/dataset.py and train.py so both pipelines look and behave
#         the same way.
#
#   - Progress
#         A live redrawing progress bar for operations with a known total
#         (bytes read, rows processed, files extracted, ...).
#
#   - substep(label)
#         A context manager for a short, named sub-operation inside a step
#         (e.g. "loading logon.csv") — prints the label immediately (so the
#         person sees it start) and appends elapsed time + an optional
#         detail string when it finishes.
#
#   - Heartbeat(label)
#         A context manager for a blocking call whose duration can't be
#         measured incrementally (e.g. a scikit-learn .fit() call). Prints a
#         "still running (Ns elapsed)" line on a background thread every few
#         seconds *only if* the call actually runs long enough to need it —
#         fast calls finish before the first tick and print nothing extra.
##

import sys
import threading
import time

from src.colors import blue, green, red, amber, dim

_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_RESET = "\033[0m"

_PROGRESS_INTERVAL_S = 0.5   # redraw throttle for interactive (TTY) output
_HEARTBEAT_INTERVAL_S = 3.0  # default tick interval for Heartbeat


def _is_tty() -> bool:
    return sys.stdout.isatty()


def human_size(n: float) -> str:
    """@brief Format a byte count as a short human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


# ---------------------------------------------------------------------------
# Step / key-value console formatting (shared by script/dataset.py, train.py)
# ---------------------------------------------------------------------------

def divider(char: str = "─", width: int = 70, color_fn=None) -> None:
    print((color_fn or blue)(char * width))


def header(title: str, width: int = 70) -> None:
    divider("═", width)
    print(blue(_BOLD + f"  {title}" + _RESET))
    divider("═", width)
    print()


def step(n: int, title: str) -> None:
    print()
    print(blue(_BOLD + f">> Step {n}: {title}" + _RESET))
    divider("─", 70)


def skip(n: int, title: str, reason: str = "cache hit") -> None:
    print(f"   {dim(f'Step {n}: {title}')}  {green(f'[{reason}]')}")


def ok(label: str) -> None:
    print(f"  {green(chr(0x2713), bold=True)}  {label}")


def warn(text: str) -> None:
    print(f"  {amber(chr(0x26a0))}  {amber(text)}")


def err(text: str) -> None:
    print(f"  {red(chr(0x2717))}  {red(text)}")


def kv(label: str, value, width: int = 30) -> None:
    print(f"{_DIM}    {label}".ljust(width + 4) + f"{_RESET} {_BOLD}{value}{_RESET}")


def elapsed_str(t0: float) -> str:
    s = time.perf_counter() - t0
    return amber(f"{s:.1f}s") if s < 60 else amber(f"{s/60:.1f}min")


# ---------------------------------------------------------------------------
# Progress — live bar for operations with a known total
# ---------------------------------------------------------------------------

class Progress:
    """
    @brief Live, single-line progress reporter for a long-running operation
           with a known total (bytes streamed, rows processed, files
           extracted, ...).

    On an interactive terminal this redraws in place via a carriage return,
    throttled to avoid spamming fast operations. On non-interactive output
    (redirected to a file/pipe, e.g. inside CI logs), carriage returns don't
    redraw, so instead of time-throttling it prints one line per 10%
    milestone crossed — visible progress without flooding the log.
    """

    def __init__(self, label: str, total: int, unit: str = "bytes"):
        """
        @param label  Short description shown alongside the bar.
        @param total  Expected total (bytes, rows, items, ...). 0 disables
                       percentage/bar rendering but counts still print.
        @param unit   "bytes" formats counts via human_size(); anything else
                      (e.g. "rows", "items") formats as a plain integer.
        """
        self.label = label
        self.total = total
        self.unit = unit
        self.written = 0
        self.start = time.time()
        self._last_draw = 0.0
        self._last_pct_drawn = -1
        self._is_tty = _is_tty()

    def _fmt(self, n: float) -> str:
        return human_size(n) if self.unit == "bytes" else f"{n:,.0f}"

    def _draw(self) -> None:
        elapsed = max(time.time() - self.start, 1e-9)
        rate = self.written / elapsed
        pct = (self.written / self.total * 100) if self.total else 100.0
        bar_len = 24
        filled = min(bar_len, int(bar_len * pct / 100)) if self.total else bar_len
        bar = "#" * filled + "-" * (bar_len - filled)
        rate_str = f"{human_size(rate)}/s" if self.unit == "bytes" else f"{rate:,.0f}/s"
        totals = f"{self._fmt(self.written)}/{self._fmt(self.total)}" if self.total else self._fmt(self.written)
        line = f"  {dim(self.label)} [{blue(bar)}] {pct:5.1f}%  {totals}  {rate_str}"
        if self._is_tty:
            sys.stdout.write("\r\033[K" + line)
        else:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def update(self, n: int) -> None:
        """@brief Advance the counter by n and redraw if it's time to."""
        self.written += n
        if self._is_tty:
            now = time.time()
            if now - self._last_draw < _PROGRESS_INTERVAL_S and self.written < self.total:
                return
            self._last_draw = now
            self._draw()
        else:
            pct = int(self.written / self.total * 100) if self.total else 100
            milestone = (pct // 10) * 10
            if milestone == self._last_pct_drawn and self.written < self.total:
                return
            self._last_pct_drawn = milestone
            self._draw()

    def done(self) -> None:
        """@brief Clear the live line (TTY) once the operation is complete."""
        if self._is_tty:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# substep — named sub-operation with a start message and an elapsed summary
# ---------------------------------------------------------------------------

class _SubStep:
    def __init__(self, label: str):
        self.label = label
        self.t0 = None
        self.detail = None
        self._tty = _is_tty()

    def __enter__(self) -> "_SubStep":
        self.t0 = time.time()
        sys.stdout.write(f"  {blue(chr(0x2192))} {self.label} ...")
        if not self._tty:
            sys.stdout.write("\n")
        sys.stdout.flush()
        return self

    def set_detail(self, text: str) -> None:
        """@brief Attach a short result summary shown when the step completes."""
        self.detail = text

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.time() - self.t0
        if exc_type is not None:
            msg = f"  {red(chr(0x2717))} {self.label} — failed after {elapsed:.1f}s ({exc})"
            (sys.stdout.write("\r\033[K" + msg + "\n") if self._tty else sys.stdout.write(msg + "\n"))
            sys.stdout.flush()
            return False
        suffix = f" — {self.detail}" if self.detail else ""
        msg = f"  {green(chr(0x2713))} {self.label} ({elapsed:.1f}s){suffix}"
        (sys.stdout.write("\r\033[K" + msg + "\n") if self._tty else sys.stdout.write(msg + "\n"))
        sys.stdout.flush()
        return False


def substep(label: str) -> _SubStep:
    """
    @brief Context manager for a short, named sub-operation.

    Usage:
        with substep("loading logon.csv") as s:
            df = pd.read_csv(...)
            s.set_detail(f"{len(df):,} rows")
    """
    return _SubStep(label)


# ---------------------------------------------------------------------------
# Heartbeat — periodic "still running" ticks for opaque blocking calls
# ---------------------------------------------------------------------------

class Heartbeat:
    """
    @brief Context manager that prints a periodic "still running" line on a
           background thread while a blocking call of unknown duration runs
           (e.g. sklearn's .fit(), SHAP's shap_values()).

    The point isn't to show real progress (there isn't any to show for these
    calls) — it's to make the difference between "still computing" and
    "hung" visible. Fast calls that finish before the first tick print
    nothing beyond their normal start/stop messages; only genuinely
    long-running calls grow the extra ticks.
    """

    def __init__(self, label: str, interval: float = _HEARTBEAT_INTERVAL_S):
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None
        self._ticked = False

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            elapsed = time.time() - self._t0
            self._ticked = True
            line = f"  {dim(f'... {self.label} still running ({elapsed:.0f}s elapsed)')}"
            if _is_tty():
                sys.stdout.write("\r\033[K" + line)
            else:
                sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def __enter__(self) -> "Heartbeat":
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self._ticked and _is_tty():
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
        return False
