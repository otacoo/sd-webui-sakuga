"""Small terminal styling helpers for startup and status output.

Colors are emitted only for interactive terminals, or when ``FORCE_COLOR`` is
set.  ``NO_COLOR`` always disables them.  Everything degrades to plain text
when the output is redirected, so logs and issue reports stay clean.
"""

import contextlib
import logging
import os
import sys
import threading

from rich.console import Console
from rich.table import Table
from rich.text import Text

_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "underline": "\033[4m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "gray": "\033[90m",
}


def _supports_color():
    if "NO_COLOR" in os.environ:
        return False

    force = os.environ.get("FORCE_COLOR")
    if force is not None:
        return force.lower() not in ("0", "false", "no", "off")

    if not getattr(sys.stdout, "isatty", lambda: False)():
        return False

    if os.name != "nt":
        return True

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


ENABLED = _supports_color()


def style(text, *names):
    text = str(text)
    if not ENABLED or not names:
        return text
    return "".join(_CODES[name] for name in names) + text + _CODES["reset"]


_console = Console(highlight=False, emoji=False, soft_wrap=False)


class _IndentedStream:
    def __init__(self, stream):
        self._stream = stream
        self._at_line_start = True
        self._lock = threading.Lock()

    def write(self, text):
        if not text:
            return 0
        prefix = _indent_prefix
        with self._lock:
            parts = []
            for line in text.splitlines(keepends=True):
                if self._at_line_start and prefix and line not in ("\n", "\r", "\r\n"):
                    parts.append(prefix)
                parts.append(line)
                self._at_line_start = line.endswith(("\n", "\r"))
            self._stream.write("".join(parts))
        return len(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


_indent_prefix = ""


def _stream_handlers():
    loggers = [logging.getLogger()]
    loggers.extend(obj for obj in logging.root.manager.loggerDict.values() if isinstance(obj, logging.Logger))
    seen = set()
    for logger in loggers:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and id(handler) not in seen:
                seen.add(id(handler))
                yield handler


@contextlib.contextmanager
def indent_output(prefix="    "):
    global _indent_prefix
    previous_prefix = _indent_prefix
    _indent_prefix = prefix

    stdout, stderr = sys.stdout, sys.stderr
    out = _IndentedStream(stdout)
    err = _IndentedStream(stderr)

    changed = []
    for handler in _stream_handlers():
        stream = getattr(handler, "stream", None)
        if stream is stdout:
            handler.setStream(out)
            changed.append((handler, stdout))
        elif stream is stderr:
            handler.setStream(err)
            changed.append((handler, stderr))

    sys.stdout, sys.stderr = out, err
    try:
        yield
    finally:
        out.flush()
        err.flush()
        sys.stdout, sys.stderr = stdout, stderr
        for handler, stream in changed:
            handler.setStream(stream)
        _indent_prefix = previous_prefix


def banner():
    _console.print()
    title = Text.assemble((" SAKUGA", "bold cyan"), ("  Stable Diffusion WebUI ", "dim"))
    _console.rule(title, align="left", style="dim", characters="-")
    _console.print()


def section(title):
    _console.print()
    text = Text("  ")
    text.append(">", style="bold cyan")
    text.append(" ")
    text.append(title, style="bold")
    _console.print(text)


def kv(label, value, width=14, indent=2):
    label = f"{label}:"
    text = Text(" " * indent)
    text.append(label, style="dim")
    text.append(" " * max(1, width - len(label) + 1))
    text.append(str(value))
    _console.print(text)


def url(local_url=None, share_url=None):
    if not local_url and not share_url:
        return
    _console.print()
    if local_url:
        text = Text("  ")
        text.append("Running on local URL:", style="dim")
        text.append("  ")
        text.append(local_url.rstrip("/"), style="cyan underline")
        _console.print(text)
    if share_url:
        text = Text("  ")
        text.append("Public URL:", style="dim")
        text.append("  ")
        text.append(share_url, style="cyan underline")
        _console.print(text)


def command(args):
    if args:
        _console.print(Text("    " + " ".join(args), style="dim"))
    _console.print()


def hint(message):
    text = Text("  ")
    text.append("Hint:", style="bold yellow")
    text.append(" ")
    text.append(str(message))
    _console.print(text)


def warn(message):
    text = Text("  ")
    text.append("Warning:", style="bold yellow")
    text.append(" ")
    text.append(str(message))
    _console.print(text)


def ready(summary, local_url=None, share_url=None):
    total, _, details = summary.partition(" (")
    _console.print()
    _console.rule(style="dim", characters="-")

    text = Text("  ")
    text.append("Ready", style="bold green")
    text.append(" in ")
    text.append(total, style="bold")
    _console.print(text)

    if details:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim", no_wrap=True)
        table.add_column(style="dim", overflow="fold")
        table.add_row("  Timings:", details.rstrip(")"))
        _console.print(table)

    if local_url or share_url:
        url(local_url, share_url)

    _console.rule(style="dim", characters="-")
    _console.print()
