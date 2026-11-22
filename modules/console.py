"""Small terminal styling helpers for startup and status output.

Colors are emitted only for interactive terminals, or when ``FORCE_COLOR`` is
set.  ``NO_COLOR`` always disables them.  Everything degrades to plain text
when the output is redirected, so logs and issue reports stay clean.
"""

import os
import sys
import textwrap

from rich.console import Console
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


def banner():
    _console.print()
    _console.print(Text("-" * 58, style="dim"))
    _console.print(Text.assemble((" SAKUGA", "bold cyan"), ("  Stable Diffusion WebUI", "dim")))
    _console.print(Text("-" * 58, style="dim"))


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
        _console.print(Text("  " + " ".join(args), style="dim"))
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
    _console.print(Text("-" * 58, style="dim"))

    text = Text("  ")
    text.append("Ready", style="bold green")
    text.append(" in ")
    text.append(total, style="bold")
    _console.print(text)

    if details:
        lines = textwrap.wrap(details.rstrip(")"), width=47) or [""]
        row = Text("  ")
        row.append("Timings:", style="dim")
        row.append(" ")
        row.append(lines[0], style="dim")
        _console.print(row)
        for extra in lines[1:]:
            row = Text(" " * 11)
            row.append(extra, style="dim")
            _console.print(row)

    if local_url or share_url:
        url(local_url, share_url)

    _console.print(Text("-" * 58, style="dim"))
    _console.print()
