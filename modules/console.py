"""Small terminal styling helpers for startup and status output.

Colors are emitted only for interactive terminals, or when ``FORCE_COLOR`` is
set.  ``NO_COLOR`` always disables them.  Everything degrades to plain text
when the output is redirected, so logs and issue reports stay clean.
"""

import os
import sys

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


def bold(text):
    return style(text, "bold")


def dim(text):
    return style(text, "dim")


def cyan(text):
    return style(text, "cyan")


def green(text):
    return style(text, "green")


def yellow(text):
    return style(text, "yellow")


def red(text):
    return style(text, "red")


def banner():
    line = style("-" * 58, "dim")
    print()
    print(line)
    print(f" {style('SAKUGA', 'cyan', 'bold')}  {style('Stable Diffusion WebUI', 'dim')}")
    print(line)


def section(title):
    print()
    print(f"  {style('>', 'cyan', 'bold')} {style(f'{title}:', 'bold')}")


def kv(label, value, width=11, indent=2):
    label = f"{label}:"
    padding = " " * max(1, width - len(label))
    print(f"{' ' * indent}{style(label, 'dim')}{padding}{value}")


def detail(label, value, width=11):
    kv(label, style(value, "dim"), width=width)


def path(value):
    print(f"    {style(value, 'dim')}")


def command(args):
    if args:
        print(f"    {style(' '.join(args), 'dim')}")
    print()


def ok(message):
    print(f"  {style('OK:', 'green', 'bold')} {message}")


def hint(message):
    print(f"  {style('Hint:', 'yellow', 'bold')} {message}")


def warn(message):
    print(f"  {style('Warning:', 'yellow', 'bold')} {message}")


def error(message):
    print(f"  {style('Error:', 'red', 'bold')} {message}")


def ready(summary, local_url=None, share_url=None):
    total, _, details = summary.partition(" (")
    line = style("-" * 58, "dim")

    print()
    print(line)
    print(f"  {style('Ready', 'green', 'bold')} in {style(total, 'bold')}")
    if details:
        print(f"  {style('Timings:', 'dim')} {style(details.rstrip(')'), 'dim')}")
    if local_url:
        print()
        print(f"  {style('Running on local URL:', 'dim')}  {style(local_url.rstrip('/'), 'cyan', 'underline')}")
    if share_url:
        print(f"  {style('Public URL:', 'dim')}  {style(share_url, 'cyan', 'underline')}")
    print(line)
    print()