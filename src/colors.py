##
# @file src/colors.py
# @brief Shared colour constants and ANSI terminal helpers.
##

HEX_BLUE   = "#1565C0"
HEX_GREEN  = "#2E7D32"
HEX_RED    = "#D32F2F"
HEX_AMBER  = "#F59E0B"
HEX_DARK   = "#0D1117"
HEX_PURPLE = "#6A1B9A"

HEX_BG_BLUE  = "rgba(21,101,192,0.15)"
HEX_BG_GREEN = "rgba(46,125,50,0.15)"
HEX_BG_RED   = "rgba(211,47,47,0.15)"
HEX_BG_AMBER = "rgba(245,158,11,0.15)"

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"


def _wrap(code: str, text: str, bold: bool = False) -> str:
    """
    @brief Wrap text in an ANSI colour code with an optional bold prefix.

    @param code  ANSI escape code for the desired colour.
    @param text  Text to wrap.
    @param bold  If True, prepend the bold escape code.
    @return      Formatted string with ANSI codes and a reset suffix.
    """
    prefix = _BOLD if bold else ""
    return f"{prefix}{code}{text}{_RESET}"


def blue(text: str, bold: bool = False) -> str:
    """@brief Return text formatted in blue."""
    return _wrap("\033[34m", text, bold)


def green(text: str, bold: bool = False) -> str:
    """@brief Return text formatted in green."""
    return _wrap("\033[32m", text, bold)


def red(text: str, bold: bool = False) -> str:
    """@brief Return text formatted in red."""
    return _wrap("\033[31m", text, bold)


def amber(text: str, bold: bool = False) -> str:
    """@brief Return text formatted in amber/yellow."""
    return _wrap("\033[33m", text, bold)


def dim(text: str) -> str:
    """@brief Return text formatted in dim style."""
    return f"{_DIM}{text}{_RESET}"


def step(n: int, title: str) -> None:
    """@brief Print a numbered step header followed by a horizontal rule."""
    print(f"\n{blue(f'[{n}]', bold=True)} {title}")
    print("─" * 50)
