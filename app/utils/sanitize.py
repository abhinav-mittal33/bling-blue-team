"""
app/utils/sanitize.py — Log injection sanitization (P1-5).

Strips ANSI escape sequences and control characters from values before
they reach structlog. Prevents log injection / terminal hijacking attacks
via attacker-controlled transaction fields (payee_vpa, transaction_id, etc.).

Usage:
    from app.utils.sanitize import sanitize_for_log
    log.info("event", payee=sanitize_for_log(txn.payee_vpa))
"""
from __future__ import annotations
import re

# ANSI escape sequences (color codes, cursor movement) + C0/C1 control characters
_CONTROL_CHARS = re.compile(
    r"[\x00-\x1f\x7f\x9b-\x9f]"   # C0 controls + DEL + C1 controls
    r"|\x1b\[[0-9;]*[mGKHFABCDJMPR]"  # CSI sequences (ANSI color + movement)
    r"|\x1b[()#%]?"                 # other ESC sequences
)
_MAX_LOG_LEN = 500


def sanitize_for_log(value: object) -> str:
    """
    Strip control characters and ANSI escape sequences. Truncate to 500 chars.
    Safe to call with any type — converts to string first.
    """
    cleaned = _CONTROL_CHARS.sub("", str(value))
    return cleaned[:_MAX_LOG_LEN]
