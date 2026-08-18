"""Host-platform helpers for tools that spawn NInfer build-tree binaries.

Build-tree binaries carry a ``.exe`` suffix on Windows, and a clean server
shutdown there needs a console control event: ``Popen.terminate()`` is
``TerminateProcess``, which never runs ninfer-serve's ``SetConsoleCtrlHandler``
path, so the request log would end without its final records. Runners
therefore start the server in its own process group and stop it with
``CTRL_BREAK_EVENT``.

Binary paths assume a single-config generator (Ninja, the documented choice).
A Visual Studio generator would place binaries under a per-config
subdirectory such as ``build/apps/Release/`` and is not supported here.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

WINDOWS = os.name == "nt"

# Creation flags for server processes that must receive CTRL_BREAK_EVENT:
# the event is delivered per process group, so the server gets its own.
SERVER_POPEN_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP if WINDOWS else 0


def binary_path(relative: str) -> Path:
    """Resolve a repo-relative build-tree binary, adding ``.exe`` on Windows."""
    path = REPO_ROOT / relative
    return path.with_suffix(".exe") if WINDOWS else path


def request_stop(process: subprocess.Popen) -> None:
    """Ask a server process to shut down cleanly.

    POSIX: SIGTERM. Windows: CTRL_BREAK_EVENT (requires the process to have
    been started with SERVER_POPEN_FLAGS). Callers keep their existing
    wait-then-kill escalation.
    """
    if WINDOWS:
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.terminate()


def format_command(command: Sequence[object]) -> str:
    """Render a command list as a reproducible shell line for this platform."""
    parts = [str(part) for part in command]
    if WINDOWS:
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)
