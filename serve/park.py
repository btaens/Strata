"""Warm park (opt-in, "warm": true / --warm): between requests the engine's GPU state is suspended with NVIDIA's
cuda-checkpoint instead of the process being ended.  The VRAM (the CUDA context included) goes back to the system, the
process keeps its RAM (the expert arena, its pinned pages, the CUDA graphs' host side) and the next request restores it.
Linux and an NVIDIA driver of 570 or newer; anything else keeps the plain unload."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

RAM_SPARE_MIB = 2048             # RAM kept free beside the suspended VRAM image


class ParkFailed(RuntimeError):
    """The engine could not be parked.  When it was started to be parked it has been ended: the model is unloaded."""


def find_checkpoint(configured: str | None = None) -> str | None:
    """The cuda-checkpoint program ("cuda_checkpoint" in the config, $STRATA_CUDA_CHECKPOINT, else PATH); None when there is none."""
    if not sys.platform.startswith("linux"):
        return None
    path = configured or os.environ.get("STRATA_CUDA_CHECKPOINT") or shutil.which("cuda-checkpoint")
    return path if path and os.path.isfile(path) and os.access(path, os.X_OK) else None


def run_checkpoint(binary: str, action: str, pid: int, timeout_s: float, lock_ms: int = 0) -> None:
    """`cuda-checkpoint --action lock|checkpoint|restore|unlock --pid N`; ParkFailed with its words when it fails."""
    cmd = [binary, "--action", action, "--pid", str(pid)]
    if action == "lock" and lock_ms > 0:
        cmd += ["--timeout", str(int(lock_ms))]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as e:
        raise ParkFailed(f"cuda-checkpoint {action} could not run ({e})") from e
    if r.returncode != 0:
        raise ParkFailed(f"cuda-checkpoint {action} failed (exit {r.returncode}): {(r.stderr or r.stdout).strip()[:300]}")


def ram_available_mib() -> int | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def process_vram_mib(pid: int) -> int | None:
    """The VRAM this process holds (nvidia-smi), or None when it cannot be told (a container's PID, no nvidia-smi)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    total, seen = 0, False
    for line in r.stdout.splitlines():
        p, _, used = line.partition(",")
        if p.strip() == str(pid) and used.strip().isdigit():
            total += int(used)
            seen = True
    return total if seen else None


def check_room(held_mib: int | None, free_mib: int | None, total_mib: int | None) -> None:
    """The suspended VRAM image is copied into RAM: ParkFailed when the RAM available cannot hold it.  `held_mib` is the
    engine's own VRAM when it can be told, else what the card reports used (it may count other programs: an upper bound)."""
    if held_mib is None and total_mib is not None and free_mib is not None:
        held_mib = max(0, total_mib - free_mib)
    room = ram_available_mib()
    if held_mib is None or room is None:
        return
    if held_mib + RAM_SPARE_MIB > room:
        raise ParkFailed(f"its {held_mib} MiB of VRAM would not fit in the {room} MiB of RAM available "
                         f"({RAM_SPARE_MIB} MiB are kept free)")
