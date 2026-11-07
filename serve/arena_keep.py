"""Keep the experts in system RAM while the model is unloaded from the GPU (opt-in).

Why: `POST /unload` and an idle unload stop the engine, which is how Strata gives the GPU back to a game or another
server (serve/server.py: `Service.unload`). The next request then starts a new engine, and that engine reads tens of
GiB of experts off the disk again - measured 19-20 s for 39.97 GiB at 6.40 GiB/s on an RTX PRO 4500 pair
(bench/results/2026-10-02-community-2x-rtx-pro-4500-engine-0.1.36/kaltstart) and ~35 s at 1.42-1.52 GiB/s on a
Ryzen 7 9800X3D (docs/DETAILS.md, "Running it at startup"). With this option on, one long-lived engine process holds
the experts in RAM the whole time the server runs, and every serving engine that starts attaches to pages that are
already there: the engine's own state word says the arena is filled, so it skips the read (src/core/expert_source.cpp,
the protocol the replicas and split serving already use). What goes back when the model is unloaded is the GPU; what
stays is the system RAM.

It is the same memory object the shared replica arena uses (`--shared-expert-arena`, src/core/pinned.cu): a memfd the
server creates and keeps open (so the size of /dev/shm does not cap it), a file in /dev/shm when memfd is not
available, or a named section on Windows. The holder process is an engine started with `--hold-expert-arena`: it
reads the experts into that object, prints `ARENA READY`, and then owns nothing but the mapping, with no GPU context,
no weights and no VRAM - which is the point: it must not hold the card the other program is waiting for.

    "keep_experts_in_ram": true                     (or serve/server.py --keep-experts-in-ram)

Refused, in one line, when the model has no resident arena to keep (--mmap-experts, --resident-experts,
--resident-budget-gib, a config that names an arena of its own, the DeepSeek-V4.1 engine, an engine build without
the mode) or when the RAM cannot hold the arena beside an engine and the headroom (measured per-quant sizes in
strata_control/replica_profile.py). The option is opt-in and nothing changes without it.

Not what it is: it does not keep the K/V, the dense weights, the drafter or the CUDA context - those are re-made on
every start, and they stay the smaller part. And a warm page cache already covers part of this: when the RAM can hold
the experts, the engine reads them buffered on purpose so a restart finds them in the cache (src/core/pinned.cu,
`experts_unbuffered`). The win is largest where the cache cannot hold them and the RAM can.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time

ARENA_KEEP_PREFIX = "strata-keep-"
MIN_FREE_RAM = 8 << 30          # a /dev/shm arena: at least this much room on the mount (the engine reserves the size)
HEADROOM_GIB = 12.0             # the OS, the server, the page cache and the engine's own RAM kept above the arena
ENGINE_READY_S = 1800.0         # the holder's read is the slow one: minutes, not seconds (an NVMe at ~1.4 GiB/s)
RESTART_MIN_S = 20.0            # a holder that dies is started again at most this often (a machine out of RAM would spin)


def requested(cfg: dict | None = None, environ=None) -> bool:
    """Opt-in: "keep_experts_in_ram": true in the run config, or STRATA_KEEP_EXPERTS_IN_RAM=1. Any other value: off."""
    environ = os.environ if environ is None else environ
    if isinstance(cfg, dict) and cfg.get("keep_experts_in_ram") is not None:
        return cfg.get("keep_experts_in_ram") is True
    return str(environ.get("STRATA_KEEP_EXPERTS_IN_RAM", "")).strip().lower() in ("1", "true", "on", "yes")


def _args(cfg: dict) -> list[str]:
    out = [str(a) for a in cfg.get("args") or []]
    for entry in (cfg.get("replicas") if isinstance(cfg.get("replicas"), list) else []):
        if isinstance(entry, dict):
            out += [str(a) for a in entry.get("args") or []]
    return out


def _arg_value(args: list[str], flag: str):
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def unsupported_reason(cfg: dict) -> str | None:
    """None when this config has a resident expert arena that can be kept; the reason in one line when it does not.
    The same rule as the shared replica arena (serve/replicas.py, arena_incompatible_reason): the mapped and budgeted
    RAM modes have no arena, and the DeepSeek engine has no Qwen arena.  The engine family is the config's "backend"
    (setup writes cuda | hip | sycl | metal) because every family builds a program named "strata", so the file name
    says nothing."""
    from serve import replicas as rp
    why = rp.arena_incompatible_reason(cfg, for_what="keep")
    if why:
        return why
    backend = str(cfg.get("backend") or "").strip().lower()
    if backend == "sycl":
        # the Intel tree has the shared arena but not the no-pin form of it: its arena_pin_cap_gib() reads
        # STRATA_ARENA_PIN_GIB=none as "engine decides" and it has no arena_no_register(), so a holder there could
        # register the arena with the device and take VRAM with it - the one thing a holder must never do
        return "the Intel (SYCL) engine build has no arena-holding mode yet"
    if backend == "metal":
        return "the Mac (Metal) engine has no shared expert arena to keep"
    return None


def arena_bytes_for(cfg: dict) -> int:
    """The experts' size in RAM, or 0 when it cannot be known here: <pack>/experts.bin when the pack has one (that is
    the arena's own size), else the measured figure for the quant (strata_control/replica_profile.ARENA_GB)."""
    args = _args(cfg)
    pack = _arg_value(args, "--pack")
    if pack:
        for name in ("experts.bin", "experts-rank1.bin"):
            f = os.path.join(cfg.get("cwd") or ".", pack, name)
            if os.path.isfile(f):
                return os.path.getsize(f)
        f = os.path.join(pack, "experts.bin")           # a pack named absolutely
        if os.path.isfile(f):
            return os.path.getsize(f)
    try:
        from strata_control import replica_profile as rp
        q = rp.quant_of(cfg)
        if q:
            return int(rp.ARENA_GB[q] * (1 << 30))
    except Exception:       # noqa: BLE001 - the table covers the measured quants only; an unknown one says so below
        pass
    return 0


def free_ram() -> int:
    """What the host can still take, or 0 when it cannot be read. 0 lets the plan go on - the engine checks the real
    size and reserves it before it loads (src/core/pinned.cu), the same backstop the shared replica arena relies on -
    a measured figure too small is what refuses here."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import ctypes
        if os.name == "nt":
            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong)] + [("", ctypes.c_ulonglong)] * 4
            ms = _MS()
            ms.dwLength = ctypes.sizeof(ms)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return int(ms.ullAvailPhys)
    except Exception:       # noqa: BLE001
        pass
    return 0


def plan(cfg: dict, *, shm: str = "/dev/shm", platform: str | None = None,
         free_bytes: int | None = None, use_memfd: bool | None = None) -> tuple[str | None, str]:
    """(the arena object for --shared-expert-arena, or None, and one line to print). None carries its reason in the
    line: the option never forces anything and never starts a holder it cannot pay for."""
    if platform is None:
        platform = "win32" if os.name == "nt" else sys.platform
    why = unsupported_reason(cfg)
    if why:
        return None, f"keep experts in RAM: {why}; unloading works as before"
    arena_bytes = arena_bytes_for(cfg)
    if not arena_bytes:
        return None, ("keep experts in RAM: this pack's experts are not sized here (no experts.bin in the pack and a "
                      "quant whose size is not measured); unloading works as before")
    need = arena_bytes + int(HEADROOM_GIB * (1 << 30))
    if platform == "win32":
        # the arena is a named section the engine creates; its pages come from RAM + the pagefile, so the pagefile is
        # what has to be able to hold it (src/core/pinned.cu checks the same thing before taking it)
        name = f"win:Local{chr(92)}{ARENA_KEEP_PREFIX}{os.getpid()}"
        return name, (f"the experts stay in RAM between unloads ({arena_bytes / 2**30:.1f} GiB, one named section "
                      f"{name}): a reload skips the read")
    if not platform.startswith("linux"):
        # the engine's shared expert arena (src/core/pinned.cu) is a MAP_SHARED file or a Windows section; a Mac has
        # neither, and its engine build has no arena at all
        return None, (f"keep experts in RAM: Linux and Windows only (the engine's shared expert arena has no "
                      f"backing here); unloading works as before")
    avail = free_ram() if free_bytes is None else free_bytes
    if avail and avail < need:
        return None, (f"keep experts in RAM: the experts are {arena_bytes / 2**30:.1f} GiB and this PC has "
                      f"{avail / 2**30:.1f} GiB free; the arena plus {HEADROOM_GIB:g} GiB for the OS and the engine "
                      f"needs {need / 2**30:.1f} GiB. Unloading works as before (the model is read again from the disk)")
    if use_memfd is None:
        use_memfd = hasattr(os, "memfd_create")
    if use_memfd:
        try:
            fd = os.memfd_create("strata-keep-arena", os.MFD_CLOEXEC)
        except OSError:
            fd = -1
        if fd >= 0:
            _MEMFDS[f"/proc/{os.getpid()}/fd/{fd}"] = fd
            path = f"/proc/{os.getpid()}/fd/{fd}"
            return path, (f"the experts stay in RAM between unloads ({arena_bytes / 2**30:.1f} GiB, one memory object "
                          f"of this server, not limited by the size of /dev/shm): a reload skips the read")
    if not (os.path.isdir(shm) and os.access(shm, os.W_OK)):
        return None, f"keep experts in RAM: {shm} is not usable and this kernel has no memfd; unloading works as before"
    sweep_stale(shm)
    uid = os.getuid() if hasattr(os, "getuid") else 0
    path = os.path.join(shm, f"{ARENA_KEEP_PREFIX}{uid}-{os.getpid()}.bin")
    free = shutil.disk_usage(shm).free
    if free < arena_bytes + MIN_FREE_RAM:
        return None, (f"keep experts in RAM: {shm} has {free / 2**30:.1f} GiB free and the experts are "
                      f"{arena_bytes / 2**30:.1f} GiB; unloading works as before (make {shm} bigger, or the model is "
                      f"read again from the disk)")
    return path, (f"the experts stay in RAM between unloads ({arena_bytes / 2**30:.1f} GiB in {path}): a reload skips the read")


_MEMFDS: dict[str, int] = {}    # memfds this server made: the path the engines open -> the fd to keep open


def sweep_stale(shm: str = "/dev/shm", uid: int | None = None) -> list[str]:
    """Remove the arena files of servers that no longer run (a killed server cannot unlink its own). A file of a
    living server, or of another user, is never touched."""
    if uid is None:
        uid = os.getuid() if hasattr(os, "getuid") else 0
    gone, head = [], f"{ARENA_KEEP_PREFIX}{uid}-"
    try:
        names = os.listdir(shm)
    except OSError:
        return gone
    for n in names:
        if not (n.startswith(head) and n.endswith(".bin")):
            continue
        try:
            pid = int(n[len(head):-4])
        except ValueError:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        except PermissionError:
            continue
        else:
            continue
        try:
            os.unlink(os.path.join(shm, n))
            gone.append(n)
        except OSError:
            pass
    return gone


def remove_arena(path: str | None) -> None:
    """Close the memfd / unlink the file (at the server's end): a holder or engine that still maps it keeps its pages."""
    if not path or path.startswith("win:"):
        return                                  # (a Windows section goes with its last handle)
    fd = _MEMFDS.pop(path, None)
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
        return
    try:
        os.unlink(path)
    except OSError:
        pass


class ArenaHolder:
    """One engine process whose only job is to own the arena: started at the server's start, watched, and started
    again if it dies (its RAM going back is the failure the option exists to avoid). It reads the experts once, prints
    ARENA READY on stdout, and then blocks; the server ends it by closing its stdin (see src/program/generate.cpp,
    whose wait loop reads the descriptor rather than std::cin for exactly that reason). Its stderr is the engine's own
    lines: they go to the log, and the last few are what a failed start reports - the pack-hash refusal, a layout that
    does not load, an arena the OS would not give."""

    def __init__(self, exe: str, args: list[str], arena: str, *, cwd: str | None = None,
                 env: dict | None = None, log=None, quiet: bool = False):
        self.exe, self.arena = exe, arena
        self.args, self.cwd, self.env, self.log = list(args), cwd, env, log
        self.quiet = quiet
        self.proc, self.pump, self.err_pump = None, None, None
        self.ready, self.dead, self.stopped = threading.Event(), threading.Event(), False
        self._tail: list[str] = []
        self._log_file, self._own_log = None, False
        self._last_start, self._last_report = 0.0, 0.0
        self._lock = threading.Lock()

    def alive(self) -> bool:
        proc = self.proc
        return proc is not None and proc.poll() is None

    def ram_dropped(self) -> bool:
        """True when the holder has ended and was not stopped on purpose: the RAM it held is back, and the next load
        reads the experts off the disk again until the watch loop has started it again."""
        return self.dead.is_set() and not self.stopped and not self.alive()

    def start(self) -> None:
        """Start the holder and wait for ARENA READY. Raises RuntimeError naming the holder's own reason when the
        process ends before it prints it (it prints nothing else on stdout, so an exit is the failure)."""
        with self._lock:
            if self.alive():
                return
            self.stopped = False
            self.ready.clear()
            self.dead.clear()
            self._tail = []
            self._last_start = time.monotonic()
            self._own_log = isinstance(self.log, (str, os.PathLike))
            self._log_file = open(self.log, "a", encoding="utf-8") if self._own_log else self.log
            cmd = [self.exe, "--hold-expert-arena", "--shared-expert-arena", self.arena, *self.args]
            self.proc = subprocess.Popen(cmd, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                         bufsize=1, env=self.env)
            self.pump = threading.Thread(target=self._read_out, args=(self.proc,), daemon=True)
            self.err_pump = threading.Thread(target=self._read_err, args=(self.proc,), daemon=True)
            self.pump.start()
            self.err_pump.start()
            deadline = time.monotonic() + ENGINE_READY_S
            while not self.ready.wait(0.5):
                if time.monotonic() >= deadline:
                    break
                proc = self.proc
                if proc is not None and proc.poll() is not None:
                    for t in (self.pump, self.err_pump):        # what it printed is still in the pipes
                        if t is not None:
                            t.join(10)
                    break
            if self.ready.is_set():
                return
            code = self.proc.poll() if self.proc is not None else None
            why = " | ".join(ln.strip() for ln in self._tail[-3:] if ln.strip())
            self._stop_locked(kill=True)
            if code is None:
                raise RuntimeError(f"the process holding the experts in RAM did not report ARENA READY in "
                                   f"{ENGINE_READY_S:g} s" + (f": {why}" if why else ""))
            raise RuntimeError(f"the process holding the experts in RAM ended (exit {code}) before the arena was "
                               f"ready" + (f": {why}" if why else ""))

    def _read_out(self, proc) -> None:
        try:
            for line in proc.stdout:
                if line.startswith("ARENA READY"):
                    self.ready.set()
                    if not self.quiet:
                        size = line.split()[1] if len(line.split()) > 1 else "?"
                        print(f"[strata] the experts are held in system RAM ({size} bytes); an unloaded model loads "
                              "without reading them again", flush=True)
        except (OSError, ValueError):
            pass
        finally:
            self.dead.set()

    def _read_err(self, proc) -> None:
        try:
            for line in proc.stderr:
                self._tail.append(line)
                del self._tail[:-40]
                if self._log_file is not None:
                    try:
                        self._log_file.write(line)
                        self._log_file.flush()
                    except OSError:
                        pass
        except (OSError, ValueError):
            pass

    def watch(self) -> None:
        """Start the holder again when it died, at most once per RESTART_MIN_S (a machine that cannot pay for the
        arena would otherwise spin, and every try reads tens of GiB). Gives up after three failed starts in a row: a
        start that cannot work is said once, not forever. A dead tick inside the backoff is not a try."""
        tries = 0
        while not self.stopped:
            time.sleep(5.0)
            if self.stopped or self.alive():
                continue
            left = RESTART_MIN_S - (time.monotonic() - self._last_start)
            if left > 0:
                continue
            if tries >= 3:
                return
            tries += 1
            try:
                if not self.quiet:
                    print("[strata] the process holding the experts in RAM ended; starting it again", flush=True)
                self.start()
                tries = 0
            except Exception as e:                                     # noqa: BLE001 - the holder is never a hard failure
                now = time.monotonic()
                if now - self._last_report >= RESTART_MIN_S:
                    print(f"[strata] keep experts in RAM: not restarted ({str(e).splitlines()[0][:200]}); the model "
                          "loads from the disk as before", flush=True)
                    self._last_report = now

    def _stop_locked(self, kill: bool = False) -> None:
        proc = self.proc
        self.proc = None
        if self._own_log and self._log_file is not None:       # (a log the caller owns is left open for it)
            try:
                self._log_file.close()
            except OSError:
                pass
        self._own_log, self._log_file = False, None
        if proc is None or proc.poll() is not None:
            return
        try:                                                          # closing stdin is the polite end
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            pass
        if not kill:
            try:
                proc.terminate()
                proc.wait(timeout=10)
                return
            except (OSError, subprocess.TimeoutExpired):
                pass
        try:
            proc.kill()
            proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def stop(self) -> None:
        """End the holder on purpose: its RAM goes back to the OS with it (the last mapping's end is what frees it)."""
        with self._lock:
            self.stopped = True
            self._stop_locked(kill=True)

