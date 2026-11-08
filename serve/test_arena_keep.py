"""Keeping the experts in system RAM over an unload (serve/arena_keep.py).

Two things are tested: the plan (who may keep an arena, and the RAM budget guard that refuses before anything is
spent), and the holder's lifecycle against a fake engine that speaks the protocol src/program/generate.cpp's
--hold-expert-arena speaks - ARENA READY on stdout, then the end on stdin's EOF or QUIT. A third part tests the
arena's NUMA placement (numa_env) against a fake /sys tree: no real sysfs, no syscall."""
import os
import shutil
import sys
import tempfile
import time
import unittest

from serve import arena_keep as ak
from serve.arena_keep import ArenaHolder, plan, remove_arena, requested, unsupported_reason

# Qwen3.8-Flash-Next at IQ3_S: 50.3 GiB of experts (strata_control/replica_profile.ARENA_GB), what the guard counts
CFG = {"exe": "engine/strata-cuda", "args": ["--pack", "pack-iq3_s"], "model_name": "Qwen3.8-Flash-Next IQ3_S"}


class OptIn(unittest.TestCase):
    def test_only_true_turns_it_on(self):
        self.assertTrue(requested({"keep_experts_in_ram": True}))
        self.assertFalse(requested({"keep_experts_in_ram": False}))
        self.assertFalse(requested({"keep_experts_in_ram": "true"}))       # the config field is a real boolean
        self.assertFalse(requested({}))
        self.assertTrue(requested({}, {"STRATA_KEEP_EXPERTS_IN_RAM": "1"}))
        self.assertFalse(requested({}, {"STRATA_KEEP_EXPERTS_IN_RAM": "0"}))
        self.assertTrue(requested({"keep_experts_in_ram": True}, {}))


class Refusals(unittest.TestCase):
    """A config with no resident expert arena keeps nothing, and says which rule it hit."""

    def test_the_modes_without_an_arena(self):
        for flag in ("--mmap-experts", "--resident-experts", "--resident-budget-gib"):
            why = unsupported_reason({**CFG, "args": CFG["args"] + [flag]})
            self.assertIsNotNone(why, flag)
            self.assertIn("no resident expert arena", why, flag)

    def test_an_arena_this_server_does_not_own(self):
        why = unsupported_reason({**CFG, "args": CFG["args"] + ["--shared-expert-arena", "/x"]})
        self.assertIn("already names a shared expert arena", why)

    def test_deepseek_has_no_qwen_arena(self):
        self.assertIn("DeepSeek", unsupported_reason({**CFG, "arch": "deepseek41"}))

    def test_engine_families_without_the_mode(self):
        # every family builds a program named "strata", so the family comes from the config's "backend"
        self.assertIn("SYCL", unsupported_reason({**CFG, "backend": "sycl"}))
        self.assertIn("Metal", unsupported_reason({**CFG, "backend": "metal"}))
        self.assertIsNone(unsupported_reason({**CFG, "backend": "cuda"}))
        self.assertIsNone(unsupported_reason({**CFG, "backend": "hip"}))       # the same generate.cpp as CUDA
        self.assertIsNone(unsupported_reason(CFG))                             # no backend key: the CUDA default

    def test_a_refusal_plans_nothing_and_says_unloading_is_as_before(self):
        path, note = plan({**CFG, "arch": "deepseek41"})
        self.assertIsNone(path)
        self.assertIn("unloading works as before", note)


class Budget(unittest.TestCase):
    """The guard runs at the server's start, where a refusal is free. It never refuses at an unload."""

    def test_ram_that_cannot_hold_the_arena_refuses_with_both_numbers(self):
        path, note = plan(CFG, free_bytes=30 << 30)
        self.assertIsNone(path)
        self.assertIn("50.3 GiB", note)                 # the experts
        self.assertIn("30.0 GiB free", note)            # what the PC has
        self.assertIn("62.3 GiB", note)                 # the arena plus the headroom
        self.assertIn("nloading works as before", note)

    def test_enough_ram_takes_the_memfd(self):
        path, note = plan(CFG, free_bytes=96 << 30)
        self.assertTrue(path and path.startswith("/proc/") and "/fd/" in path, path)
        self.assertIn("not limited by the size of /dev/shm", note)
        self.assertIn("/proc/self/fd".replace("self", str(os.getpid())), path)
        remove_arena(path)                              # closes our fd; harmless twice (atexit + the closer)
        self.assertNotIn(path, ak._MEMFDS)

    def test_an_unmeasurable_free_ram_defers_to_the_engine(self):
        # free_ram() returns 0 when /proc/meminfo cannot be read; 0 is "not measured", not "no room" - the engine
        # reserves the real size before it loads (src/core/pinned.cu), as it does for the shared replica arena
        path, note = plan(CFG, free_bytes=0)
        self.assertIsNotNone(path)

    def test_dev_shm_without_memfd_needs_room_on_the_mount(self):
        with tempfile.TemporaryDirectory() as d:
            if shutil.disk_usage(d).free < 59 << 30:      # the guard needs a mount able to hold 50.3 GiB
                self.skipTest(f"{d} has too little free room to test the accepting path")
            path, note = plan(CFG, shm=d, free_bytes=96 << 30, use_memfd=False)
            self.assertTrue(path and path.startswith(d) and "strata-keep-" in os.path.basename(path), path)
            self.assertIn("a reload skips the read", note)
            remove_arena(path)

    def test_a_pack_larger_than_the_mount_is_refused_with_both_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            pack = os.path.join(d, "pack")
            os.makedirs(pack)
            with open(os.path.join(pack, "experts.bin"), "wb") as f:
                f.truncate(60 << 30)                    # sparse: only the size matters
            cfg = {"exe": "e", "args": ["--pack", pack]}
            free = __import__("shutil").disk_usage(d).free
            if free >= 68 << 30:                        # a tmpfs that can hold it cannot show the refusal
                self.skipTest(f"{d} has {free / 2**30:.0f} GiB free")
            path, note = plan(cfg, shm=d, free_bytes=96 << 30, use_memfd=False)
            self.assertIsNone(path)
            self.assertIn("60.0 GiB", note)
            self.assertIn("make", note)

    def test_windows_gets_one_named_section(self):
        path, note = plan(CFG, platform="win32")
        self.assertTrue(path and path.startswith("win:Local") and "strata-keep-" in path, path)
        self.assertIn("named section", note)
        remove_arena(path)                              # (nothing to remove: harmless, as for the replicas')

    def test_darwin_refuses_the_same_way_the_replicas_do(self):
        path, note = plan(CFG, platform="darwin", free_bytes=96 << 30)
        self.assertIsNone(path)
        self.assertIn("Linux and Windows only", note)
        self.assertIn("unloading works as before", note)

    def test_a_pack_this_helper_cannot_size_says_so(self):
        path, note = plan({"exe": "e", "args": ["--pack", "/no/such/pack"]}, free_bytes=96 << 30)
        self.assertIsNone(path)
        self.assertIn("not sized here", note)


# the fake holder: exactly the protocol --hold-expert-arena speaks (PY marks the shebang; fake() fills it in)
READY = """#!PY
import sys
sys.stderr.write("strata generate: expert arena: mapped shared file\\n")
sys.stderr.write("strata generate: holding the expert arena in RAM: 50.30 GiB read in 19.4 s (2.59 GiB/s)\\n")
sys.stderr.flush()
print("ARENA READY " + str(50 * 2**30), flush=True)
sys.stdin.read()          # QUIT or stdin's EOF (the server closed it): then the RAM goes back
sys.stderr.write("strata generate: holding the expert arena: ended, the RAM goes back\\n")
"""

FAILS = """#!PY
import sys
sys.stderr.write("strata generate: hold expert arena: shared arena was written for pack hash 0000abcd, "
                 "expected 1111beef\\n")
sys.stderr.flush()
sys.exit(1)
"""

DIES = """#!PY
import os, sys
print("ARENA READY " + str(50 * 2**30), flush=True)
if not os.path.exists("died-marker"):        # the holder runs in the test's cwd
    open("died-marker", "w").close()
    sys.exit(3)                              # the OOM killer, say: the RAM it held goes back with it
sys.stdin.read()                             # the restarted holder: ready and staying
"""


def fake(source: str, d: str, name: str = "fake-holder") -> str:
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(source.replace("#!PY", "#!" + sys.executable, 1))
    os.chmod(p, 0o755)
    return p


class Holder(unittest.TestCase):
    """The lifecycle the server depends on: it waits for ARENA READY before an engine may attach, says the holder's
    own reason when it never gets there, and ends it by closing stdin."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.arena = os.path.join(self.tmp.name, "arena.bin")

    def holder(self, source: str, **kw) -> ArenaHolder:
        return ArenaHolder(fake(source, self.tmp.name), ["--pack", "p"], self.arena, cwd=self.tmp.name, quiet=True, **kw)

    def test_start_returns_only_once_the_arena_is_ready(self):
        h = self.holder(READY)
        self.addCleanup(h.stop)
        t0 = time.monotonic()
        h.start()                                          # blocks until the ARENA READY line
        self.assertLess(time.monotonic() - t0, 30.0)
        self.assertTrue(h.alive())
        self.assertFalse(h.ram_dropped())

    def test_a_holder_that_refuses_reports_its_own_reason(self):
        h = self.holder(FAILS)
        with self.assertRaises(RuntimeError) as got:
            h.start()
        msg = str(got.exception)
        self.assertIn("pack hash", msg)                    # the engine's own line, from its stderr
        self.assertIn("exit 1", msg)
        self.assertFalse(h.alive())

    def test_stop_ends_it_through_stdin(self):
        h = self.holder(READY)
        h.start()
        proc = h.proc
        h.stop()
        proc.wait(10)                                      # ended, and ended on purpose
        self.assertFalse(h.alive())
        self.assertFalse(h.ram_dropped())                  # stopped: not a lost arena

    def test_a_holder_that_died_says_the_ram_is_back(self):
        h = self.holder(DIES)
        h.start()                                          # ready, then it dies on its own
        deadline = time.monotonic() + 15
        while h.alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertFalse(h.alive())
        self.assertTrue(h.ram_dropped())

    def test_watch_starts_a_dead_holder_again(self):
        import threading
        h = self.holder(DIES)
        h.start()                                  # ready, then the fake ends itself (the marker's first time)
        first_pid = h.proc.pid
        t = threading.Thread(target=h.watch, daemon=True)
        t.start()
        self.addCleanup(h.stop)
        restarted, deadline = False, time.monotonic() + 90     # RESTART_MIN_S is the watch's own backoff
        while time.monotonic() < deadline:
            if h.alive() and h.proc is not None and h.proc.pid != first_pid:
                restarted = True
                break
            time.sleep(0.5)
        self.assertTrue(restarted, "the watch loop did not start the holder again")
        self.assertFalse(h.ram_dropped())          # the RAM is held again
        h.stop()

    def test_the_log_gets_the_holder_s_own_lines(self):
        log = os.path.join(self.tmp.name, "strata.log")
        h = self.holder(FAILS, log=log)
        with self.assertRaises(RuntimeError):
            h.start()
        with open(log, encoding="utf-8") as f:
            self.assertIn("was written for pack hash", f.read())

    def test_remove_arena_is_harmless_twice_and_on_windows_names(self):
        remove_arena(None)
        remove_arena("win:Local\\strata-keep-1")
        remove_arena(os.path.join(self.tmp.name, "gone.bin"))


class NumaPlacement(unittest.TestCase):
    """numa_env: the holder-only STRATA_ARENA_NUMA_NODE the keeper adds (serve/arena_keep.py), tested against a
    fake /sys tree - no real sysfs, no syscall. Unset must stay {} and "": without the key the holder is started
    with exactly the environment it was before."""

    GIB = 2**30

    def sysfs(self, d: str, cards: list[tuple[str, str, str]], nodes: dict[int, int | None]) -> str:
        """A fake /sys: `cards` = (pci address, class, numa_node or "") in creation order (the helper must sort),
        `nodes` = {node: MemFree KiB or None for no folder}."""
        for addr, cls, node in cards:
            p = os.path.join(d, "bus", "pci", "devices", f"0000:{addr}")
            os.makedirs(p)
            with open(os.path.join(p, "class"), "w", encoding="ascii") as f:
                f.write(cls + "\n")
            if node != "":
                with open(os.path.join(p, "numa_node"), "w", encoding="ascii") as f:
                    f.write(node + "\n")
        for n, free_kib in nodes.items():
            if free_kib is None:
                continue
            p = os.path.join(d, "devices", "system", "node", f"node{n}")
            os.makedirs(p)
            with open(os.path.join(p, "meminfo"), "w", encoding="ascii") as f:
                f.write(f"Node {n} MemTotal:       131954300 kB\n"
                        f"Node {n} MemFree:        {free_kib} kB\n"
                        f"Node {n} MemUsed:        100 kB\n")
        return d

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_auto_names_the_node_of_the_named_card(self):
        s = self.sysfs(self.d, [("05:00.0", "0x030000", "1"), ("00:01.0", "0x030200", "0")],
                       {0: 100 * 1024 * 1024, 1: 100 * 1024 * 1024})
        # the fake lists the node-1 card first: the helper must still number them by PCI address, node 0 first
        env, note = ak.numa_env({"arena_numa_node": "auto", "gpu": 1}, 48 * self.GIB, environ={}, sys_root=s,
                                platform="linux")
        self.assertEqual(env, {"STRATA_ARENA_NUMA_NODE": "1"})
        self.assertIn("NUMA node 1", note)
        env, note = ak.numa_env({"arena_numa_node": "auto", "gpu": [0, 1]}, 48 * self.GIB, environ={}, sys_root=s,
                                platform="linux")
        self.assertEqual(env, {})                             # a split across two nodes: no single node for "auto"
        self.assertIn("one arena node cannot serve both", note)

    def test_only_display_cards_count(self):
        s = self.sysfs(self.d, [("05:00.0", "0x030000", "1"), ("00:17.0", "0x010601", "0")],
                       {0: 100 * 1024 * 1024, 1: 100 * 1024 * 1024})
        env, _ = ak.numa_env({"arena_numa_node": "auto", "gpu": 0}, 48 * self.GIB, environ={}, sys_root=s,
                             platform="linux")
        self.assertEqual(env, {"STRATA_ARENA_NUMA_NODE": "1"})    # the SATA controller is not a card

    def test_the_config_number_beats_the_environment(self):
        s = self.sysfs(self.d, [("05:00.0", "0x030000", "1")], {0: 100 * 1024 * 1024, 1: 100 * 1024 * 1024})
        cfg = {"arena_numa_node": 0, "gpu": 0}
        env, _ = ak.numa_env(cfg, 48 * self.GIB, environ={"STRATA_ARENA_NUMA_NODE": "1"}, sys_root=s,
                             platform="linux")
        self.assertEqual(env, {"STRATA_ARENA_NUMA_NODE": "0"})
        env, _ = ak.numa_env({"gpu": 0}, 48 * self.GIB, environ={"STRATA_ARENA_NUMA_NODE": "1"}, sys_root=s,
                             platform="linux")
        self.assertEqual(env, {"STRATA_ARENA_NUMA_NODE": "1"})    # no config key: the environment's turn

    def test_off_and_unset_place_nothing(self):
        s = self.sysfs(self.d, [("05:00.0", "0x030000", "1")], {0: 100 * 1024 * 1024})
        for cfg, environ in (({"arena_numa_node": "off"}, {}), ({"arena_numa_node": None}, {"STRATA_ARENA_NUMA_NODE": "off"}),
                             ({}, {}), ({}, {"STRATA_ARENA_NUMA_NODE": ""})):
            self.assertEqual(ak.numa_env(cfg, 48 * self.GIB, environ=environ, sys_root=s, platform="linux"),
                             ({}, ""))

    def test_off_in_the_config_overrides_an_inherited_environment_number(self):
        # child_env copies the server's own environment to the holder: a config that says "off" has to silence a
        # raw STRATA_ARENA_NUMA_NODE the server happens to carry, not just ignore it
        s = self.sysfs(self.d, [("05:00.0", "0x030000", "1")], {0: 100 * 1024 * 1024, 1: 100 * 1024 * 1024})
        env, note = ak.numa_env({"arena_numa_node": "off"}, 48 * self.GIB,
                                environ={"STRATA_ARENA_NUMA_NODE": "1"}, sys_root=s, platform="linux")
        self.assertEqual(env, {"STRATA_ARENA_NUMA_NODE": "off"})   # the holder reads "off" as off, never as node 1
        self.assertEqual(note, "")

    def test_a_node_the_arena_does_not_fit_in_is_refused_with_both_numbers(self):
        s = self.sysfs(self.d, [("05:00.0", "0x030000", "1")], {0: 40 * 1024 * 1024, 1: 20 * 1024 * 1024})
        env, note = ak.numa_env({"arena_numa_node": 1, "gpu": 0}, 48 * self.GIB, environ={}, sys_root=s,
                                platform="linux")
        self.assertEqual(env, {})
        self.assertIn("node 1 has 20.0 GiB free", note)
        self.assertIn("the arena is 48.0 GiB", note)
        self.assertIn("as before", note)                          # the sentence ends in what still happens

    def test_a_node_that_cannot_say_how_much_it_has_goes_on_with_a_note(self):
        s = self.sysfs(self.d, [("05:00.0", "0x030000", "1")], {})   # no node folders at all
        env, note = ak.numa_env({"arena_numa_node": 1}, 48 * self.GIB, environ={}, sys_root=s, platform="linux")
        self.assertEqual(env, {"STRATA_ARENA_NUMA_NODE": "1"})       # unknown free: the holder's own line tells later
        self.assertIn("does not say how much the node has free", note)

    def test_auto_needs_a_card_and_a_sysfs_that_can_answer(self):
        s = self.sysfs(self.d, [], {})
        env, note = ak.numa_env({"arena_numa_node": "auto"}, 48 * self.GIB, environ={}, sys_root=s, platform="linux")
        self.assertEqual(env, {})
        self.assertIn("'auto' needs the model to name a card", note)
        env, note = ak.numa_env({"arena_numa_node": "auto", "gpu": 3}, 48 * self.GIB, environ={}, sys_root=s,
                                platform="linux")
        self.assertEqual(env, {})
        self.assertIn("0 display card(s)", note)
        s2 = self.sysfs(tempfile.mkdtemp(dir=self.d), [("05:00.0", "0x030000", "-1")], {})
        env, note = ak.numa_env({"arena_numa_node": "auto", "gpu": 0}, 48 * self.GIB, environ={}, sys_root=s2,
                                platform="linux")
        self.assertEqual(env, {})
        self.assertIn("cannot tell which CPU the card sits on", note)

    def test_a_value_that_is_not_a_node_says_so_and_places_as_before(self):
        s = self.sysfs(self.d, [("05:00.0", "0x030000", "1")], {0: 100 * 1024 * 1024, 1: 100 * 1024 * 1024})
        for v in ("banana", 64, -1):
            env, note = ak.numa_env({"arena_numa_node": v}, 48 * self.GIB, environ={}, sys_root=s, platform="linux")
            self.assertEqual(env, {})
            self.assertIn("as before", note)

    def test_windows_and_mac_get_nothing(self):
        for plat in ("win32", "darwin"):
            self.assertEqual(ak.numa_env({"arena_numa_node": 0}, 48 * self.GIB, environ={}, sys_root=self.d,
                                          platform=plat), ({}, ""))


class UnloadSaysWhatIsKept(unittest.TestCase):
    """The window's line must not promise that RAM goes back: with the holder up, only the GPU does."""

    class FakeHolder:
        def __init__(self, alive):
            self._alive = alive

        def alive(self):
            return self._alive

    def svc(self):
        from pathlib import Path
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, Service
        from serve.test_lifecycle import ResidentEngine
        tok = ByteTokenizer()
        tpl = Path(__file__).parent / "chat_template.jinja"
        return Service(ResidentEngine(tok), tok, ChatTemplate(tpl), model_name="m")

    def unload_line(self, svc) -> str:
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(svc.unload(), "unloaded")
        return buf.getvalue()

    def test_with_a_holder_the_line_names_the_kept_experts(self):
        svc = self.svc()
        svc.engine.restart()                               # ResidentEngine starts unloaded
        svc.arena_holder = self.FakeHolder(True)
        out = self.unload_line(svc)
        self.assertIn("the experts stay in system RAM", out)
        self.assertNotIn("the next request loads it again", out)

    def test_without_a_holder_the_line_is_today_s_word_for_word(self):
        svc = self.svc()
        svc.engine.restart()
        self.assertIn("the next request loads it again", self.unload_line(svc))

    def test_the_status_says_whether_the_experts_are_kept(self):
        # v1_status is the model-state surface (/v1, the TUI): the key sits beside "loaded" and "auto_load"
        svc = self.svc()
        self.assertIsNone(svc.v1_status().get("experts_in_ram", None))   # off: null, the option is not on
        svc.arena_holder = self.FakeHolder(True)
        self.assertTrue(svc.v1_status()["experts_in_ram"])
        svc.arena_holder = self.FakeHolder(False)                        # the holder died: the RAM is back
        self.assertFalse(svc.v1_status()["experts_in_ram"])
