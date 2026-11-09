"""Warm park: unload gives the VRAM back by suspending the engine (cuda-checkpoint) instead of ending it.  No GPU is used:
a fake engine answers PARK / UNPARK and a fake cuda-checkpoint logs what it was asked."""
import json
import os
import stat
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve import park as park_mod
from serve import server
from serve.frontend import ChatTemplate
from serve.park import ParkFailed
from serve.server import ByteTokenizer, Service, StrataEngine, serve
from serve.test_lifecycle import ResidentEngine

FAKE = r'''import queue, sys, threading
cmds = queue.Queue()
def reader():
    for line in sys.stdin:
        cmds.put(line.strip())
    cmds.put('QUIT')
threading.Thread(target=reader, daemon=True).start()
print('READY 4096 stop' + ('' if '--no-park' in sys.argv else ' park'), flush=True)
parks, frozen = 0, False
while True:
    line = cmds.get()
    if frozen and line != 'UNPARK':
        continue                      # a suspended process runs nothing (and cannot end itself)
    if line == 'QUIT':
        break
    if line == 'PARK':
        parks += 1
        frozen = True
        print('VRAM reserve_mib=1000000000 expert_slots=128 vram_free_mib=900', flush=True)
        print(f'PARKED vram_free_mib=100 vram_total_mib=1000 expert_slots=128 n={parks}', flush=True)
    elif line == 'UNPARK':
        frozen = False
        print(f'UNPARKED vram_free_mib=50 vram_total_mib=1000 expert_slots=4096 n={parks}', flush=True)
'''

CHECKPOINT = '''#!/bin/sh
echo "$@" >> "$0.log"
case "$*" in *"--action $FAIL_ON "*) echo "boom" >&2; exit 1;; esac
exit 0
'''


@unittest.skipUnless(sys.platform.startswith("linux"), "cuda-checkpoint is Linux only")
class Protocol(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.script = Path(self.tmp.name) / "engine.py"
        self.script.write_text(FAKE, encoding="utf-8")
        self.checkpoint = Path(self.tmp.name) / "cuda-checkpoint"
        self.checkpoint.write_text(CHECKPOINT, encoding="utf-8")
        self.checkpoint.chmod(self.checkpoint.stat().st_mode | stat.S_IXUSR)
        self.log = Path(str(self.checkpoint) + ".log")
        self.fail_on = mock.patch.dict(os.environ, {"FAIL_ON": "none"})
        self.fail_on.start()
        self.addCleanup(self.fail_on.stop)
        self.ram = mock.patch.object(park_mod, "ram_available_mib", return_value=64000)
        self.ram.start()
        self.addCleanup(self.ram.stop)
        self.smi = mock.patch.object(park_mod, "process_vram_mib", return_value=None)
        self.smi.start()
        self.addCleanup(self.smi.stop)
        self.quit = mock.patch.object(server, "ENGINE_QUIT_WAIT_S", 1)     # the fake ignores QUIT once PARK froze it
        self.quit.start()
        self.addCleanup(self.quit.stop)

    def engine(self, *args):
        real_popen = server.subprocess.Popen
        with mock.patch.object(server.subprocess, "Popen", lambda cmd, **kw: real_popen([sys.executable, str(self.script), *cmd[1:]], **kw)):
            eng = server.StrataEngine("strata", list(args))
        eng.checkpoint_bin = str(self.checkpoint)
        self.addCleanup(eng.close)
        return eng

    def actions(self):
        return [line.split()[1] for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_park_and_unpark_suspend_the_process_not_end_it(self):
        eng = self.engine()
        self.assertTrue(eng.can_park)
        pid = eng.proc.pid
        info = eng.park()
        self.assertEqual(info["n"], 1)
        self.assertTrue(eng.parked)
        self.assertFalse(eng.alive())                 # not serving: the server loads it again for the next request
        self.assertTrue(eng.unloaded)
        self.assertIsNone(eng.proc.poll())            # ... but the process is still there, with its RAM
        self.assertEqual(self.actions(), ["lock", "checkpoint"])
        self.assertIn(f"--pid {pid}", self.log.read_text())
        self.assertIn("--timeout 30000", self.log.read_text().splitlines()[0])
        info = eng.unpark()
        self.assertEqual(info["expert_slots"], 4096)
        self.assertTrue(eng.alive())
        self.assertFalse(eng.parked or eng.unloaded)
        self.assertEqual(eng.proc.pid, pid)
        self.assertEqual(self.actions(), ["lock", "checkpoint", "restore", "unlock"])
        self.assertEqual(eng.park()["n"], 2)          # and again

    def test_a_parked_engine_is_killed_by_the_plain_unload(self):
        eng = self.engine()
        eng.park()
        proc = eng.proc
        started = time.monotonic()
        eng.unload()                                   # it ignores QUIT while suspended: close() must not wait for it
        self.assertLess(time.monotonic() - started, 10)
        self.assertIsNotNone(proc.poll())
        self.assertIsNone(eng.proc)
        self.assertFalse(eng.parked)

    def test_too_little_ram_for_the_vram_image_ends_the_engine(self):
        eng = self.engine()
        with mock.patch.object(park_mod, "ram_available_mib", return_value=1000):    # 900 MiB held + 2 GiB spare
            with self.assertRaisesRegex(ParkFailed, "would not fit"):
                eng.park()
        self.assertIsNone(eng.proc)
        self.assertTrue(eng.unloaded)
        self.assertEqual(self.actions(), [])           # nothing was suspended

    def test_the_engine_vram_alone_decides_when_it_can_be_told(self):
        eng = self.engine()
        with mock.patch.object(park_mod, "process_vram_mib", return_value=100), \
                mock.patch.object(park_mod, "ram_available_mib", return_value=2300):
            eng.park()                                 # 100 MiB held + 2048 spare fit in 2300

    def test_a_failed_checkpoint_after_the_lock_kills_the_engine(self):
        eng = self.engine()
        os.environ["FAIL_ON"] = "checkpoint"
        started = time.monotonic()
        with self.assertRaisesRegex(ParkFailed, "boom"):
            eng.park()
        self.assertLess(time.monotonic() - started, 10)
        self.assertIsNone(eng.proc)
        self.assertEqual(self.actions(), ["lock", "checkpoint"])

    def test_a_failed_restore_ends_the_engine_for_a_restart(self):
        eng = self.engine()
        eng.park()
        os.environ["FAIL_ON"] = "restore"
        with self.assertRaisesRegex(ParkFailed, "boom"):
            eng.unpark()
        self.assertIsNone(eng.proc)
        self.assertFalse(eng.parked)
        self.assertTrue(eng.unloaded)

    def test_an_engine_without_park_or_without_cuda_checkpoint_is_left_alone(self):
        eng = self.engine("--no-park")
        self.assertFalse(eng.can_park)
        with self.assertRaisesRegex(ParkFailed, "does not offer PARK"):
            eng.park()
        self.assertTrue(eng.alive())
        eng = self.engine()
        eng.checkpoint_bin = None
        with self.assertRaisesRegex(ParkFailed, "cuda-checkpoint was not found"):
            eng.park()
        self.assertTrue(eng.alive())


class Parkable(ResidentEngine):
    can_park = True
    checkpoint_bin = "cuda-checkpoint"

    def __init__(self, tok):
        super().__init__(tok)
        self.parked, self.parks, self.unparks, self.park_error, self.unpark_error = False, 0, 0, None, None

    def alive(self):
        return self.loaded and not self.parked

    def park(self):
        if self.park_error:
            self.close()
            self.unloaded = True
            raise ParkFailed(self.park_error)
        self.parks += 1
        self.parked, self.unloaded = True, True
        return {"ms": 1500}

    def unpark(self):
        if self.unpark_error:
            self.unload()
            raise ParkFailed(self.unpark_error)
        self.unparks += 1
        self.parked, self.unloaded = False, False
        return {"ms": 800}

    def close(self):
        self.parked = False
        super().close()


class WarmService(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = Parkable(tok)
        self.engine.loaded, self.engine.unloaded = True, False
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.warm = True
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.chat = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32, "reasoning_effort": "none"}

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode())

    def test_unload_parks_and_the_next_request_unparks(self):
        code, body = self.request("/unload", {})
        self.assertEqual((code, body), (200, {"status": "unloaded"}))      # the answer clients already know
        self.assertEqual((self.engine.parks, self.engine.closes), (1, 0))
        status = self.request("/v1/status")[1]
        self.assertEqual((status["loaded"], status["parked"], status["warm"]), (False, True, True))
        self.assertEqual(self.request("/v1/models")[1]["data"][0]["status"]["value"], "unloaded")
        self.assertEqual(self.request("/unload", {})[1]["status"], "not loaded")      # parked already
        code, answer = self.request("/v1/chat/completions", self.chat)
        self.assertEqual(code, 200)
        self.assertEqual((self.engine.unparks, self.engine.starts), (1, 0))             # no restart: it came back
        self.assertTrue(self.request("/v1/status")[1]["loaded"])

    def test_idle_unload_parks_too(self):
        self.svc.last_request_at = time.time() - 100
        self.assertEqual(self.svc.unload(idle_for=10), "unloaded")
        self.assertEqual((self.engine.parks, self.engine.closes), (1, 0))

    def test_full_ends_a_loaded_or_a_parked_engine(self):
        self.assertEqual(self.request("/unload?full=1", {})[1]["status"], "unloaded")
        self.assertEqual((self.engine.parks, self.engine.closes), (0, 1))
        self.engine.loaded, self.engine.unloaded = True, False
        self.svc.unload()
        self.assertTrue(self.engine.parked)
        self.assertEqual(self.request("/v1/unload", {"full": True})[1]["status"], "unloaded")
        self.assertFalse(self.engine.parked)
        self.assertEqual(self.engine.closes, 2)
        self.request("/v1/chat/completions", self.chat)
        self.assertEqual((self.engine.starts, self.engine.unparks), (1, 0))

    def test_a_failed_park_is_the_plain_unload(self):
        self.engine.park_error = "no room"
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertFalse(self.engine.loaded)
        self.request("/v1/chat/completions", self.chat)
        self.assertEqual(self.engine.starts, 1)

    def test_a_failed_unpark_starts_the_engine_again(self):
        self.svc.unload()
        self.engine.unpark_error = "the card is full"
        code, _ = self.request("/v1/chat/completions", self.chat)
        self.assertEqual(code, 200)
        self.assertEqual(self.engine.starts, 1)

    def test_without_warm_unload_is_unchanged(self):
        self.svc.warm = False
        self.svc.unload()
        self.assertEqual((self.engine.parks, self.engine.closes), (0, 1))

    def test_an_engine_that_cannot_park_says_so_once_and_unloads(self):
        self.engine.can_park = False
        with mock.patch("builtins.print") as said:
            self.svc.unload()
        self.assertEqual(self.engine.closes, 1)
        self.assertTrue(any("cannot be parked" in str(c.args[0]) for c in said.call_args_list))
        self.assertTrue(self.svc.warm_said)

    def test_before_load_and_the_free_vram_check_still_gate_the_unpark(self):
        self.svc.unload()
        self.svc.min_free_vram_mib = 8000
        self.svc.free_vram_mib = lambda: 100
        self.svc.vram_free_wait_s = 0
        with self.assertRaises(server.GpuBusy):
            self.svc.load()
        self.assertEqual(self.engine.unparks, 0)
        self.assertTrue(self.engine.parked)


class Config(unittest.TestCase):
    def test_find_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            tool = Path(d) / "cuda-checkpoint"
            tool.write_text("#!/bin/sh\n")
            tool.chmod(0o755)
            if sys.platform.startswith("linux"):
                self.assertEqual(park_mod.find_checkpoint(str(tool)), str(tool))
                self.assertIsNone(park_mod.find_checkpoint(str(Path(d) / "missing")))
                with mock.patch.dict(os.environ, {"STRATA_CUDA_CHECKPOINT": str(tool)}):
                    self.assertEqual(park_mod.find_checkpoint(), str(tool))
            else:
                self.assertIsNone(park_mod.find_checkpoint(str(tool)))

    def test_environment_settings(self):
        env = {"STRATA_WARM": "On", "STRATA_LAZY_LOAD": "0", "STRATA_IDLE_UNLOAD_S": "90.5", "BAD_B": "maybe", "BAD_S": "-3"}
        with mock.patch.dict(os.environ, env), mock.patch("builtins.print") as said:
            self.assertTrue(server.env_bool("STRATA_WARM", False))
            self.assertFalse(server.env_bool("STRATA_LAZY_LOAD", True))
            self.assertTrue(server.env_bool("UNSET_B", True))                 # unset: the config's value
            self.assertTrue(server.env_bool("BAD_B", True))                   # not 1 or 0: ignored, said
            self.assertEqual(server.env_seconds("STRATA_IDLE_UNLOAD_S", 5), 90.5)
            self.assertEqual(server.env_seconds("UNSET_S", 5), 5)
            self.assertEqual(server.env_seconds("BAD_S", 5), 5)
        self.assertEqual(len(said.call_args_list), 2)

    def test_warm_is_an_editable_run_config_key(self):
        from serve import runconfig
        new, changed = runconfig.apply({"args": []}, {"warm": True})
        self.assertEqual((new["warm"], changed), (True, ["warm"]))
        with self.assertRaises(ValueError):
            runconfig.apply({"args": []}, {"warm": "yes"})


if __name__ == "__main__":
    unittest.main()
