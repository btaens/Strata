"""serve/test_docker_env.py - the container settings a `docker run -e` sets once: STRATA_LAZY_LOAD,
STRATA_IDLE_UNLOAD_S and STRATA_DISABLE_WEBUI (docs/DOCKER.md; docker/entrypoint.sh passes them through).
No GPU, no pack, no engine binary: a lazy StrataEngine never starts one, and the mock engine answers.

    python -m pytest serve/test_docker_env.py -q
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (ByteTokenizer, MockEngine, Service, env_flag, env_seconds,  # noqa: E402
                          serve)

ROOT = Path(__file__).resolve().parents[1]


class EnvHelpers(unittest.TestCase):
    """The two readers every env setting goes through."""

    def test_env_flag_forms(self):
        for on in ("1", "true", "TRUE", "yes", "on", " 1 "):
            with mock.patch.dict(os.environ, {"STRATA_X": on}):
                self.assertTrue(env_flag("STRATA_X"), repr(on))
        for off in ("", "0", "false", "no", "off", "maybe"):
            with mock.patch.dict(os.environ, {"STRATA_X": off}):
                self.assertFalse(env_flag("STRATA_X"), repr(off))
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("STRATA_X", None)
            self.assertFalse(env_flag("STRATA_X"))           # unset: off

    def test_env_seconds_forms(self):
        with mock.patch.dict(os.environ, {"STRATA_Y": "1800"}):
            self.assertEqual(env_seconds("STRATA_Y"), 1800.0)
        with mock.patch.dict(os.environ, {"STRATA_Y": " 90.5 "}):
            self.assertEqual(env_seconds("STRATA_Y"), 90.5)
        for empty in ("", "   "):
            with mock.patch.dict(os.environ, {"STRATA_Y": empty}):
                self.assertIsNone(env_seconds("STRATA_Y"))   # empty is unset, not 0 typed wrong
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("STRATA_Y", None)
            self.assertIsNone(env_seconds("STRATA_Y"))
        with mock.patch.dict(os.environ, {"STRATA_Y": "30m"}):
            with self.assertRaises(SystemExit) as e:
                env_seconds("STRATA_Y")
            self.assertIn("STRATA_Y='30m'", str(e.exception))  # a typo names the variable and what it said


class _Reached(Exception):
    """raised by the patched serve(): main() got all the way through setup."""


class MainTakesTheEnvironment(unittest.TestCase):
    """main(), --engine strata with a lazy start (no engine process is ever spawned), so the three
    settings can be seen on the Service that reaches serve()."""

    def run_main(self, cfg, extra_env=None, argv_extra=()):
        """-> (the fields serve() saw or None, stdout, the exception main() raised or None)."""
        import serve.server as S
        seen = {}

        def fake_serve(svc, host=None, port=None):
            seen["idle"] = svc.idle_unload_s
            seen["webui"] = svc.webui
            seen["engine_lazy"] = getattr(svc.engine, "unloaded", None)   # a lazy engine stays unloaded
            raise _Reached

        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, extra_env or {}), \
                mock.patch.object(S, "serve", fake_serve), \
                mock.patch.object(sys, "stderr", err), redirect_stdout(out):
            os.environ.pop("STRATA_API_KEY", None)
            root = Path(d)
            # a real (byte) tokenizer so main() passes its tokenizer check the way a profile's does
            from strata_tokenizer import BYTE_TO_UNICODE
            tok_dir = root / "tokenizer"
            tok_dir.mkdir()
            tokens = [BYTE_TO_UNICODE[i] for i in range(256)] + ["ab", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
            (tok_dir / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}), encoding="utf-8")
            (tok_dir / "merges.txt").write_text("a b", encoding="utf-8")
            (tok_dir / "token_type.json").write_text(json.dumps([0] * 257 + [3, 3, 3]), encoding="utf-8")
            p = root / "cfg.json"
            p.write_text(json.dumps({"exe": str(root / "no-such-engine"), "tokenizer": str(tok_dir),
                                     "args": [], "cwd": str(root), "model_name": "test"} | cfg), encoding="utf-8")
            raised = None
            with mock.patch.object(sys, "argv", ["server.py", "--engine", "strata", "--port", "0",
                                                 "--config", str(p), *argv_extra]):
                try:
                    S.main()
                except _Reached:
                    pass
                except (SystemExit, Exception) as e:      # ap.error -> SystemExit(2); both are the answer
                    raised = e
        return seen, out.getvalue(), raised

    def test_lazy_from_the_environment(self):
        seen, out, raised = self.run_main({}, {"STRATA_LAZY_LOAD": "1"})
        self.assertIsNone(raised, out)
        self.assertTrue(seen, "serve() was never reached")
        self.assertTrue(seen["engine_lazy"], "the engine started loaded despite STRATA_LAZY_LOAD")
        self.assertIn("model unloaded; the first request loads it", out)

    def reach_lazy(self, cfg, env, argv_extra=()):
        """main() with the lazy start forced (no engine binary exists), so the Service that reaches serve()
        can be inspected; returns the same triple as run_main."""
        seen, out, raised = self.run_main({"lazy_load": True, **cfg}, env, argv_extra)
        self.assertIsNone(raised, out)
        self.assertTrue(seen, "serve() was never reached")
        return seen, out

    def test_lazy_default_stays_off(self):
        # no env, no flag, no config key: the byte-identical default. The engine is missing, so main()
        # fails at the start - after the ready line shows it chose the loaded path, not the lazy one
        _, out, raised = self.run_main({}, {})
        self.assertIsNotNone(raised)
        self.assertIn("loading the model", out)
        self.assertNotIn("the first request loads it", out)

    def test_idle_unload_from_the_environment(self):
        seen, _ = self.reach_lazy({}, {"STRATA_IDLE_UNLOAD_S": "1800"})
        self.assertEqual(seen["idle"], 1800.0)

    def test_the_config_key_wins_over_the_environment(self):
        seen, _ = self.reach_lazy({"idle_unload_s": 60}, {"STRATA_IDLE_UNLOAD_S": "1800"})
        self.assertEqual(seen["idle"], 60.0)

    def test_the_flag_wins_over_both(self):
        seen, _ = self.reach_lazy({"idle_unload_s": 60}, {"STRATA_IDLE_UNLOAD_S": "1800"},
                                  argv_extra=("--idle-unload", "10"))
        self.assertEqual(seen["idle"], 10.0)

    def test_a_typo_in_the_seconds_stops_the_start(self):
        _, out, raised = self.run_main({"lazy_load": True}, {"STRATA_IDLE_UNLOAD_S": "thirty"})
        self.assertIsInstance(raised, SystemExit)
        self.assertIn("STRATA_IDLE_UNLOAD_S='thirty'", str(raised))

    def test_webui_off_from_the_environment(self):
        seen, out = self.reach_lazy({}, {"STRATA_DISABLE_WEBUI": "1"})
        self.assertFalse(seen["webui"])
        self.assertIn("the web page is off", out)

    def test_webui_stays_on_by_default(self):
        seen, _ = self.reach_lazy({}, {})
        self.assertTrue(seen["webui"])

    def test_lazy_off_by_default_for_the_engine_object(self):
        seen, out, raised = self.run_main({"lazy_load": True}, {})   # the config key still works
        self.assertIsNone(raised, out)
        self.assertTrue(seen["engine_lazy"])


class LazyHealthcheck(unittest.TestCase):
    """docker/healthcheck.sh passes a lazy container: GET /v1/models must answer 200 with a non-empty list,
    the model shown as "unloaded" until the first request loads it."""

    @classmethod
    def setUpClass(cls):
        import serve.server as S
        tok = ByteTokenizer()
        # a lazy StrataEngine: never starts a process (.unloaded stays True), so no engine binary is needed
        cls.engine = S.StrataEngine("no-such-engine", ["--max-context", "4096"], lazy=True)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_models_lists_the_lazy_model_as_unloaded(self):
        with urllib.request.urlopen(self.base + "/v1/models", timeout=10) as r:
            data = json.loads(r.read())["data"]
        self.assertTrue(data, "the health check would call a lazy container unhealthy")
        self.assertEqual(data[0]["status"]["value"], "unloaded")


class WebPageOff(unittest.TestCase):
    """svc.webui = False: the page's own routes answer 404, everything a client or an operator uses stays."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.webui = False
        cls.svc.api_monitor = True        # /api/requests answers when its own switch is on: separate from the page
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read()

    def test_the_page_routes_answer_404(self):
        for path in ("/", "/api-monitor", "/web/app.js", "/web/i18n/en.json", "/fonts/inter.woff2"):
            self.assertEqual(self.get(path)[0], 404, path)

    def test_the_api_and_health_stay(self):
        self.assertEqual(self.get("/health")[0], 200)
        status, body = self.get("/v1/models")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["data"], "the health check's model list must not go empty")
        self.assertEqual(self.get("/metrics")[0], 200)
        self.assertEqual(self.get("/api/requests")[0], 200)   # the operator's route, not the page's

    def test_the_default_service_still_serves_the_page(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertTrue(svc.webui)                            # the field's default: byte-identical default path
        httpd = serve(svc, port=0)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/", timeout=10) as r:
                self.assertEqual(r.status, 200)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
