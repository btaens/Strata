"""Build-time patch of the installed Terminal: lazy loading and idle unload from the environment, and --no-browser."""
import py_compile
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/strata/terminal")

OPEN_SERVER = '    ap.add_argument("--open", action="store_true", help="open the local page in the browser once the model is ready")\n'
OPEN_LAUNCH = ('    o.add_argument("--open", action="store_true", help="open the server\'s web page in your browser once it is serving '
               '(nothing opens a browser without this)")\n')

SERVER_EDITS = [
    (
        'lazy = a.lazy or cfg.get("lazy_load") is True\n',
        'lazy = a.lazy or cfg.get("lazy_load") is True or \\\n'
        '               os.environ.get("STRATA_LAZY_LOAD", "").strip().lower() in ("1", "true", "yes", "on")\n',
    ),
    (
        'svc.idle_unload_s = a.idle_unload if a.idle_unload is not None else float(cfg.get("idle_unload_s") or 0)\n',
        'svc.idle_unload_s = a.idle_unload if a.idle_unload is not None else \\\n'
        '        float(cfg.get("idle_unload_s") or os.environ.get("STRATA_IDLE_UNLOAD_S") or 0)\n',
    ),
    (
        OPEN_SERVER,
        OPEN_SERVER + '    ap.add_argument("--no-browser", action="store_false", dest="open", help="never open a browser (the default)")\n',
    ),
    (
        '        def do_GET(self):\n            path = self.path.split("?")[0].rstrip("/")\n',
        '        def do_GET(self):\n            path = self.path.split("?")[0].rstrip("/")\n'
        '            if os.environ.get("STRATA_DISABLE_WEBUI", "").strip().lower() in ("1", "true", "yes", "on") and \\\n'
        '                    (path in ("", "/api-monitor") or path.startswith(("/web/", "/fonts/"))):\n'
        '                self._json(404, {"error": {"message": "not found"}})\n'
        '                return\n',
    ),
]

LAUNCH_EDITS = [
    (
        OPEN_LAUNCH,
        OPEN_LAUNCH + '    o.add_argument("--no-browser", action="store_false", dest="open", help="never open a browser (the default)")\n',
    ),
]

EDITS = {"serve/server.py": SERVER_EDITS, "strata_cli/launch/cmd.py": LAUNCH_EDITS}

for rel, edits in EDITS.items():
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    for old, new in edits:
        if text.count(old) != 1:
            sys.exit(f"patch_server.py: expected exactly one match in {path} for: {old.strip()}\n"
                     "(this Strata version differs from the one this patch was written for)")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    print("patched", path)
