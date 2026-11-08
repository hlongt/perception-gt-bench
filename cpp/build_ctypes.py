"""Compile the C++17 kernels into a ctypes-loadable shared library.

Compiler discovery order: $GTB_CXX -> g++ -> clang++ -> cl -> zig c++
(``python -m ziglang c++``). The last one is a pip-installable, self-contained
toolchain, which makes the native backend reproducible on a clean Windows box.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

CPP_DIR = Path(__file__).resolve().parent
REPO = CPP_DIR.parent
INCLUDE = CPP_DIR / "include"
SOURCES = [CPP_DIR / "src" / "stereo_postprocess.cpp", CPP_DIR / "src" / "metrics.cpp"]
SUFFIX = ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
OUT = REPO / "src" / "gtbench" / ("_native" + SUFFIX)


def _zig_cmd():
    try:
        import ziglang  # noqa: F401
    except ImportError:
        return None
    r = subprocess.run(
        [sys.executable, "-m", "ziglang", "version"], capture_output=True, text=True
    )
    return [sys.executable, "-m", "ziglang", "c++"] if r.returncode == 0 else None


def find_compiler():
    env = os.environ.get("GTB_CXX")
    if env:
        return env.split()
    for name in ("g++", "clang++"):
        p = shutil.which(name)
        if p:
            return [p]
    if sys.platform == "win32" and shutil.which("cl"):
        return ["cl"]
    return _zig_cmd()


def build() -> tuple[bool, str]:
    cc = find_compiler()
    if cc is None:
        return False, "no C++ compiler found; install ziglang (pip install ziglang) or set GTB_CXX"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if cc and cc[0] == "cl":
        cmd = cc + ["/nologo", "/O2", "/EHsc", "/LD", f"/I{INCLUDE}"]
        cmd += [str(s) for s in SOURCES] + [f"/Fe:{OUT}"]
    else:
        cmd = cc + ["-shared", "-fPIC", "-O2", "-std=c++17", f"-I{INCLUDE}"]
        cmd += [str(s) for s in SOURCES] + ["-o", str(OUT)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout)[-2000:]
    return True, f"built {OUT} with {' '.join(cc)}"


if __name__ == "__main__":
    ok, msg = build()
    print(("OK: " if ok else "FAILED: ") + msg)
    sys.exit(0 if ok else 1)
