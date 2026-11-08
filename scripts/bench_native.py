from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gtbench import native  # noqa: E402
from gtbench.bench_native import benchmark  # noqa: E402

if __name__ == "__main__":
    print("native status:", native.status())
    print(json.dumps(benchmark(), ensure_ascii=False, indent=2))
