import os
import subprocess
import sys
from pathlib import Path


def test_run_demo_script(tmp_path):
    out = tmp_path / "demo_output"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    res = subprocess.run([sys.executable, "examples/run_demo.py", "--out-dir", str(out)], env=env, check=True)
    assert out.exists()
    assert (out / "report.html").exists()
    assert (out / "results.csv").exists()