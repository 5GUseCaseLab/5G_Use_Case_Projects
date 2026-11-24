"""Run all offline tests:  python run_tests.py"""
import subprocess
import sys

ok = True
for t in ("tests/test_logic.py", "tests/test_area.py"):
    r = subprocess.run([sys.executable, t])
    ok &= r.returncode == 0
print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)
