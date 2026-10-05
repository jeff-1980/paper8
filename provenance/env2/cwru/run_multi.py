# Runs several train_leakfree.py configs sequentially in ONE process (avoids repeated Mamba-3 kernel JIT).
import sys, runpy, traceback, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "cwru"))
for name in sys.argv[1:]:
    print(f"\n######## CONFIG {name} ########", flush=True)
    sys.argv = ["train_leakfree.py", "--config", f"cfg/cwru_fe/{name}.yaml"]
    try:
        runpy.run_path("scripts/cwru/train_leakfree.py", run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0): print("EXIT", e.code, flush=True)
    except Exception:
        traceback.print_exc()
