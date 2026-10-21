import importlib
for name in ["numpy","scipy","pandas","sklearn","mne","torch","yaml"]:
    try:
        m=importlib.import_module(name); print(f"{name}: OK ({getattr(m,'__version__','version unavailable')})")
    except Exception as e: print(f"{name}: ERROR ({e})")
