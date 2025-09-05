import importlib, sys
mods = ['backend.kalshi_trade','backend.finfill']
for m in mods:
    try:
        importlib.import_module(m)
        print(f'OK: imported {m}')
    except Exception as e:
        print(f'ERR importing {m}: {e}')
        raise
