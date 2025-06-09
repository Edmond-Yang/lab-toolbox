# model/__init__.py
import os, pkgutil, importlib

# 1) 限定 __path__ 只指向這個 model/ 資料夾
_pkg_dir = os.path.dirname(__file__)
__path__ = [_pkg_dir]

# 2) 從這個乾淨路徑裡掃描 .py / package
available_models = {
    name.lower(): name
    for _, name, ispkg in pkgutil.iter_modules([_pkg_dir])
    if  ispkg
}

model_name = os.environ.get("MODEL_NAME", "").lower()
# print(f"Loading model: {model_name!r}, available: {list(available_models.keys())}")

if model_name in available_models:
    print(f"Loading model: {model_name!r}")
    selected = available_models[model_name]
    # 3) 用絕對路徑 import：module = model.selected
    module = importlib.import_module(f"{__name__}.{selected}")
    # 4) 拿 __all__ 或 fallback
    export = getattr(module, "__all__", [n for n in dir(module) if not n.startswith("_")])
    for sym in export:
        globals()[sym] = getattr(module, sym)
else:
    raise f"No such model: {model_name!r}, available: {list(available_models.keys())}"
