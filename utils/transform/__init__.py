import os
import pkgutil
import importlib

_pkg_dir = os.path.dirname(__file__)
__path__ = [_pkg_dir]

aug_name = os.environ.get("AUG_NAME", "").lower()


available = {
    name.lower(): name
    for _, name, _ in pkgutil.iter_modules([_pkg_dir])
}


if aug_name in available:
    print(f'Augmentation {aug_name} applied. ')
    mod = importlib.import_module(f"{__name__}.{available[aug_name]}")
    symbols = getattr(mod, "__all__", [n for n in dir(mod) if not n.startswith("_")])
    for sym in symbols:
        globals()[sym] = getattr(mod, sym)
else:
    print('No Augmentation applied. ')
    def augment(rgb, nir, video_data, idx, length, num_augments):
        return video_data
