"""Check that all Python dependencies from pixi.toml are importable and satisfy version bounds."""

import importlib
import logging
import os
import sys
import warnings
from packaging.version import Version

PACKAGES = [
    # (import_name, pip_name, min_version, max_version)
    ("numpy", "numpy", "2.4.6", "3"),
    ("pandas", "pandas", "3.0.3", "4"),
    ("sklearn", "scikit-learn", "1.9.0", "2"),
    ("scipy", "scipy", "1.18.0", "2"),
    ("matplotlib", "matplotlib", "3.11.0", "4"),
    ("dowhy", "dowhy", "0.14", "0.15"),
    ("statsmodels", "statsmodels", "0.14.6", "0.15"),
    ("patsy", "patsy", "1.0.2", "2"),
    ("lifelines", "lifelines", "0.30.0", "0.31"),
    ("xgboost", "xgboost", "3.3.0", "4"),
    ("umap", "umap-learn", "0.5.12", "0.6"),
    ("ipykernel", "ipykernel", "7.3.0", "8"),
    ("seaborn", "seaborn", "0.13.2", "0.14"),
    ("pymc", "pymc", "6.0.1", "7"),
    ("arviz", "arviz", "1.2.0", "2"),
    ("bambi", "bambi", "0.18.0", "0.19"),
    ("shap", "shap", "0.49.1", "0.53"),
    ("networkx", "networkx", "3.6.1", "4"),
    ("keras", "keras", "3.15.0", "4"),
    ("tensorflow", "tensorflow", "2.21.0", "3"),
]

_WIN = os.name == "nt"

GREEN = "\033[32m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

_OK = "OK" if _WIN else "✓"
_FAIL = "FAIL" if _WIN else "✗"


def _check_version(version_str, min_ver, max_ver):
    v = Version(version_str)
    if min_ver and v < Version(min_ver):
        return False, f">={min_ver}"
    if max_ver and v >= Version(max_ver):
        return False, f"<{max_ver}"
    return True, None


def _quiet_import(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            return importlib.import_module(name)
        finally:
            logging.disable(logging.NOTSET)


def main():
    print(f"\n{BOLD}Checking Python packages…{RESET}\n")
    print(f"  Python {DIM}{sys.version.split()[0]}{RESET}\n")

    failures = []
    for import_name, pip_name, min_ver, max_ver in PACKAGES:
        mod = _quiet_import(import_name)
        if mod is None:
            failures.append(pip_name)
            print(f"  {RED}{_FAIL}{RESET}  {pip_name} — not installed")
            continue

        version = getattr(mod, "__version__", None)
        if version and (min_ver or max_ver):
            ok, bound = _check_version(version, min_ver, max_ver)
            if not ok:
                failures.append(pip_name)
                print(f"  {RED}{_FAIL}{RESET}  {pip_name} {version} — expected {bound}")
                continue

        ver_label = f" {DIM}{version}{RESET}" if version else ""
        print(f"  {GREEN}{_OK}{RESET}  {pip_name}{ver_label}")

    print()
    if failures:
        print(f"{RED}{BOLD}{len(failures)} package(s) failed:{RESET} {', '.join(failures)}")
        sys.exit(1)
    else:
        print(f"{GREEN}{BOLD}All {len(PACKAGES)} packages OK.{RESET}\n")

if __name__ == "__main__":
    main()