#!/usr/bin/env python3
import importlib.util
import sys

REQUIRED_MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "wholeslidedata": "wholeslidedata",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "pyyaml": "yaml",
}


def main() -> int:
    missing = []
    print(f"python={sys.executable}")
    for package_name, module_name in REQUIRED_MODULES.items():
        found = importlib.util.find_spec(module_name) is not None
        print(f"{package_name}: {'ok' if found else 'missing'}")
        if not found:
            missing.append(package_name)

    if missing:
        print("Missing dependencies:", ", ".join(missing))
        return 1

    print("Runtime check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
