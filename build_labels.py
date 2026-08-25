"""
build_labels.py

Step 1 of training: define the unified 47 + N class list and save it to
labels.json, so main.py, train.py, and sweep_test.py all agree on class order.

EDIT `SYMBOL_FOLDER_MAP` below after running:
    python emnist_symbols_dataset.py /path/to/extracted/symbols/dataset
to see the ACTUAL folder names in your downloaded dataset — CROHME-derived
datasets aren't perfectly consistent about naming (e.g. "times" vs "mult",
"decimal" vs "."), so don't trust this file's defaults blindly.
"""

import json
from torchvision.datasets import EMNIST

# ── EDIT THIS to match your extracted dataset's actual folder names ────────
# key   = folder name on disk (run list_symbol_folders() to check)
# value = display character the app will show / feed into equation eval
#
# NOTE: "times" (×) was deliberately REMOVED from this map. × and the
# letter X are the same shape — creating a separate × class alongside
# EMNIST's existing X class gave the model contradictory training signal
# for the identical stroke, which is why X (and things near it) were
# reading as garbage. A drawn X-shape now just resolves to the existing
# EMNIST "X" class — same as how a human reads it from context.
SYMBOL_FOLDER_MAP = {
    "+": "+",
    "-": "-",
    "div": "÷",
    "=": "=",
}
# ─────────────────────────────────────────────────────────────────────────

EMNIST_ROOT = "data"  # where torchvision will download EMNIST


def main():
    print("Fetching EMNIST ByMerge class list (downloads ~560MB if not cached)...")
    emnist = EMNIST(root=EMNIST_ROOT, split="bymerge", train=True, download=True)
    emnist_classes = list(emnist.classes)
    print(f"EMNIST ByMerge: {len(emnist_classes)} classes -> {emnist_classes}")

    symbol_classes = list(SYMBOL_FOLDER_MAP.values())
    print(f"Symbols: {len(symbol_classes)} classes -> {symbol_classes}")

    all_classes = emnist_classes + symbol_classes
    num_classes = len(all_classes)

    if len(set(all_classes)) != num_classes:
        dupes = [c for c in all_classes if all_classes.count(c) > 1]
        raise ValueError(f"Duplicate class labels detected: {set(dupes)}. "
                          f"Rename a symbol in SYMBOL_FOLDER_MAP to disambiguate.")

    labels_config = {
        "classes": all_classes,          # index -> display string, used directly by main.py
        "num_classes": num_classes,
        "emnist_class_count": len(emnist_classes),
        "symbol_folder_map": SYMBOL_FOLDER_MAP,
    }

    with open("labels.json", "w") as f:
        json.dump(labels_config, f, indent=2, ensure_ascii=False)

    print(f"\nSaved labels.json — {num_classes} total classes.")
    print("Next: run train.py")


if __name__ == "__main__":
    main()