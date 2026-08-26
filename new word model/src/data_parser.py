"""
Data parser for the IAM On-Line Handwriting Database.

Extracts and parses:
  - lineStrokes XML files: sequences of (x, y, time) pen coordinates per text line
  - ASCII text files: ground-truth transcriptions (CSR section = per-line text)

Pairs each stroke XML with its corresponding text label and caches the result.
"""

import os
import re
import tarfile
import pickle
import random
import xml.etree.ElementTree as ET
from collections import defaultdict

from . import config


def extract_archives():
    """Extract tar.gz archives if not already extracted."""
    for archive, target_dir in [
        (config.ASCII_ARCHIVE, config.ASCII_DIR),
        (config.STROKES_ARCHIVE, config.STROKES_DIR),
    ]:
        if os.path.isdir(target_dir) and os.listdir(target_dir):
            print(f"  [skip] {target_dir} already exists")
            continue
        print(f"  Extracting {os.path.basename(archive)} ...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=config.DATASET_DIR)
        print(f"  Done → {target_dir}")


def parse_stroke_xml(xml_path):
    """
    Parse a lineStrokes XML file and return a list of strokes.

    Each stroke is a list of dicts: {'x': int, 'y': int, 'time': float}

    Returns:
        list[list[dict]]: list of strokes, each stroke is a list of points
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    strokes = []
    for stroke_elem in root.iter("Stroke"):
        points = []
        for point_elem in stroke_elem.iter("Point"):
            points.append({
                "x": int(point_elem.get("x")),
                "y": int(point_elem.get("y")),
                "time": float(point_elem.get("time")),
            })
        if points:
            strokes.append(points)

    return strokes


def parse_ascii_labels(ascii_path):
    """
    Parse an ASCII transcription file and extract CSR lines.

    The file format has an OCR section and a CSR section separated by headers.
    We use the CSR (as-written) section which matches line-by-line with stroke files.

    Returns:
        list[str]: per-line text transcriptions from the CSR section
    """
    with open(ascii_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Extract the CSR section (comes after "CSR:\n\n")
    csr_match = re.search(r"CSR:\s*\n\s*\n(.*)", content, re.DOTALL)
    if not csr_match:
        return []

    csr_text = csr_match.group(1).strip()
    # Each non-empty line is the transcription for one handwritten line
    lines = [line.strip() for line in csr_text.split("\n") if line.strip()]
    return lines


def _find_ascii_file(form_id):
    """
    Find the ASCII file for a given form ID (e.g., 'a01-003').

    The ASCII files can have various suffixes: a01-003.txt, a01-003x.txt,
    a01-003w.txt, a01-003z.txt, etc. We prefer the plain version without
    a suffix, falling back to any available variant.
    """
    writer_id = form_id.split("-")[0]  # e.g., 'a01'
    form_dir = os.path.join(config.ASCII_DIR, writer_id, form_id)

    if not os.path.isdir(form_dir):
        return None

    # Look for plain file first (e.g., a01-003.txt)
    plain_file = os.path.join(form_dir, f"{form_id}.txt")
    if os.path.isfile(plain_file):
        return plain_file

    # Fall back to any .txt file in the directory
    for fname in sorted(os.listdir(form_dir)):
        if fname.endswith(".txt"):
            return os.path.join(form_dir, fname)

    return None


def _extract_form_and_line(xml_filename):
    """
    Extract form ID and line number from an XML filename.

    Example: 'a01-003-05.xml' → ('a01-003', 5)
             'a01-000u-01.xml' → ('a01-000u', 1)
    """
    base = os.path.splitext(xml_filename)[0]  # remove .xml
    # The last part after the final hyphen is the line number
    parts = base.rsplit("-", 1)
    if len(parts) != 2:
        return None, None

    form_id = parts[0]  # e.g., 'a01-003' or 'a01-000u'
    try:
        line_num = int(parts[1])
    except ValueError:
        return None, None

    return form_id, line_num


def _get_ascii_form_id(stroke_form_id):
    """
    Map a stroke form ID to the corresponding ASCII form ID.

    Stroke files may have suffix letters that match ASCII file suffixes.
    E.g., stroke form 'a01-000u' → ASCII dir 'a01-000', file 'a01-000u.txt'
    """
    # Check if the form_id has a letter suffix after the numeric part
    # Pattern: writer-number[letter] e.g., a01-000u
    match = re.match(r"^([a-z]\d+)-(\d+)([a-z]?)$", stroke_form_id)
    if not match:
        return stroke_form_id, stroke_form_id

    writer = match.group(1)
    num = match.group(2)
    suffix = match.group(3)

    base_form_id = f"{writer}-{num}"  # e.g., 'a01-000'
    full_form_id = stroke_form_id     # e.g., 'a01-000u'

    return base_form_id, full_form_id


def build_dataset():
    """
    Build the complete dataset by pairing stroke XMLs with text labels.

    Returns:
        list[dict]: each entry has:
            - 'strokes': list of strokes (list of point dicts)
            - 'label': text transcription string
            - 'xml_path': path to the source XML file
            - 'writer_id': writer identifier for train/val/test splitting
    """
    print("Building dataset...")

    # Step 1: Extract archives
    extract_archives()

    # Step 2: Collect all XML stroke files
    xml_files = []
    for root_dir, dirs, files in os.walk(config.STROKES_DIR):
        for fname in files:
            if fname.endswith(".xml"):
                xml_files.append(os.path.join(root_dir, fname))

    xml_files.sort()
    print(f"  Found {len(xml_files)} stroke XML files")

    # Step 3: Parse ASCII labels, grouped by form
    # Cache: form_id → {ascii_form_id: [lines]}
    ascii_cache = {}

    def get_label(stroke_form_id, line_num):
        """Get the text label for a given form and line number."""
        base_form_id, full_form_id = _get_ascii_form_id(stroke_form_id)

        cache_key = full_form_id
        if cache_key not in ascii_cache:
            # Try to find the ASCII file
            # First look for the full form id file in the base form directory
            writer_id = base_form_id.split("-")[0]
            form_dir = os.path.join(config.ASCII_DIR, writer_id, base_form_id)

            label_lines = []
            if os.path.isdir(form_dir):
                # Look for exact match first: e.g., a01-000u.txt
                exact_file = os.path.join(form_dir, f"{full_form_id}.txt")
                if os.path.isfile(exact_file):
                    label_lines = parse_ascii_labels(exact_file)
                else:
                    # Fall back to base form file: e.g., a01-000.txt
                    base_file = os.path.join(form_dir, f"{base_form_id}.txt")
                    if os.path.isfile(base_file):
                        label_lines = parse_ascii_labels(base_file)
                    else:
                        # Try any file in the directory
                        for f in sorted(os.listdir(form_dir)):
                            if f.endswith(".txt"):
                                label_lines = parse_ascii_labels(
                                    os.path.join(form_dir, f)
                                )
                                if label_lines:
                                    break

            ascii_cache[cache_key] = label_lines

        lines = ascii_cache[cache_key]
        # Line numbers are 1-indexed in the XML filenames
        idx = line_num - 1
        if 0 <= idx < len(lines):
            return lines[idx]
        return None

    # Step 4: Pair strokes with labels
    dataset = []
    skipped = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(xml_files, desc="  Parsing stroke XMLs")
    except ImportError:
        iterator = xml_files

    for xml_path in iterator:
        xml_fname = os.path.basename(xml_path)
        form_id, line_num = _extract_form_and_line(xml_fname)

        if form_id is None or line_num is None:
            skipped += 1
            continue

        # Get text label
        label = get_label(form_id, line_num)
        if label is None or len(label) == 0:
            skipped += 1
            continue

        # Filter out labels with characters not in our charset
        if not all(ch in config.CHAR_TO_IDX for ch in label):
            # Keep anyway but filter unknown chars
            label = "".join(ch for ch in label if ch in config.CHAR_TO_IDX)
            if len(label) == 0:
                skipped += 1
                continue

        # Parse strokes
        try:
            strokes = parse_stroke_xml(xml_path)
        except Exception as e:
            print(f"  [warn] Failed to parse {xml_path}: {e}")
            skipped += 1
            continue

        if not strokes:
            skipped += 1
            continue

        # Extract writer ID for writer-independent splitting
        writer_id = form_id.split("-")[0]

        dataset.append({
            "strokes": strokes,
            "label": label,
            "xml_path": xml_path,
            "writer_id": writer_id,
        })

    print(f"  Paired {len(dataset)} samples ({skipped} skipped)")
    return dataset


def split_dataset(dataset, seed=None):
    """
    Split dataset into train/val/test by writer ID (writer-independent split).

    This ensures no writer appears in multiple splits, preventing the model
    from memorizing individual handwriting styles.

    Returns:
        (train_data, val_data, test_data): three lists of sample dicts
    """
    if seed is None:
        seed = config.SEED
    rng = random.Random(seed)

    # Group by writer
    writer_samples = defaultdict(list)
    for sample in dataset:
        writer_samples[sample["writer_id"]].append(sample)

    # Shuffle writers
    writers = list(writer_samples.keys())
    rng.shuffle(writers)

    # Calculate split points based on number of samples per writer
    total = len(dataset)
    train_target = int(total * config.TRAIN_RATIO)
    val_target = int(total * config.VAL_RATIO)

    train_data, val_data, test_data = [], [], []
    train_count, val_count = 0, 0

    for writer in writers:
        samples = writer_samples[writer]
        if train_count < train_target:
            train_data.extend(samples)
            train_count += len(samples)
        elif val_count < val_target:
            val_data.extend(samples)
            val_count += len(samples)
        else:
            test_data.extend(samples)

    print(f"  Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return train_data, val_data, test_data


def load_or_build_dataset():
    """
    Load parsed dataset from cache, or build and cache it.

    Returns:
        (train_data, val_data, test_data)
    """
    if os.path.isfile(config.CACHE_FILE):
        print("Loading cached dataset...")
        with open(config.CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        print(f"  Loaded: train={len(data['train'])}, val={len(data['val'])}, test={len(data['test'])}")
        return data["train"], data["val"], data["test"]

    dataset = build_dataset()
    train_data, val_data, test_data = split_dataset(dataset)

    # Cache to disk
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(config.CACHE_FILE, "wb") as f:
        pickle.dump({
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }, f)
    print(f"  Cached to {config.CACHE_FILE}")

    return train_data, val_data, test_data


if __name__ == "__main__":
    train, val, test = load_or_build_dataset()

    # Print some samples
    print("\n--- Sample from training set ---")
    for i, sample in enumerate(train[:3]):
        n_points = sum(len(s) for s in sample["strokes"])
        print(f"  [{i}] Label: '{sample['label']}'")
        print(f"       Strokes: {len(sample['strokes'])}, Points: {n_points}")
        print(f"       Writer: {sample['writer_id']}")
        print(f"       File: {os.path.basename(sample['xml_path'])}")

