#!/usr/bin/env python
"""Pack a stratified percentage of ImageNet-1k (ILSVRC CLS-LOC) into ONE HDF5 file.

Motivation: clusters cap the *number* of files, not bytes, so 500k loose JPEGs is a
non-starter. This writes a single .h5 holding both splits.

Layout (per split group, "train" and "val"):
    data      uint8  (total_bytes,)   all encoded JPEGs concatenated
    offsets   int64  (N+1,)           image i is data[offsets[i]:offsets[i+1]]
    labels    int16  (N,)             index into wnids
    wnids     str    (C,)             class index -> WordNet id, sorted
    filenames str    (N,)             "<wnid>_<original_stem>.JPEG"

Every split is one flat, pre-shuffled pile of images -- no per-class subgroups -- and the
class is encoded in the stored filename, imagenette-style. That matters for the ~50k val
images, which ship as ILSVRC2012_val_00000293.JPEG and carry no class in their own name;
here they become n01751748_ILSVRC2012_val_00000293.JPEG, so the label survives even if the
images are later exported back to a flat directory.

The concatenated-bytes + offsets layout is deliberate: HDF5's variable-length dtype
stores every record in the global heap, which fragments and defeats chunk locality.
A flat byte array with an offset index gives an O(1) slice per image instead.

Preprocessing matches imagenette2-320: shortest side resized to --short-side, aspect
ratio preserved, re-encoded as JPEG. Use --store raw to skip decode cost at train time
(fixed-size square uint8 tensors) at the price of a much larger file.

Examples
--------
    # 40% of both splits, imagenette-style 320px shortest side
    python tools/build_imagenet_subset.py 0.4 /scratch/imagenet40.h5

    # 10% at 128px, stored pre-decoded for maximum dataloader throughput
    python tools/build_imagenet_subset.py 0.1 /scratch/in10_raw.h5 --store raw --raw-size 128
"""

import argparse
import io
import json
import math
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import h5py
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # a handful of ImageNet files trip the decompression-bomb guard

DEFAULT_ROOT = "/media/tico/BACKUP-DIDI/imageNet"

# Module-level so forked workers inherit it instead of pickling opts per image.
_OPTS = {}


# --------------------------------------------------------------------------------------
# Indexing the source tree
# --------------------------------------------------------------------------------------

def index_train(root):
    """{wnid: [abs_path, ...]} from Data/CLS-LOC/train/<wnid>/*.JPEG."""
    train_dir = os.path.join(root, "Data", "CLS-LOC", "train")
    if not os.path.isdir(train_dir):
        raise SystemExit(f"No train directory at {train_dir}")

    per_class = {}
    wnids = sorted(d for d in os.listdir(train_dir) if d.startswith("n"))
    for i, wnid in enumerate(wnids):
        cls_dir = os.path.join(train_dir, wnid)
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(".jpeg"))
        per_class[wnid] = [os.path.join(cls_dir, f) for f in files]
        if (i + 1) % 100 == 0:
            print(f"  indexed {i + 1}/{len(wnids)} classes", flush=True)
    return per_class


def index_val(root, cache_path=None):
    """{wnid: [abs_path, ...]} for the flat val directory.

    ImageNet's val split ships as 50k unlabeled files; the class lives in the CLS-LOC
    annotation XML (<object><name>nXXXXXXXX</name>). Parsing 50k small XMLs off a
    spinning/USB disk is slow, so the resulting map is cached next to the output.
    """
    val_dir = os.path.join(root, "Data", "CLS-LOC", "val")
    ann_dir = os.path.join(root, "Annotations", "CLS-LOC", "val")
    if not os.path.isdir(val_dir):
        raise SystemExit(f"No val directory at {val_dir}")

    labels = None
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path) as fh:
            labels = json.load(fh)
        print(f"  reusing cached val labels from {cache_path}")

    if labels is None:
        if not os.path.isdir(ann_dir):
            raise SystemExit(
                f"No val annotations at {ann_dir}. Without them the val split has no class "
                f"labels and cannot be stratified. Either restore Annotations/CLS-LOC/val "
                f"or build val by holding out part of train instead."
            )
        print("  parsing val annotations for class labels (one-time, then cached)...")
        labels = {}
        files = sorted(f for f in os.listdir(ann_dir) if f.endswith(".xml"))
        for i, fname in enumerate(files):
            try:
                obj = ET.parse(os.path.join(ann_dir, fname)).getroot().find("object")
                if obj is None:
                    continue
                labels[fname[:-4]] = obj.find("name").text.strip()
            except ET.ParseError:
                continue
            if (i + 1) % 10000 == 0:
                print(f"    {i + 1}/{len(files)}", flush=True)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w") as fh:
                json.dump(labels, fh)

    per_class = {}
    for fname in sorted(os.listdir(val_dir)):
        if not fname.lower().endswith(".jpeg"):
            continue
        wnid = labels.get(os.path.splitext(fname)[0])
        if wnid is None:
            continue
        per_class.setdefault(wnid, []).append(os.path.join(val_dir, fname))
    return per_class


def stratified_sample(per_class, fraction, seed, storage_order="shuffled"):
    """Take the same `fraction` from every class, so class balance is preserved exactly.

    Rounds up and keeps at least one image per class, so no class silently vanishes at
    small fractions.

    storage_order controls the order images are written in -- which is also the order they
    are READ from the source during the build, and that matters enormously on a hard disk:

      'shuffled' -- class-mixed on disk, so the file is usable with a cheap sequential
          reader or a shuffle buffer. Costs 1.28M random seeks across ~130 GB at build
          time; on a spinning disk that is the difference between a 3-hour and a 12-hour
          build.
      'disk'     -- keep source order (grouped by class). Storage order is then
          class-clustered, which is irrelevant when training with a map-style dataset and
          a full random sampler (shuffle=True) -- the access pattern is random either way.
          Build reads become near-sequential. Use this on an HDD.
    """
    rng = random.Random(seed)
    wnids = sorted(per_class)
    picked, labels = [], []
    for idx, wnid in enumerate(wnids):
        files = sorted(per_class[wnid])
        k = min(len(files), max(1, math.ceil(fraction * len(files))))
        chosen = rng.sample(files, k)
        picked.extend(sorted(chosen))
        labels.extend([idx] * k)

    if storage_order == "disk":
        return picked, labels, wnids

    order = list(range(len(picked)))
    rng.shuffle(order)
    return [picked[i] for i in order], [labels[i] for i in order], wnids


# --------------------------------------------------------------------------------------
# Per-image work (runs in pool workers)
# --------------------------------------------------------------------------------------

def encoded_name(path, wnid):
    """'<wnid>_<original_stem>.JPEG', the label-carrying name stored in the file.

    Train images are already named nXXXXXXXX_NNNN.JPEG, so they are left alone rather than
    doubled up; val images gain the prefix they never had.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith(wnid + "_"):
        return stem + ".JPEG"
    return f"{wnid}_{stem}.JPEG"


def _init_worker(opts):
    global _OPTS
    _OPTS = opts


def _resize_short_side(img, target, upscale=False):
    """Scale so the SHORTEST side becomes `target`, aspect ratio preserved.

    upscale=False keeps images that are already smaller than `target` untouched, which is the
    imagenette2-320 convention (fastai only ever downscales) and what the non-square jpeg path
    wants. upscale=True forces the resize in both directions -- required by
    _center_crop_square, which cannot crop a `target`-sized box out of a smaller image.
    """
    w, h = img.size
    if target <= 0:
        return img
    if min(w, h) <= target and not upscale:
        return img
    scale = target / min(w, h)
    # max(target, ...) guards the rounding: without it a 250x255 image can land on 255 and
    # leave the crop one pixel short.
    return img.resize((max(target, round(w * scale)), max(target, round(h * scale))),
                      Image.BICUBIC)


def _center_crop_square(img, size):
    """Shortest side -> `size` (aspect preserved), then center crop to `size` x `size`.

    The upscale=True is load-bearing: ~7.5% of ImageNet train images have a shortest side
    below 256. Without it those are left small and PIL's crop pads the out-of-range region
    with BLACK, silently corrupting ~96k images.
    """
    img = _resize_short_side(img, size, upscale=True)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def process_image(path):
    """-> (payload, ok). payload is JPEG bytes or a raw uint8 array; None when unreadable."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")  # normalizes the CMYK / grayscale / PNG-in-disguise files
            if _OPTS["store"] == "raw":
                arr = np.asarray(_center_crop_square(im, _OPTS["raw_size"]), dtype=np.uint8)
                return arr, True
            if _OPTS["square"]:
                im = _center_crop_square(im, _OPTS["short_side"])
            else:
                im = _resize_short_side(im, _OPTS["short_side"])
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=_OPTS["quality"], optimize=False)
            return buf.getvalue(), True
    except Exception as exc:  # truncated / corrupt source files: skip, do not abort the build
        print(f"  ! skipping {os.path.basename(path)}: {type(exc).__name__}: {exc}", flush=True)
        return None, False


# --------------------------------------------------------------------------------------
# Writing one split
# --------------------------------------------------------------------------------------

def write_split(h5, split, paths, labels, wnids, opts):
    grp = h5.create_group(split)
    n = len(paths)
    str_dt = h5py.string_dtype(encoding="utf-8")
    grp.create_dataset("wnids", data=np.array(wnids, dtype=object), dtype=str_dt)

    kept_labels, kept_names = [], []
    t0 = time.time()

    if opts["store"] == "raw":
        s = opts["raw_size"]
        images = grp.create_dataset(
            "images", shape=(n, s, s, 3), dtype=np.uint8,
            chunks=(1, s, s, 3),  # one image per chunk: a random read touches exactly one
        )
        write_at = 0
    else:
        # 256 KiB chunks. Records average ~60 KB at 320px, so a random read touches one or
        # two chunks -- small enough to keep read amplification low, large enough that the
        # chunk B-tree stays cheap.
        data = grp.create_dataset(
            "data", shape=(0,), maxshape=(None,), dtype=np.uint8,
            chunks=(opts["chunk_kb"] * 1024,),
        )
        buf, buf_len, total = [], 0, 0
        offsets = [0]
        flush_bytes = 256 * 1024 * 1024

        def flush():
            nonlocal buf, buf_len, total
            if not buf_len:
                return
            block = np.frombuffer(b"".join(buf), dtype=np.uint8)
            data.resize((total + block.size,))
            data[total:total + block.size] = block
            total += block.size
            buf, buf_len = [], 0

    with Pool(opts["workers"], initializer=_init_worker, initargs=(opts,)) as pool:
        for i, (payload, ok) in enumerate(
            pool.imap(process_image, paths, chunksize=32)  # imap preserves input order
        ):
            if ok:
                if opts["store"] == "raw":
                    images[write_at] = payload
                    write_at += 1
                else:
                    buf.append(payload)
                    buf_len += len(payload)
                    offsets.append(offsets[-1] + len(payload))
                    if buf_len >= flush_bytes:
                        flush()
                kept_labels.append(labels[i])
                kept_names.append(encoded_name(paths[i], wnids[labels[i]]))

            if (i + 1) % 5000 == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                print(f"  [{split}] {i + 1}/{n}  {rate:.0f} img/s  "
                      f"eta {(n - i - 1) / rate / 60:.1f} min", flush=True)

    if opts["store"] == "raw":
        images.resize((len(kept_labels), opts["raw_size"], opts["raw_size"], 3))
    else:
        flush()
        grp.create_dataset("offsets", data=np.asarray(offsets, dtype=np.int64))

    grp.create_dataset("labels", data=np.asarray(kept_labels, dtype=np.int16))
    grp.create_dataset("filenames", data=np.array(kept_names, dtype=object), dtype=str_dt)
    grp.attrs["num_images"] = len(kept_labels)
    grp.attrs["num_classes"] = len(wnids)
    grp.attrs["num_skipped"] = n - len(kept_labels)
    print(f"  [{split}] done: {len(kept_labels)} images, {n - len(kept_labels)} skipped, "
          f"{(time.time() - t0) / 60:.1f} min")
    return len(kept_labels)


# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Pack a stratified slice of ImageNet-1k into a single HDF5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("percentage", type=float,
                    help="fraction of EACH class to keep; accepts 0.4 or 40")
    ap.add_argument("output", help="destination .h5 path")
    ap.add_argument("--imagenet-root", default=DEFAULT_ROOT,
                    help="dir holding Data/CLS-LOC and Annotations/CLS-LOC")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--store", choices=["jpeg", "raw"], default="jpeg",
                    help="jpeg: compressed bytes, ~10x smaller. raw: pre-decoded uint8, no "
                         "decode cost at train time but fixed square resolution")
    ap.add_argument("--short-side", type=int, default=320,
                    help="jpeg mode: resize shortest side to this, aspect preserved "
                         "(imagenette2-320 convention). -1 keeps the original size")
    ap.add_argument("--square", action="store_true",
                    help="jpeg mode: center-crop to a --short-side square instead of keeping "
                         "the aspect ratio. Store exactly what the model consumes: smaller "
                         "file, no squashing, and no resize left to do at train time")
    ap.add_argument("--quality", type=int, default=90, help="jpeg mode: encoder quality")
    ap.add_argument("--raw-size", type=int, default=128,
                    help="raw mode: shortest side resized then center-cropped to this square")
    ap.add_argument("--chunk-kb", type=int, default=256, help="jpeg mode: HDF5 chunk size in KiB")
    ap.add_argument("--storage-order", choices=["shuffled", "disk"], default="shuffled",
                    help="'disk' reads the source in on-disk order instead of shuffling "
                         "first. On a spinning disk this is the single biggest build-speed "
                         "lever (~4x); storage order is irrelevant when training with a "
                         "map-style dataset and shuffle=True")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="On an HDD, MORE IS NOT BETTER: concurrent workers make the head "
                         "seek between their respective read positions. 4-6 is the sweet "
                         "spot there; on an SSD use all cores")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frac = args.percentage / 100.0 if args.percentage > 1.0 else args.percentage
    if not 0 < frac <= 1.0:
        raise SystemExit(f"percentage must land in (0, 1] or (0, 100]; got {args.percentage}")

    out = os.path.abspath(args.output)
    if not out.endswith((".h5", ".hdf5")):
        out += ".h5"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; move or delete it first")

    opts = {
        "store": args.store, "short_side": args.short_side, "quality": args.quality,
        "raw_size": args.raw_size, "workers": args.workers, "chunk_kb": args.chunk_kb,
        "square": args.square,
    }

    print(f"Source : {args.imagenet_root}")
    print(f"Output : {out}")
    print(f"Sample : {frac:.1%} per class | store={args.store} | workers={args.workers}\n")

    # Index and sample everything up front so a bad path fails before hours of encoding.
    plans = {}
    for split in args.splits:
        print(f"Indexing {split}...")
        if split == "train":
            per_class = index_train(args.imagenet_root)
        else:
            per_class = index_val(args.imagenet_root,
                                  cache_path=os.path.join(os.path.dirname(out),
                                                          ".imagenet_val_labels.json"))
        total = sum(len(v) for v in per_class.values())
        paths, labels, wnids = stratified_sample(per_class, frac, args.seed,
                                                 args.storage_order)
        plans[split] = (paths, labels, wnids)
        print(f"  {total} available in {len(wnids)} classes -> sampling {len(paths)}\n")

    with h5py.File(out, "w", libver="latest") as h5:
        h5.attrs.update({
            "source": args.imagenet_root, "percentage": frac, "seed": args.seed,
            "store": args.store, "short_side": args.short_side, "quality": args.quality,
            "raw_size": args.raw_size, "square": args.square,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "command": " ".join(sys.argv),
        })
        for split in args.splits:
            paths, labels, wnids = plans[split]
            print(f"Encoding {split} ({len(paths)} images)...")
            write_split(h5, split, paths, labels, wnids, opts)

    size_gb = os.path.getsize(out) / 1e9
    print(f"\nWrote {out}  ({size_gb:.1f} GB, 1 file)")


if __name__ == "__main__":
    main()
