"""
Build a YOLO-format dataset (images/ + labels/ + dataset.yaml) from the COCO animal
images you already downloaded.

Output layout:
    YOLO_DIR/
      images/train, images/dev, images/test
      labels/train, labels/dev, labels/test
      dataset.yaml

Splits (all from official COCO splits, so no pretrained-weight leakage on dev/test):
    train : random N_TRAIN animal images from train2017
    dev   : half of the animal images in val2017 (use for checkpoint selection / tuning)
    test  : other half of val2017 (touch once, for final numbers)
"""
import os
import random
import shutil
from pycocotools.coco import COCO

coco_data_loc = "/content/coco_data"
YOLO_DIR = "/content/yolo_animals"
N_TRAIN = 8000
SEED = 42


def split_dir(split):
    return f"{coco_data_loc}/data/{split}2017"


def collect(split):
    """Return (coco, animal_ids, image ids that are animal images AND downloaded)."""
    coco = COCO(f"{coco_data_loc}/annotations/instances_{split}2017.json")
    animal_ids = sorted(coco.getCatIds(supNms=["animal"]))      # 10 classes -> YOLO ids 0..9
    ids = set()
    for cid in animal_ids:
        ids.update(coco.getImgIds(catIds=[cid]))
    ids = sorted(i for i in ids
                 if os.path.exists(f"{split_dir(split)}/{coco.imgs[i]['file_name']}"))
    return coco, animal_ids, ids


def write_split(coco, animal_ids, img_ids, src_split, out_name):
    """Copy images and write YOLO label files. Returns number of images written."""
    cls_of = {cid: i for i, cid in enumerate(animal_ids)}
    img_out = f"{YOLO_DIR}/images/{out_name}"
    lbl_out = f"{YOLO_DIR}/labels/{out_name}"
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    n = 0
    for iid in img_ids:
        info = coco.imgs[iid]
        W, H = info["width"], info["height"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=iid, catIds=animal_ids, iscrowd=False))

        lines = []
        for a in anns:
            x, y, w, h = a["bbox"]                       # COCO: [x_min, y_min, w, h] in pixels
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, W), min(y + h, H)
            if x2 - x1 <= 1 or y2 - y1 <= 1:             # skip degenerate boxes
                continue
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            lines.append(f"{cls_of[a['category_id']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines:                                    # e.g. only crowd boxes -> skip image
            continue

        fname = info["file_name"]
        shutil.copy2(f"{split_dir(src_split)}/{fname}", f"{img_out}/{fname}")
        with open(f"{lbl_out}/{os.path.splitext(fname)[0]}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        n += 1
    return n


def main():
    yaml_path = f"{YOLO_DIR}/dataset.yaml"
    if os.path.exists(yaml_path):
        print(f"{yaml_path} already exists, skipping. Delete {YOLO_DIR} to rebuild.")
        return

    rng = random.Random(SEED)

    # ---- train: random subset of train2017 animal images ----
    coco, animal_ids, ids = collect("train")
    rng.shuffle(ids)
    n_tr = write_split(coco, animal_ids, ids[:N_TRAIN], "train", "train")
    names = [coco.cats[c]["name"] for c in animal_ids]

    # ---- dev/test: split the animal images of val2017 in half ----
    vcoco, vids_animals, vids = collect("val")
    rng.shuffle(vids)
    half = len(vids) // 2
    n_dev = write_split(vcoco, vids_animals, vids[:half], "val", "dev")
    n_test = write_split(vcoco, vids_animals, vids[half:], "val", "test")

    with open(yaml_path, "w") as f:
        f.write(f"path: {YOLO_DIR}\n")
        f.write("train: images/train\nval: images/dev\ntest: images/test\n")
        f.write("names:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")

    print(f"train: {n_tr} | dev: {n_dev} | test: {n_test} images")
    print(f"classes: {names}")
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
