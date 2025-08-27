#!/usr/bin/env python3
import argparse, json, os, shutil

MAP = {
    "train2017/": "images_train/",
    "val2017/":   "images_val/",
    "test2017/":   "images_test/",
    "\\train2017\\": "\\images_train\\",   # 윈도우 경로 보호
    "\\val2017\\":   "\\images_val\\",
    "\\test2017\\":   "\\images_test\\",
}

def rewrite_one(s: str) -> str:
    for k, v in MAP.items():
        s = s.replace(k, v)
    return s

def patch(data: dict) -> int:
    changed = 0
    for img in data.get("images", []):
        for key in ("file_name", "coco_url"):
            if isinstance(img.get(key), str):
                new = rewrite_one(img[key])
                if new != img[key]:
                    img[key] = new
                    changed += 1
    return changed

def main():
    ap = argparse.ArgumentParser(description="Rewrite LVIS paths in file_name (and coco_url).")
    ap.add_argument("inputs", nargs="+", help="LVIS json files (e.g., lvis_v1_train.json lvis_v1_val.json)")
    ap.add_argument("-o", "--outdir", help="Output dir. Omit to overwrite input (a .bak backup is kept).")
    args = ap.parse_args()

    for src in args.inputs:
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)

        n = patch(data)

        if args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
            dst = os.path.join(args.outdir, os.path.basename(src))
        else:
            # overwrite with backup
            dst = src
            bak = src + ".bak"
            if not os.path.exists(bak):
                shutil.copy2(src, bak)

        with open(dst, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"{src} -> {dst} | updated {n} fields")

if __name__ == "__main__":
    main()

