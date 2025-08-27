# vis_base_json.py
# LVIS v1 train → LVIS-base(866) 파생 ann 생성기
# - base_texts(= lvis_v1_base_class_captions.json)의 모든 alias로 LVIS 카테고리 매칭
# - 괄호(… )/구두점/공백 정규화
# - 힌트(HINTS)로 흔한 미스 매칭 보완
# - 중복 매핑 회피(같은 LVIS 카테고리에 여러 base가 붙지 않게)
# 사용 예:
# python vis_base_json.py \
#   --src data/coco/annotations/lvis_v1_train.json \
#   --base_texts data/texts/lvis_v1_base_class_captions.json \
#   --dst data/coco/annotations/lvis_v1_train_base.json \
#   --drop_empty_images

import argparse
import collections
import difflib
import json
import re
from typing import Dict, List, Set, Tuple


def norm(s: str) -> str:
    """Normalize a class string:
    - lowercase
    - remove parenthesis content
    - keep only [a-z0-9] as tokens (others → space)
    - collapse spaces
    """
    s = s.lower()
    s = re.sub(r"\([^)]*\)", " ", s)        # remove (...) content
    s = re.sub(r"[^a-z0-9]+", " ", s)       # non-alnum → space
    s = re.sub(r"\s+", " ", s).strip()
    return s


# 흔한 미스 매칭 보정 힌트 (alias → LVIS 원문 카테고리명)
HINTS: Dict[str, str] = {
    "car": "car_(automobile)",
    "bus": "bus_(vehicle)",
    "camper": "camper_(vehicle)",
    "cab": "cab_(taxi)",
    "cap": "cap_(headwear)",
    "beef": "beef_(food)",
    "cayenne": "cayenne_(spice)",
    "blinder": "blinder_(for_horses)",
    # 사용 중 보고된 케이스
    "horned cow": "cow",
    "monitor": "monitor_(computer_equipment) computer_monitor",
    # 선택: 필요시 추가
    # "bow": "bow_(weapon)",   # 또는 "bow_(decorative_ribbons)"
    # "hair drier": "hair_dryer",
}


def build_canon_maps(categories: List[dict]) -> Tuple[Dict[str, Set[int]], Dict[int, str]]:
    """Build:
    - canon2ids: normalized name -> set(category_id)
    - id2name  : category_id -> original LVIS name
    """
    canon2ids: Dict[str, Set[int]] = collections.defaultdict(set)
    id2name: Dict[int, str] = {}
    for c in categories:
        cid = c["id"]
        name = c["name"]
        id2name[cid] = name
        canon2ids[norm(name)].add(cid)
    return canon2ids, id2name


def choose_best(cands: List[int], alias: str, id2name: Dict[int, str],
                assigned: Set[int]) -> int:
    """Pick best candidate category id for a given alias.
    - 우선 아직 배정 안된 cid 우선
    - difflib ratio로 alias vs original-name 유사도 최대
    - 동률이면 원문 이름이 짧은 것 우선
    """
    alias_n = norm(alias)
    # 1) 미배정 필터 우선
    unassigned = [cid for cid in cands if cid not in assigned]
    pool = unassigned if unassigned else cands

    def score(cid: int) -> Tuple[float, int]:
        name = id2name[cid]
        name_n = norm(name)
        sim = difflib.SequenceMatcher(None, alias_n, name_n).ratio()
        return (sim, -len(name))  # sim desc, length asc

    best = max(pool, key=score)
    return best


def row_aliases(row) -> List[str]:
    """row가 ["class", "alias1", ...] 또는 "class"일 수 있음 → 리스트로 통일"""
    if isinstance(row, list):
        return [str(x) for x in row if isinstance(x, (str, int))]
    elif isinstance(row, str):
        return [row]
    else:
        return [str(row)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="LVIS v1 train JSON (full)")
    ap.add_argument("--base_texts", required=True,
                    help="lvis_v1_base_class_captions.json")
    ap.add_argument("--dst", required=True,
                    help="Output LVIS-base train JSON")
    ap.add_argument("--drop_empty_images", action="store_true",
                    help="Drop images with no remaining annotations")
    ap.add_argument("--map_out", default=None,
                    help="Optional: write mapping CSV (base->LVIS)")
    args = ap.parse_args()

    # 1) Load LVIS full
    with open(args.src, "r") as f:
        lvis = json.load(f)
    images = lvis["images"]
    annotations = lvis["annotations"]
    categories = lvis["categories"]

    canon2ids, id2name = build_canon_maps(categories)

    # 2) Load base texts (list of lists)
    with open(args.base_texts, "r") as f:
        base_list = json.load(f)

    keep_cat_ids: Set[int] = set()
    assigned: Dict[int, str] = {}  # cid -> base primary name
    misses: List[List[str]] = []
    dup_rows: List[Tuple[str, int]] = []  # (base_primary, chosen_cid) when duplicate unavoidable
    mapping_rows: List[Tuple[str, int, str]] = []  # (base_primary, cid, lvis_name)

    for row in base_list:
        aliases = row_aliases(row)
        if not aliases:
            continue
        base_primary = aliases[0]
        found = False

        # (A) alias들로 직접 매칭
        for alias in aliases:
            key = norm(alias)
            if key in canon2ids and canon2ids[key]:
                cands = sorted(list(canon2ids[key]))
                chosen = choose_best(cands, alias, id2name, assigned=set(assigned.keys()))
                if chosen in assigned:
                    # 이미 다른 base가 쓴 cid → 불가피한 중복 (최대한 피하지만 허용)
                    dup_rows.append((base_primary, chosen))
                keep_cat_ids.add(chosen)
                assigned[chosen] = base_primary
                mapping_rows.append((base_primary, chosen, id2name[chosen]))
                found = True
                break

        if found:
            continue

        # (B) HINTS 사용
        for alias in aliases:
            a_l = alias.lower()
            if a_l in HINTS:
                hinted = HINTS[a_l]
                key = norm(hinted)
                if key in canon2ids and canon2ids[key]:
                    cands = sorted(list(canon2ids[key]))
                    chosen = choose_best(cands, hinted, id2name, assigned=set(assigned.keys()))
                    if chosen in assigned:
                        dup_rows.append((base_primary, chosen))
                    keep_cat_ids.add(chosen)
                    assigned[chosen] = base_primary
                    mapping_rows.append((base_primary, chosen, id2name[chosen]))
                    found = True
                    break

        if not found:
            # (C) 유사도 기반 완전 탐색(마지막 수단): alias vs 모든 LVIS 이름
            # 비용 고려해 가장 긴 alias 하나만 사용 (primary)
            alias = base_primary
            alias_n = norm(alias)
            best_cid, best_sim = None, -1.0
            for cid, name in id2name.items():
                name_n = norm(name)
                sim = difflib.SequenceMatcher(None, alias_n, name_n).ratio()
                if sim > best_sim:
                    best_sim, best_cid = sim, cid
            # 임계치 낮게라도 붙이고, 로그로 남김
            if best_cid is not None and best_sim >= 0.6:
                if best_cid in assigned:
                    dup_rows.append((base_primary, best_cid))
                keep_cat_ids.add(best_cid)
                assigned[best_cid] = base_primary
                mapping_rows.append((base_primary, best_cid, id2name[best_cid]))
                found = True

        if not found:
            misses.append(aliases)

    # 3) Filter annotations/images/categories
    anns_kept = [a for a in annotations if a["category_id"] in keep_cat_ids]
    if args.drop_empty_images:
        keep_img_ids = {a["image_id"] for a in anns_kept}
        imgs_kept = [img for img in images if img["id"] in keep_img_ids]
    else:
        imgs_kept = images
    cats_kept = [c for c in categories if c["id"] in keep_cat_ids]

    out = {"images": imgs_kept, "annotations": anns_kept, "categories": cats_kept}
    with open(args.dst, "w") as f:
        json.dump(out, f)

    # 4) Logs
    print(f"kept categories: {len(cats_kept)}, anns: {len(anns_kept)}, images: {len(imgs_kept)}")
    if misses:
        print("MISSING COUNT:", len(misses))
        ex = [m[0] if m else "" for m in misses[:20]]
        print("EXAMPLES:", ex)
    if dup_rows:
        print("DUPLICATE ASSIGNMENTS (base -> existing LVIS cid reused):", len(dup_rows))
        for base_primary, cid in dup_rows[:10]:
            print(f"  {base_primary} -> {cid} ({id2name[cid]})")

    if args.map_out:
        import csv
        with open(args.map_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["base_primary", "lvis_category_id", "lvis_name"])
            w.writerows(mapping_rows)
        print(f"wrote mapping csv -> {args.map_out}")


if __name__ == "__main__":
    main()
