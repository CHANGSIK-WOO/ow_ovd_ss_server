# -*- coding: utf-8 -*-
"""
FOMO-style Attribute Generation & Embedding Builder
- Creates:
  1) att_embeddings .pth  -> {'att_text': [...], 'att_embedding': torch.FloatTensor [N, 512]}
  2) embedding_path .npy  -> np.float32 [K, 512] (class name text embeddings)
  3) distributions .pth   -> {'positive_distributions': [dict], 'negative_distributions': [dict]} (zero-initialized)
"""

import os, re, json, math, argparse
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---- Optional: OpenAI for attribute text generation ----
try:
    import openai
except Exception:
    openai = None

# ---- Attribute dedup ----
from fuzzywuzzy import fuzz  # pip install fuzzywuzzy
# (속도 원하면: from rapidfuzz import fuzz)

# ---- OpenCLIP for text embeddings ----
# pip install open_clip_torch
import open_clip
from tqdm import tqdm


# ========== I/O Utils ==========
def read_txt_file(filename: str) -> List[str]:
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip().split(",")[0] for line in f.read().splitlines() if line.strip()]


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ========== LLM (Alg.1) ==========
# 논문 템플릿(식 (3)): "I am using a language-vision model to identify {C}. List the {Z} attributes ..."
PROMPT_TMPL = (
    "I am using a language-vision model to identify '{cls}'. "
    "List the {category} attributes of '{cls}', which will be used for detection in the domain of {domain}. "
    "Return ONLY a Python list literal of short attribute phrases (no explanations)."
)

def call_openai_list_attributes(
    class_name: str, category: str, domain: str, model: str = "gpt-3.5-turbo-0125", temperature: float = 0.2
) -> List[str]:
    """Requests an attribute list from OpenAI Chat Completions; expects a Python list literal in the content."""
    if openai is None:
        raise RuntimeError("openai package is not installed. Install or provide pre-generated attributes JSON.")
    prompt = PROMPT_TMPL.format(cls=class_name, category=category, domain=domain)
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    text = resp["choices"][0]["message"]["content"].strip()

    # Try to extract a Python list literal safely
    # - simplest: find first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        literal = text[start : end + 1]
        try:
            data = json.loads(literal.replace("'", '"'))
        except Exception:
            # fallback: naive split; keep lines with letters
            data = [s.strip("-• \n\t") for s in re.split(r"[\n,]", text) if any(c.isalpha() for c in s)]
    else:
        data = [s.strip("-• \n\t") for s in re.split(r"[\n,]", text) if any(c.isalpha() for c in s)]

    # clean short phrases
    clean = []
    for s in data:
        s = re.sub(r"\s+", " ", s).strip()
        s = s.strip("“”\"'")
        if len(s) > 0:
            clean.append(s)
    return clean


# ========== Merge / Dedup ==========
def merge_attributes_per_category(attr_per_cls: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
    merged = defaultdict(list)
    for _, cat2vals in attr_per_cls.items():
        for cat, vals in cat2vals.items():
            merged[cat].extend(vals)
    # simple set dedup inside each category
    return {cat: sorted(set(vals)) for cat, vals in merged.items()}


def remove_similar_duplicates(
    merged: Dict[str, List[str]], threshold: int = 90
) -> Dict[str, List[str]]:
    """Fuzzy dedup across each category."""
    unique = {}
    for cat, values in merged.items():
        kept = []
        seen: List[str] = []
        for v in values:
            v_low = v.lower()
            if all(fuzz.token_set_ratio(v_low, s) < threshold for s in seen):
                kept.append(v)
                seen.append(v_low)
        unique[cat] = kept
    return unique


def flatten_attributes(unique: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """[(category, attribute), ...]"""
    pairs = []
    for cat, vals in unique.items():
        for v in vals:
            pairs.append((cat, v))
    return pairs


# ========== Build CLIP Embeddings ==========
def load_clip(clip_model: str = "ViT-B-16", pretrained: str = "openai", device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(clip_model)
    model.eval()
    return model, tokenizer, device


def encode_texts(texts: List[str], model, tokenizer, device: str, normalize: bool = True, batch_size: int = 256) -> torch.Tensor:
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch = texts[i : i + batch_size]
            tok = tokenizer(batch).to(device)
            txt = model.encode_text(tok)
            if normalize:
                txt = F.normalize(txt, dim=-1)
            embs.append(txt.float().cpu())
    return torch.cat(embs, dim=0) if len(embs) > 0 else torch.zeros(0, model.text_projection.shape[1])


# ========== Attribute Phrase Builder (paper template) ==========
def build_attribute_phrases(pairs: List[Tuple[str, str]]) -> List[str]:
    phrases = []
    for cat, att in pairs:
        cat_low = cat.lower().strip().rstrip(":")
        att_clean = att.strip().strip(".")
        # "object which color is red", "object which shape is round", ...
        phrases.append(f"object which {cat_low} is {att_clean}")
    return phrases


# ========== Class Text Embeddings ==========
PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a {}",
    "a cropped photo of a {}",
]

def build_classname_texts(classnames: List[str]) -> List[str]:
    texts = []
    for c in classnames:
        for t in PROMPT_TEMPLATES:
            texts.append(t.format(c))
    return texts

def average_by_class(emb: torch.Tensor, K: int) -> torch.Tensor:
    """Average T templates per class -> (K, D)."""
    T = len(PROMPT_TEMPLATES)
    assert emb.shape[0] == K * T
    return emb.view(K, T, -1).mean(dim=1)


# ========== Distributions (empty init) ==========
def init_empty_distributions(num_attr: int, thr_list: List[float], interval: float = 1e-4):
    bins = int(1.0 / interval)
    pos_list, neg_list = [], []
    for _ in thr_list:
        pos = {i: torch.zeros(bins, dtype=torch.float32) for i in range(num_attr)}
        neg = {i: torch.zeros(bins, dtype=torch.float32) for i in range(num_attr)}
        pos_list.append(pos)
        neg_list.append(neg)
    return {"positive_distributions": pos_list, "negative_distributions": neg_list}


# ========== Main Pipeline ==========
def main():
    p = argparse.ArgumentParser()
    # Inputs
    p.add_argument("--classnames_file", type=str, required=True,
                   help="txt file with one class name per line (Task-1 known classes).")
    p.add_argument("--domain", type=str, required=True,
                   help="Short domain description used in prompts (e.g., 'aerial images of airports').")

    # Attribute generation
    p.add_argument("--use_openai", action="store_true",
                   help="If set, call OpenAI to generate attributes; otherwise load from --unique_json if provided.")
    p.add_argument("--openai_api_key", type=str, default="",
                   help="OpenAI API key. If empty, falls back to environment OPENAI_API_KEY.")
    p.add_argument("--openai_model", type=str, default="gpt-3.5-turbo-0125")
    p.add_argument("--unique_json", type=str, default="unique_attributes.json",
                   help="Where to save/load merged+deduped attributes.")

    # Output paths
    p.add_argument("--att_embeddings_path", type=str, required=True,
                   help="Output .pth for attribute embeddings (dict with att_text, att_embedding).")
    p.add_argument("--embedding_path", type=str, required=True,
                   help="Output .npy for class text embeddings (K, D).")
    p.add_argument("--distributions_path", type=str, required=True,
                   help="Output .pth for empty distributions (ready for training log).")

    # CLIP
    p.add_argument("--clip_model", type=str, default="ViT-B-16")
    p.add_argument("--clip_pretrained", type=str, default="openai")

    # Others
    p.add_argument("--fuzzy_threshold", type=int, default=90)
    p.add_argument("--attr_categories", nargs="+", default=[
        "Shape", "Color", "Texture", "Size", "Context",
        "Features", "Appearance", "Behavior", "Environment", "Material"
    ])
    p.add_argument("--thr", type=float, default=0.8, help="Same thr used in your head (for distributions list shape).")

    args = p.parse_args()

    # Setup OpenAI if needed
    if args.use_openai:
        key = args.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OpenAI key not provided. Pass --openai_api_key or set OPENAI_API_KEY.")
        if openai is None:
            raise RuntimeError("openai package not installed. pip install openai")
        openai.api_key = key

    # Read classes
    classnames = read_txt_file(args.classnames_file)
    print(f"[INFO] Loaded {len(classnames)} class names from {args.classnames_file}")

    # ===== (i) Attribute Generation (Alg.1) =====
    if args.use_openai:
        print("[INFO] Generating attributes via OpenAI...")
        per_cls: Dict[str, Dict[str, List[str]]] = {}
        for c in tqdm(classnames, desc="Classes"):
            per_cls[c] = {}
            for cat in args.attr_categories:
                atts = call_openai_list_attributes(c, cat, args.domain, model=args.openai_model)
                per_cls[c][cat] = atts

        merged = merge_attributes_per_category(per_cls)
        unique = remove_similar_duplicates(merged, threshold=args.fuzzy_threshold)
        save_json(unique, args.unique_json)
        print(f"[OK] Saved merged+dedup attributes to {args.unique_json}")
    else:
        print("[INFO] Skipping OpenAI. Loading attributes from JSON...")
        if not os.path.exists(args.unique_json):
            raise FileNotFoundError(f"{args.unique_json} not found. Either run with --use_openai or provide this JSON.")
        with open(args.unique_json, "r", encoding="utf-8") as f:
            unique = json.load(f)

    # ===== Encode Attribute Texts =====
    pairs = flatten_attributes(unique)                    # [(category, attribute), ...]
    att_text = build_attribute_phrases(pairs)              # ["object which color is red", ...]
    print(f"[INFO] #attributes after merge/dedup: {len(att_text)}")

    clip_model, tokenizer, device = load_clip(args.clip_model, args.clip_pretrained)
    att_emb = encode_texts(att_text, clip_model, tokenizer, device=device, normalize=True)  # [N, D]

    # Save attribute embeddings .pth
    torch.save({"att_text": att_text, "att_embedding": att_emb}, args.att_embeddings_path)
    print(f"[OK] Saved attribute embeddings: {args.att_embeddings_path}  (shape={tuple(att_emb.shape)})")

    # ===== Class text embeddings (.npy) =====
    # prompt ensemble → average per class
    cls_texts = build_classname_texts(classnames)
    cls_emb_all = encode_texts(cls_texts, clip_model, tokenizer, device=device, normalize=True)
    cls_emb_avg = average_by_class(cls_emb_all, K=len(classnames)).cpu().numpy().astype("float32")
    np.save(args.embedding_path, cls_emb_avg)
    print(f"[OK] Saved class text embeddings: {args.embedding_path}  (shape={cls_emb_avg.shape})")

    # ===== Empty distributions (.pth) =====
    distr = init_empty_distributions(num_attr=att_emb.shape[0], thr_list=[args.thr], interval=1e-4)
    torch.save(distr, args.distributions_path)
    print(f"[OK] Saved empty distributions: {args.distributions_path}  (thr_list={[args.thr]}, num_attr={att_emb.shape[0]})")


if __name__ == "__main__":
    main()
