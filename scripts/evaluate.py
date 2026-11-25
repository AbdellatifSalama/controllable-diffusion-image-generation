import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import open_clip

def edge_f1(image_a, image_b):
    a = cv2.Canny(cv2.cvtColor(np.array(image_a.resize((512, 512))), cv2.COLOR_RGB2GRAY), 100, 200) > 0
    b = cv2.Canny(cv2.cvtColor(np.array(image_b.resize((512, 512))), cv2.COLOR_RGB2GRAY), 100, 200) > 0
    # Tolerant matching: dilate each edge map by a small radius.
    kernel = np.ones((5, 5), np.uint8)
    a_d = cv2.dilate(a.astype(np.uint8), kernel) > 0
    b_d = cv2.dilate(b.astype(np.uint8), kernel) > 0
    precision = (a & b_d).sum() / max(a.sum(), 1)
    recall = (b & a_d).sum() / max(b.sum(), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-12))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--control_image")
    p.add_argument("--output_json", default="outputs/evaluation.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    image = Image.open(args.image).convert("RGB")
    with torch.no_grad():
        image_features = model.encode_image(preprocess(image).unsqueeze(0).to(device))
        text_features = model.encode_text(tokenizer([args.prompt]).to(device))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_similarity = float((image_features @ text_features.T).item())

    result = {
        "image": args.image,
        "prompt": args.prompt,
        "clip_cosine_similarity": clip_similarity,
        "interpretation": "Embedding similarity proxy only; not a definitive measure of prompt adherence or image quality."
    }
    if args.control_image:
        control = Image.open(args.control_image).convert("RGB")
        result["canny_edge_f1"] = edge_f1(image, control)
        result["edge_metric_note"] = "Approximate edge overlap after dilation; output and control edges need not align perfectly due to perspective and generated detail."

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
