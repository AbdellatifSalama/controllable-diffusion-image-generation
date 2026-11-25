import argparse
from pathlib import Path
import pandas as pd

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def main():
    p = argparse.ArgumentParser(description="Create an editable metadata CSV for architectural images.")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output_csv", default="data/metadata.csv")
    args = p.parse_args()
    image_dir = Path(args.image_dir)
    rows = []
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            rows.append({"file_name": path.name, "text": "REPLACE WITH AN ACCURATE ARCHITECTURAL CAPTION"})
    if not rows:
        raise SystemExit(f"No supported images found in {image_dir}")
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} rows to {out}. Edit every placeholder caption before training.")

if __name__ == "__main__":
    main()
