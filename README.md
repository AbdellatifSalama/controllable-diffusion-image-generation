# Controllable Architectural Image Generation

A reproducible research project for **text-to-image generation, architecture-style adaptation with LoRA, and sketch/edge-guided generation with ControlNet**.

> This repository is an implementation scaffold, not a report of completed experiments. No training metrics or generated results are claimed until you run the experiments and record them.

## Research questions

1. How does LoRA fine-tuning change a pretrained latent diffusion model's ability to generate a target architectural style?
2. How well does Canny ControlNet preserve the geometry of an architectural reference image?
3. What trade-offs appear between prompt alignment, structural consistency, inference time, and memory use?

## Pipeline

1. Prepare a licensed dataset of architectural images and captions.
2. Generate a baseline with Stable Diffusion 1.5.
3. Fine-tune UNet attention projections using LoRA.
4. Generate images with the adapted LoRA.
5. Apply Canny ControlNet to guide composition and edges.
6. Evaluate prompt alignment with CLIP and edge consistency with an edge-overlap metric. Treat both as imperfect proxies, not definitive measures of image quality.

## Repository layout

```text
.
├── configs/
│   └── default.yaml
├── data/
│   ├── README.md
│   └── metadata.example.csv
├── outputs/
│   └── .gitkeep
├── scripts/
│   ├── prepare_metadata.py
│   ├── train_lora.py
│   ├── generate.py
│   ├── generate_controlnet.py
│   └── evaluate.py
├── src/
│   └── archdiff/
│       ├── __init__.py
│       └── utils.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 1. Environment

Recommended: Linux with an NVIDIA GPU. A GPU with 16–24 GB VRAM is a practical starting point for the default 512px training configuration; memory needs vary by batch size, attention implementation, and precision.

```bash
git clone <your-repository-url>
cd controllable-diffusion-image-generation

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build appropriate for your CUDA driver from https://pytorch.org/get-started/locally/ if the default installation does not match your machine.

## 2. Prepare the architectural dataset

Use images you own or are licensed to use. Record the source and license for every dataset. Do not scrape websites or redistribute copyrighted images without permission.

Put images in `data/images/`. Create a CSV with exactly these columns:

```csv
file_name,text
building_001.jpg,"a contemporary minimalist house, concrete facade, large windows, daylight"
building_002.jpg,"a traditional stone courtyard house, arched openings, warm sunlight"
```

You can create a starter CSV from your image folder:

```bash
python scripts/prepare_metadata.py --image_dir data/images --output_csv data/metadata.csv
```

This creates editable placeholder captions. **Review and replace them with accurate descriptions**; generic filenames are not meaningful training captions.

Then split the data into `data/train/` and `data/validation/` if you plan to run a held-out evaluation. The included training script accepts one metadata CSV and image directory; keep a separate held-out set for your own evaluation.

## 3. Train LoRA

Default base model: `stable-diffusion-v1-5/stable-diffusion-v1-5` (confirm model access and license on Hugging Face before use).

```bash
accelerate config
accelerate launch scripts/train_lora.py \
  --pretrained_model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --train_data_dir data/images \
  --metadata_file data/metadata.csv \
  --output_dir outputs/architecture-lora \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --max_train_steps 1500 \
  --rank 16 \
  --mixed_precision fp16
```

Start with a small smoke test (e.g. `--max_train_steps 20`) before a full run. Inspect samples during development and watch for overfitting. Dataset size, caption quality, style consistency, and training duration matter more than blindly increasing steps.

## 4. Generate with the base model or LoRA

Baseline:

```bash
python scripts/generate.py \
  --prompt "a minimalist contemporary house, exposed concrete, warm daylight, architectural photography" \
  --output outputs/baseline.png
```

With LoRA:

```bash
python scripts/generate.py \
  --prompt "a minimalist contemporary house, exposed concrete, warm daylight, architectural photography" \
  --lora_path outputs/architecture-lora \
  --lora_scale 0.8 \
  --output outputs/lora.png
```

Use the same prompt, seed, resolution, and inference settings for fair comparisons.

## 5. ControlNet edge-guided generation

Supply an architectural photograph, drawing, or sketch as the control image:

```bash
python scripts/generate_controlnet.py \
  --control_image path/to/architectural_sketch.png \
  --prompt "a photorealistic modern concrete villa, natural materials, architectural photography" \
  --output outputs/controlnet.png
```

Optionally combine ControlNet with your trained LoRA:

```bash
python scripts/generate_controlnet.py \
  --control_image path/to/architectural_sketch.png \
  --prompt "a photorealistic modern concrete villa, natural materials" \
  --lora_path outputs/architecture-lora \
  --output outputs/controlnet_lora.png
```

## 6. Evaluation

CLIP score measures image-text embedding similarity; Canny edge F1 compares the output's edge map with the control edge map. Neither fully captures architectural correctness, aesthetics, or usability.

```bash
python scripts/evaluate.py \
  --image outputs/controlnet.png \
  --prompt "a photorealistic modern concrete villa, natural materials" \
  --control_image path/to/architectural_sketch.png \
  --output_json outputs/evaluation.json
```

For a meaningful study:
- Compare base, LoRA, ControlNet, and LoRA + ControlNet.
- Use a fixed prompt set and fixed seeds.
- Report mean and standard deviation across multiple prompts/seeds.
- Include qualitative grids and failure cases.
- Record GPU, software versions, runtime, peak memory, dataset source/license, and exact hyperparameters.
- Consider a small blinded human preference study if feasible.

## Reproducibility checklist

- [ ] Dataset provenance and license documented
- [ ] Captions manually checked
- [ ] Train/validation separation documented
- [ ] Baseline generated with fixed prompts and seeds
- [ ] LoRA training configuration and checkpoint saved
- [ ] ControlNet comparison completed
- [ ] Evaluation limitations discussed
- [ ] Results reproduced from a clean environment

## Limitations and responsible use

This is an educational research project, not a validated architectural design tool. Generated images may violate structural, safety, accessibility, or building-code requirements. They must not be used as construction plans. Model and dataset licenses apply independently; review them before training or publishing outputs.
