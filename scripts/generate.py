import argparse
import torch
from diffusers import StableDiffusionPipeline
from archdiff.utils import set_seed, ensure_parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default="low quality, blurry, distorted geometry, watermark, text")
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--lora_path")
    p.add_argument("--lora_scale", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--output", default="outputs/generated.png")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype, safety_checker=None)
    pipe = pipe.to(device)
    if args.lora_path:
        pipe.load_lora_weights(args.lora_path)
        pipe.fuse_lora(lora_scale=args.lora_scale)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    image = pipe(
        args.prompt, negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
        height=args.height, width=args.width, generator=generator
    ).images[0]
    ensure_parent(args.output)
    image.save(args.output)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
