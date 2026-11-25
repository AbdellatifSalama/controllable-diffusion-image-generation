import argparse
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image
from archdiff.utils import set_seed, ensure_parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control_image", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default="low quality, blurry, distorted geometry, watermark, text")
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--controlnet", default="lllyasviel/sd-controlnet-canny")
    p.add_argument("--lora_path")
    p.add_argument("--controlnet_conditioning_scale", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/controlnet.png")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.model, controlnet=controlnet, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if args.lora_path:
        pipe.load_lora_weights(args.lora_path)
        pipe.fuse_lora()
    control_image = Image.open(args.control_image).convert("RGB").resize((512, 512))
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipe(
        args.prompt, negative_prompt=args.negative_prompt,
        image=control_image, num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        generator=generator
    ).images[0]
    ensure_parent(args.output)
    result.save(args.output)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
