import argparse
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig

class ArchitectureDataset(Dataset):
    def __init__(self, image_dir, metadata_file, tokenizer, resolution):
        self.image_dir = Path(image_dir)
        self.rows = pd.read_csv(metadata_file).dropna(subset=["file_name", "text"]).to_dict("records")
        if not self.rows:
            raise ValueError("Metadata CSV is empty or missing file_name/text columns.")
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = self.image_dir / str(row["file_name"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)
        tokens = self.tokenizer(
            str(row["text"]), max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids[0]
        return {"pixel_values": pixel_values, "input_ids": tokens}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--train_data_dir", required=True)
    p.add_argument("--metadata_file", required=True)
    p.add_argument("--output_dir", default="outputs/architecture-lora")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_steps", type=int, default=1500)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    return p.parse_args()

def main():
    args = parse_args()
    if args.resolution % 8:
        raise ValueError("resolution must be divisible by 8")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.eval()
    text_encoder.eval()
    unet.train()

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    trainable = [p for p in unet.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found. Check PEFT/Diffusers compatibility.")

    dataset = ArchitectureDataset(args.train_data_dir, args.metadata_file, tokenizer, args.resolution)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=args.max_train_steps
    )

    unet, optimizer, loader, lr_scheduler = accelerator.prepare(unet, optimizer, loader, lr_scheduler)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    while global_step < args.max_train_steps:
        for batch in loader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                input_ids = batch["input_ids"].to(accelerator.device)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    encoder_hidden_states = text_encoder(input_ids)[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                with accelerator.autocast():
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % 25 == 0:
                    accelerator.print(f"step={global_step}/{args.max_train_steps} loss={loss.detach().item():.6f}")
                if accelerator.is_main_process and args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                    ckpt = Path(args.output_dir) / f"checkpoint-{global_step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(unet)
                    lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped))
                    StableDiffusionPipeline.save_lora_weights(
                        save_directory=str(ckpt), unet_lora_layers=lora_state
                    )
            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(unet)
        lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir, unet_lora_layers=lora_state
        )
        print(f"LoRA attention processors saved to {args.output_dir}")

if __name__ == "__main__":
    main()
