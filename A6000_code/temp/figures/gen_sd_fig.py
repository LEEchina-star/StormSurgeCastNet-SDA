# -*- coding: utf-8 -*-
"""Generate framework-figure candidates with local Stable Diffusion (SD 1.5, MPS)."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
)
pipe = pipe.to("mps")
pipe.enable_attention_slicing()

prompt = (
    "a clean scientific diagram of a deep learning ocean storm surge forecasting model, "
    "left panel sparse coastal gauge stations and gridded wind field and colored ocean map, "
    "middle panel U-Net neural network, right panel dense forecast map, "
    "blue green red color scheme, flat vector, white background, high quality"
)
negative = "text, letters, words, labels, numbers, equations, blurry, low quality, watermark, signature, messy, distorted"

for seed in [0, 1, 2]:
    g = torch.Generator("mps").manual_seed(seed)
    img = pipe(prompt, negative_prompt=negative, num_inference_steps=30,
               guidance_scale=7.5, width=768, height=512, generator=g).images[0]
    img.save(f"temp/figures/sd_fig_seed{seed}.png")
    print(f"saved sd_fig_seed{seed}.png")
