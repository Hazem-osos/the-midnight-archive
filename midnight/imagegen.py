"""Local image generation: Dreamshaper-8 (SD 1.5, CreativeML OpenRAIL-M) + LCM-LoRA.

Low-VRAM friendly: fp16, attention slicing, VAE tiling, one model in memory at a
time. Works on CUDA (e.g. RTX 2060 6GB), Apple MPS, or (slowly) CPU.
Two passes: a 512x768 base image, then an img2img "hires" pass at 1.5x.
"""
from __future__ import annotations

import gc
import hashlib

from .core import MODELS, settings, torch_device


def _seed_for(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


class ImageGenerator:
    def __init__(self, log):
        import torch
        from diffusers import AutoPipelineForImage2Image, LCMScheduler, StableDiffusionPipeline

        self.log = log
        self.cfg = settings()["imagegen"]
        self.device = torch_device()
        self.dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        base = MODELS / "dreamshaper-8"
        if not (base / "model_index.json").exists():
            raise RuntimeError(f"Image model missing at {base}. Run: python archive.py setup-models")
        log.info(f"Loading image model on {self.device} ({self.dtype})")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base, torch_dtype=self.dtype, variant="fp16" if (base / "unet/diffusion_pytorch_model.fp16.safetensors").exists() else None,
            safety_checker=None, requires_safety_checker=False)
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights(str(MODELS / "lcm-lora-sdv1-5"), weight_name="pytorch_lora_weights.safetensors")
        self.pipe.fuse_lora()
        self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()
        try:
            self.pipe.vae.enable_tiling()
        except Exception:
            pass
        self.pipe.set_progress_bar_config(disable=True)
        self.img2img = AutoPipelineForImage2Image.from_pipe(self.pipe)
        self.img2img.set_progress_bar_config(disable=True)
        self.torch = torch

    def generate(self, prompt: str, out_path, seed: int | None = None, negative: str | None = None,
                 style: bool = True, steps: int | None = None, guidance: float | None = None):
        c = self.cfg
        full = f"{prompt}, {c['style_suffix']}" if style else prompt
        neg = negative or c["negative_prompt"]
        seed = seed if seed is not None else _seed_for(prompt)
        gen = self.torch.Generator("cpu").manual_seed(int(seed))
        img = self.pipe(full, negative_prompt=neg, width=c["width"], height=c["height"],
                        num_inference_steps=steps or c["steps"], guidance_scale=guidance or c["guidance"],
                        generator=gen).images[0]
        if c.get("hires_scale", 1) > 1:
            w = int(c["width"] * c["hires_scale"]) // 8 * 8
            h = int(c["height"] * c["hires_scale"]) // 8 * 8
            up = img.resize((w, h), resample=3)
            gen = self.torch.Generator("cpu").manual_seed(int(seed) + 1)
            # LCM img2img: effective steps = steps * strength, so ask for more.
            img = self.img2img(full, negative_prompt=neg, image=up, strength=c["hires_strength"],
                               num_inference_steps=max(2, int(round(c["hires_steps"] / c["hires_strength"]))),
                               guidance_scale=guidance or c["guidance"], generator=gen).images[0]
        img.save(out_path)
        return seed

    def close(self):
        del self.pipe, self.img2img
        gc.collect()
        try:
            if self.device == "cuda":
                self.torch.cuda.empty_cache()
            elif self.device == "mps":
                self.torch.mps.empty_cache()
        except Exception:
            pass
