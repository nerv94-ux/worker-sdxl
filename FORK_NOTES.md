# Fork notes

Fork of `runpod-workers/worker-sdxl` (upstream last commit 2025-07-31).

Everything here exists because of one upstream behaviour: **`image_url` ran the refiner
instead of the base pipeline.** That single line invalidated img2img on this worker.

## What changed and why

### 1. `image_url` now runs img2img on the base checkpoint

Upstream:

```python
if starting_image:  # If image_url is provided, run only the refiner pipeline
    refiner_result = MODELS.refiner(prompt=..., strength=..., image=init_image, ...)
```

The README said "(runs only refiner)", so this was intentional — but it means:

- the **base checkpoint is never used** for img2img, so swapping in a custom
  checkpoint changes nothing on that path
- `negative_prompt`, `guidance_scale`, `width`, `height`, `num_images` are **dropped**
  (they're accepted by the schema, then ignored)
- the refiner is an end-of-schedule low-noise model (designed around strength ~0.2–0.3);
  ordinary img2img strengths (0.5+) are outside what it was trained for

Now `image_url` goes through `StableDiffusionXLImg2ImgPipeline` built from
`self.base.components`, and every parameter the schema accepts is passed through.
Sharing components means **no extra VRAM** — same UNet/VAE/text encoders already resident.

The text2img path still chains base → refiner exactly as before.

### 2. `refresh_worker` removed from the success path

Upstream set `results["refresh_worker"] = True` on every `image_url` job, killing the
worker afterwards. With `max_workers = 1` that's a full model reload (cold start) on
every reference-image request.

Still set on `RuntimeError` / unexpected exceptions — after a crash, recycling is correct.

### 3. `enable_model_cpu_offload()` removed

This endpoint targets ADA_24 (24GB). The pipeline fits with room to spare, so offloading
only adds CPU↔GPU round trips per forward pass.

It also matters for a likely next step: diffusers documents that `enable_model_cpu_offload()`
must be called **after** `load_ip_adapter()`, otherwise the image encoder gets offloaded
and errors. Removing it avoids that ordering trap entirely.

`enable_xformers_memory_efficient_attention()` is kept on all three pipelines.

### 4. VAE loaded once

`load_base()` and `load_refiner()` each pulled their own copy of
`madebyollin/sdxl-vae-fp16-fix`. The refiner now reuses `self.base.vae`.

### 5. CI rebuilt for forks

Upstream built only on GitHub *releases* and used `runs-on: DO`, a RunPod-internal
runner that doesn't exist here. Replaced with `.github/workflows/build.yml`:

- builds on every push to `main` (and manual dispatch)
- `runs-on: ubuntu-latest`, with a disk-cleanup step (the image is large)
- pushes two tags: `:latest` and `:<commit sha>`

**Point RunPod endpoints at the sha tag**, not `latest` — a redeploy should be an
explicit decision, not "whatever `latest` happens to be right now".

## Verifying the fix

Same seed, same reference image, before vs after. If the change took effect the outputs
differ — before, the base checkpoint wasn't involved at all.

Also check: cold-start frequency should drop sharply now that `refresh_worker` is gone
from the success path.

## Not changed (yet)

- Base checkpoint is still `stabilityai/stable-diffusion-xl-base-1.0`
- No IP-Adapter
- `compel` is still absent, so prompt weighting `(word:1.3)`, `BREAK`, and prompts
  over 77 tokens are **not** supported
