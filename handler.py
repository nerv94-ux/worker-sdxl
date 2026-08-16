import os
import base64

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
)
from diffusers.utils import load_image

from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from schemas import INPUT_SCHEMA

torch.cuda.empty_cache()


class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        # img2img built from the base pipeline's components -> ~0 extra VRAM.
        # Upstream routed image_url through the *refiner only*, which silently dropped
        # negative_prompt / guidance_scale / width / height and pushed the refiner well
        # outside its designed strength range (~0.2-0.3). See FORK_NOTES.md.
        self.img2img = None
        self.load_models()

    def load_base(self):
        # Load VAE from cache using identifier
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        # Load Base Pipeline from cache using identifier
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=True,
        ).to("cuda")
        
        # Enable memory optimizations
        base_pipe.enable_xformers_memory_efficient_attention()
        # NOTE: enable_model_cpu_offload() removed. This endpoint runs on 24GB (ADA_24)
        # where the whole pipeline fits, so offloading only adds CPU<->GPU round trips
        # per forward pass. It also breaks IP-Adapter if loaded before the adapter.

        return base_pipe

    def load_refiner(self):
        # Reuse the VAE already loaded for the base pipeline instead of loading a
        # second copy of the same weights.
        vae = self.base.vae
        # Load Refiner Pipeline from cache using identifier
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=True,
        ).to("cuda")
        
        # Enable memory optimizations
        refiner_pipe.enable_xformers_memory_efficient_attention()

        return refiner_pipe

    def load_models(self):
        self.base = self.load_base()
        self.refiner = self.load_refiner()
        # Share the base pipeline's components: same UNet/VAE/text encoders already on GPU.
        # This is what image_url should have been using all along.
        self.img2img = StableDiffusionXLImg2ImgPipeline(**self.base.components)
        self.img2img.enable_xformers_memory_efficient_attention()


MODELS = ModelHandler()


def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get("BUCKET_ENDPOINT_URL", False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
        "DPMSolverSinglestep": DPMSolverSinglestepScheduler.from_config(config),
    }[name]


@torch.inference_mode()
def generate_image(job):
    """
    Generate an image from text using your Model
    """
    # -------------------------------------------------------------------------
    # 🐞 DEBUG LOGGING
    # -------------------------------------------------------------------------
    import json, pprint

    # Log the exact structure RunPod delivers so we can see every nesting level.
    print("[generate_image] RAW job dict:")
    try:
        print(json.dumps(job, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job, depth=4, compact=False)

    # -------------------------------------------------------------------------
    # Original (strict) behaviour – assume the expected single wrapper exists.
    # -------------------------------------------------------------------------
    job_input = job["input"]

    print("[generate_image] job['input'] payload:")
    try:
        print(json.dumps(job_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job_input, depth=4, compact=False)

    # Input validation
    try:
        validated_input = validate(job_input, INPUT_SCHEMA)
    except Exception as err:
        import traceback

        print("[generate_image] validate(...) raised an exception:", err, flush=True)
        traceback.print_exc()
        # Re-raise so RunPod registers the failure (but logs are now visible).
        raise

    print("[generate_image] validate(...) returned:")
    try:
        print(json.dumps(validated_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(validated_input, depth=4, compact=False)

    if "errors" in validated_input:
        return {"error": validated_input["errors"]}
    job_input = validated_input["validated_input"]

    starting_image = job_input["image_url"]

    if job_input["seed"] is None:
        job_input["seed"] = int.from_bytes(os.urandom(2), "big")

    # Create generator with proper device handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(job_input["seed"])

    MODELS.base.scheduler = make_scheduler(
        job_input["scheduler"], MODELS.base.scheduler.config
    )

    if starting_image:
        # img2img on the BASE checkpoint (not the refiner).
        #
        # Upstream sent this through MODELS.refiner, which meant:
        #   - the base checkpoint was never used, so swapping checkpoints did nothing here
        #   - negative_prompt / guidance_scale / width / height / num_images were dropped
        #   - the refiner is a low-noise end-of-schedule model (~0.2-0.3); typical
        #     img2img strengths (0.5+) are outside what it was trained for
        # Everything the schema accepts is honoured now.
        init_image = load_image(starting_image).convert("RGB")
        MODELS.img2img.scheduler = make_scheduler(
            job_input["scheduler"], MODELS.img2img.scheduler.config
        )
        with torch.inference_mode():
            img2img_result = MODELS.img2img(
                prompt=job_input["prompt"],
                negative_prompt=job_input["negative_prompt"],
                image=init_image,
                strength=job_input["strength"],
                num_inference_steps=job_input["num_inference_steps"],
                guidance_scale=job_input["guidance_scale"],
                num_images_per_prompt=job_input["num_images"],
                generator=generator,
            )
            output = img2img_result.images
    else:
        try:
            # Generate latent image using base pipeline
            with torch.inference_mode():
                base_result = MODELS.base(
                    prompt=job_input["prompt"],
                    negative_prompt=job_input["negative_prompt"],
                    height=job_input["height"],
                    width=job_input["width"],
                    num_inference_steps=job_input["num_inference_steps"],
                    guidance_scale=job_input["guidance_scale"],
                    denoising_end=job_input["high_noise_frac"],
                    output_type="latent",
                    num_images_per_prompt=job_input["num_images"],
                    generator=generator,
                )
                image = base_result.images

            # Debug: Log tensor info
            if hasattr(image, 'dtype'):
                print(f"[DEBUG] Base output dtype: {image.dtype}, shape: {image.shape}", flush=True)
            elif isinstance(image, list) and len(image) > 0:
                print(f"[DEBUG] Base output list, first item dtype: {image[0].dtype}, shape: {image[0].shape}", flush=True)

            # Ensure latent images have correct dtype for refiner
            if hasattr(image, 'dtype') and hasattr(image, 'to'):
                image = image.to(dtype=torch.float16)
            elif isinstance(image, list) and len(image) > 0 and hasattr(image[0], 'dtype'):
                image = [img.to(dtype=torch.float16) for img in image]
            
            # Refine the image
            with torch.inference_mode():
                refiner_result = MODELS.refiner(
                    prompt=job_input["prompt"],
                    num_inference_steps=job_input["refiner_inference_steps"],
                    strength=job_input["strength"],
                    image=image,
                    num_images_per_prompt=job_input["num_images"],
                    generator=generator,
                )
                output = refiner_result.images
        except RuntimeError as err:
            print(f"[ERROR] RuntimeError in generation pipeline: {err}", flush=True)
            return {
                "error": f"RuntimeError: {err}, Stack Trace: {err.__traceback__}",
                "refresh_worker": True,
            }
        except Exception as err:
            print(f"[ERROR] Unexpected error in generation pipeline: {err}", flush=True)
            return {
                "error": f"Unexpected error: {err}",
                "refresh_worker": True,
            }

    image_urls = _save_and_upload_images(output, job["id"])

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input["seed"],
    }

    # NOTE: upstream set results["refresh_worker"] = True for every image_url job,
    # which kills the worker after each one. With max_workers=1 that means a cold
    # start (full model reload) on every reference-image request. Removed.

    return results


runpod.serverless.start({"handler": generate_image})
