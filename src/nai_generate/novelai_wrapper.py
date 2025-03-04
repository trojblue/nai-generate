# src/procslib/models/novelai_wrapper.py

import asyncio
import logging
import os
import random
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union
from zipfile import ZipFile
import pandas as pd

from PIL import Image
from curl_cffi.requests import AsyncSession
import re

logger = logging.getLogger(__name__)


class ApiCredential:
    """
    Simple credential class for NovelAI API using persistent token
    """
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.x_correlation_id = "".join([chr(ord('a') + int(random.random() * 26)) for _ in range(6)])
    
    async def get_session(self, timeout: int = 180, update_headers: dict = None):
        """
        Get a session with proper authentication headers
        """
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        
        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0",
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Origin": "https://novelai.net",
            "Referer": "https://novelai.net/",
            "x-correlation-id": self.x_correlation_id,
            "x-initiated-at": timestamp,
        }

        if update_headers:
            assert isinstance(update_headers, dict), "update_headers must be a dict"
            headers.update(update_headers)

        return AsyncSession(timeout=timeout, headers=headers, impersonate="chrome110")


class NovelAIGenerator:
    
    DEFAULT_ENDPOINT = "https://image.novelai.net/ai/generate-image"
    
    def __init__(self, persistent_token: str):
        """
        Initialize the NovelAI image generator with a persistent token.
        
        Args:
            persistent_token: The persistent token starting with "pst-"
        """
        self.persistent_token = persistent_token
        self.endpoint = os.environ.get("NOVELAI_ENDPOINT", self.DEFAULT_ENDPOINT)
        self.correlation_id = "".join([chr(ord('a') + int(random.random() * 26)) for _ in range(6)])
        
    async def generate_image(
        self, 
        prompt: str, 
        negative_prompt: str = "", 
        seed: int = None,
        width: int = 832,
        height: int = 1216,
        sampler: str = "k_euler_ancestral",
        steps: int = 28,
        scale: float = 5,
        model: str = "nai-diffusion-4-full"
    ):
        """
        Generate an image using NovelAI's API
        
        Args:
            prompt: The text prompt to generate an image from
            negative_prompt: The negative prompt to avoid certain elements
            seed: Random seed for reproducibility (optional)
            width: Image width
            height: Image height
            sampler: The sampling method
            steps: Number of diffusion steps
            scale: CFG scale
            model: Model name to use
            
        Returns:
            List of tuples containing (filename, image_data)
        """
        # Create timestamp in ISO format with Z suffix
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        
        # Set headers using persistent token
        headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-CA,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh;q=0.6",
            "authorization": f"Bearer {self.persistent_token}",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": "https://novelai.net",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://novelai.net/",
            "sec-ch-ua": "\"Not(A:Brand\";v=\"99\", \"Microsoft Edge\";v=\"133\", \"Chromium\";v=\"133\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0",
            "x-correlation-id": self.correlation_id,
            "x-initiated-at": timestamp
        }
        
        # Use provided seed or generate a random one
        if seed is None:
            seed = random.randint(0, 4294967295)
        
        # Create payload matching the captured request
        payload = {
            "input": prompt,
            "model": model,
            "action": "generate",
            "parameters": {
                "params_version": 3,
                "width": width,
                "height": height,
                "scale": scale,
                "sampler": sampler,
                "steps": steps,
                "n_samples": 1,
                "ucPreset": 0,
                "qualityToggle": False,
                "dynamic_thresholding": False,
                "controlnet_strength": 1,
                "legacy": False,
                "add_original_image": True,
                "cfg_rescale": 0,
                "noise_schedule": "karras",
                "legacy_v3_extend": False,
                "skip_cfg_above_sigma": None,   # skip_cfg_above_sigma=19 if variety_boost else None,
                "use_coords": False,
                "seed": seed,
                "characterPrompts": [],
                "v4_prompt": {
                    "caption": {
                        "base_caption": prompt,
                        "char_captions": []
                    },
                    "use_coords": False,
                    "use_order": True
                },
                "v4_negative_prompt": {
                    "caption": {
                        "base_caption": negative_prompt,
                        "char_captions": []
                    }
                },
                "negative_prompt": negative_prompt,
                "reference_image_multiple": [],
                "reference_information_extracted_multiple": [],
                "reference_strength_multiple": [],
                "deliberate_euler_ancestral_bug": False,
                "prefer_brownian": True
            }
        }
        
        # Create curl_cffi session
        async with AsyncSession(headers=headers, impersonate="chrome110") as session:
            logger.debug("Sending request to NovelAI...")
            
            # Send request
            response = await session.post(
                self.endpoint,
                json=payload
            )
            
            logger.debug(f"Response status: {response.status_code}")
            
            # Check if response is successful
            if response.status_code == 200:
                # Response is a ZIP file containing the generated image(s)
                try:
                    # Create a BytesIO object from the response content
                    zip_bytes = BytesIO(response.content)
                    
                    # Open the ZIP file
                    with ZipFile(zip_bytes) as zip_file:
                        # Get the list of files in the ZIP
                        file_list = zip_file.namelist()
                        logger.debug(f"Files in ZIP: {file_list}")
                        
                        # Read and return each file
                        result = []
                        for file_name in file_list:
                            image_data = zip_file.read(file_name)
                            result.append((file_name, image_data))
                        
                        return result
                except Exception as e:
                    logger.error(f"Error processing response: {e}")
                    return None
            else:
                logger.error(f"Error generating image: {response.status_code}")
                try:
                    logger.error(f"Response: {response.text}")
                except:
                    logger.error(f"Raw response: {response.content}")
                return None


class NovelAIInference():
    """A wrapper for NovelAI text-to-image generation.
    Generates images from text prompts using the NovelAI API.
    """
    
    def __init__(
        self,
        persistent_token: str = None,
        default_width: int = 832,
        default_height: int = 1216,
        default_steps: int = 28,
        default_sampler: str = "k_euler_ancestral",
        default_model: str = "nai-diffusion-4-full",
        default_negative_prompt: str = "blurry, lowres, error, worst quality, bad quality"
    ):
        """Initialize NovelAI inference with API token and default generation parameters.
        
        Args:
            device: Not used (keeping for compatibility)
            batch_size: Not used (keeping for compatibility)
            persistent_token: NovelAI persistent token (pst-*)
            default_width: Default image width
            default_height: Default image height
            default_steps: Default number of diffusion steps
            default_sampler: Default sampler method
            default_model: Default model name
            default_negative_prompt: Default negative prompt
        """       
        # Enable nested asyncio for Jupyter notebooks and similar environments
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            logger.warning("nest_asyncio not available. This may cause issues in Jupyter environments.")
        
        if persistent_token is None:
            # Try to get from environment variable
            persistent_token = os.environ.get("NOVELAI_TOKEN")
            
        if not persistent_token:
            raise ValueError("NovelAI persistent token must be provided either in initialization or as NOVELAI_TOKEN environment variable")
            
        self._load_model(persistent_token)
        
        # Store default parameters
        self.default_width = default_width
        self.default_height = default_height
        self.default_steps = default_steps
        self.default_sampler = default_sampler
        self.default_model = default_model
        self.default_negative_prompt = default_negative_prompt

    def _load_model(self, checkpoint_path: str):
        """Load the NovelAI generator with the given token.
        
        Args:
            checkpoint_path: The NovelAI persistent token
        """
        self.persistent_token = checkpoint_path
        self.generator = NovelAIGenerator(self.persistent_token)
        logger.info("NovelAI generator initialized")
    
    
    async def _generate_image_async(
        self, 
        prompt: str,
        negative_prompt: str = None,
        seed: int = None,
        width: int = None,
        height: int = None,
        steps: int = None,
        sampler: str = None,
        model: str = None,
    ):
        """Async version of generate_image."""
        # Use default values if not specified
        negative_prompt = negative_prompt or self.default_negative_prompt
        width = width or self.default_width
        height = height or self.default_height
        steps = steps or self.default_steps
        sampler = sampler or self.default_sampler
        model = model or self.default_model
        
        # Generate the image
        return await self.generator.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            sampler=sampler,
            model=model
        )

    def generate_image(
        self, 
        prompt: str,
        negative_prompt: str = None,
        seed: int = None,
        width: int = None,
        height: int = None,
        steps: int = None,
        sampler: str = None,
        model: str = None,
        save_path: str = None
    ) -> Image.Image:
        """Generate a single image from a text prompt.
        
        Args:
            prompt: Text prompt to generate an image from
            negative_prompt: Negative prompt (optional)
            seed: Random seed (optional)
            width: Image width (optional)
            height: Image height (optional)
            steps: Number of diffusion steps (optional)
            sampler: Sampling method (optional)
            model: Model name (optional)
            save_path: Path to save the generated image (optional)
            
        Returns:
            PIL.Image: The generated image
        """
        # Run the async function and get the result
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            # We're in an async context, use a Future
            import nest_asyncio
            nest_asyncio.apply()  # Allow nested event loops
            result = loop.run_until_complete(self._generate_image_async(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                width=width,
                height=height,
                steps=steps,
                sampler=sampler,
                model=model
            ))
        else:
            # Not in an async context, just run the coroutine
            result = loop.run_until_complete(self._generate_image_async(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                width=width,
                height=height,
                steps=steps,
                sampler=sampler,
                model=model
            ))
        
        if not result:
            logger.error("Failed to generate image")
            return None
        
        # Get the first (and usually only) image
        filename, image_data = result[0]
        
        # Save the image if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(image_data)
            logger.info(f"Saved generated image to {save_path}")
        
        # Return the image as a PIL Image
        return Image.open(BytesIO(image_data))
    
    async def _generate_multiple_async(
        self,
        prompts: List[str],
        negative_prompt: str = None,
        seeds: List[int] = None,
        width: int = None,
        height: int = None,
        steps: int = None,
        sampler: str = None,
        model: str = None,
    ) -> List[Tuple[str, int, List]]:
        """Async version to generate multiple images."""
        # If seeds not provided, generate random ones
        if seeds is None:
            seeds = [random.randint(0, 4294967295) for _ in range(len(prompts))]
        elif len(seeds) < len(prompts):
            # Extend seeds if needed
            seeds = seeds + [random.randint(0, 4294967295) for _ in range(len(prompts) - len(seeds))]
        
        results = []
        
        # We'll use gather to run requests concurrently 
        # but with lower concurrency to avoid rate limits
        from asyncio import Semaphore, gather
        
        # Limit concurrent requests
        sem = Semaphore(2)  # Only 2 concurrent requests
        
        async def generate_with_semaphore(prompt, seed):
            async with sem:
                result = await self._generate_image_async(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    width=width,
                    height=height,
                    steps=steps,
                    sampler=sampler,
                    model=model
                )
                return (prompt, seed, result)
        
        # Create tasks for all images
        tasks = [generate_with_semaphore(prompt, seed) for prompt, seed in zip(prompts, seeds)]
        
        # Run tasks and get results
        completed_results = await gather(*tasks)
        
        return completed_results

    def get_next_id(self, output_dir: str) -> int:
        """Determine the next available ID based on existing files in the directory."""
        existing_ids = []
        pattern = re.compile(r'generated_(\d{4})_\d+_.*\.png')
        
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                match = pattern.match(filename)
                if match:
                    existing_ids.append(int(match.group(1)))
        
        return max(existing_ids, default=0) + 1

    def generate_multiple(
        self, 
        prompts: List[str],
        negative_prompt: str = None,
        seeds: List[int] = None,
        width: int = None,
        height: int = None,
        steps: int = None,
        sampler: str = None,
        model: str = None,
        output_dir: str = None,
        wait: float = 1.5
    ) -> List[Dict]:
        """Generate multiple images from text prompts, continuing ID numbering from existing files."""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        results = []
        next_id = self.get_next_id(output_dir)  # Determine starting ID

        for i, prompt in enumerate(prompts):
            print(f"Generating image {i+1}/{len(prompts)}: {prompt}")
            seed = seeds[i] if seeds and i < len(seeds) else random.randint(0, 4294967295)
            
            result = self.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                width=width,
                height=height,
                steps=steps,
                sampler=sampler,
                model=model
            )
            
            if result:
                save_path = None
                if output_dir:
                    partial_prompt = prompt[:20]
                    partial_prompt_sanitized = re.sub(r'[^a-zA-Z0-9_-]+', '_', partial_prompt).strip("_")
                    
                    new_filename = f"generated_{next_id:04d}_{seed}_{partial_prompt_sanitized}.png"
                    save_path = os.path.join(output_dir, new_filename)
                    result.save(save_path)
                    logger.info(f"Saved generated image to {save_path}")
                    next_id += 1  # Increment ID after saving
                
                results.append({
                    "prompt": prompt,
                    "seed": seed,
                    "image": result,
                    "path": save_path
                })
            else:
                logger.warning(f"Failed to generate image for prompt: {prompt}")
            
            time.sleep(wait)
        
        return results
    
    def infer_one(self, text_prompt: str, **kwargs):
        """Infer for a single text prompt (overriding BaseImageInference method).
        
        This method accepts a text prompt instead of a PIL image since this is a text-to-image model.
        """
        return self.generate_image(prompt=text_prompt, **kwargs)
    
    def infer_batch(self, text_prompts: List[str], **kwargs):
        """Infer for a batch of text prompts (overriding BaseImageInference method).
        
        This method accepts a list of text prompts instead of PIL images since this is a text-to-image model.
        """
        return self.generate_multiple(prompts=text_prompts, **kwargs)
    
    def infer_many(self, prompt_list: List[str], **kwargs):
        """Infer for many text prompts using batched processing.
        
        Returns a pandas DataFrame with the results.
        """
        results = self.generate_multiple(prompts=prompt_list, **kwargs)
        
        # Convert to DataFrame format
        df_results = []
        for result in results:
            df_results.append({
                "prompt": result["prompt"],
                "seed": result["seed"],
                "output_path": result.get("path", "")
            })
            
        return pd.DataFrame(df_results)


# Demo usage
def demo_novelai_inference():
    # Initialize the model (get token from environment variable)
    inference = NovelAIInference()
    
    # Generate a single image
    prompt = "1girl, no text, best quality, very aesthetic, absurdres"
    negative_prompt = "blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality"
    
    # Set a specific seed for reproducibility
    seed = 2516170931
    
    # Generate the image
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)
    
    image = inference.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        save_path=os.path.join(output_dir, f"single_image_{seed}.png")
    )
    
    print(f"Generated image with seed: {seed}")
    
    # Generate multiple images with different prompts
    prompts = [
        "a beautiful landscape with mountains and a lake, vibrant colors",
        "a futuristic city with flying cars and tall skyscrapers",
        "a magical forest with glowing mushrooms and fairy lights"
    ]
    
    # Use fixed seeds for reproducibility
    seeds = [1234567, 7654321, 9876543]
    
    results = inference.generate_multiple(
        prompts=prompts,
        negative_prompt=negative_prompt,
        seeds=seeds,
        output_dir=output_dir
    )
    
    print(f"Generated {len(results)} images with multiple prompts")
    
    # Save results to DataFrame
    df = pd.DataFrame([{
        "prompt": r["prompt"],
        "seed": r["seed"],
        "path": r["path"]
    } for r in results])
    
    df.to_csv(os.path.join(output_dir, "generation_results.csv"), index=False)
    print("Results saved to generation_results.csv")
    
    return inference, results



V4_FULL_DEFAULT = {
    "prompt": "no text, best quality, very aesthetic, absurdres",
    "negative_prompt": "blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, multiple views, logo, too many watermarks",
}


DEFAULT_TEMPLATE = V4_FULL_DEFAULT

def apply_template(prompt:str, negative_prompt:str, template:dict | None = None):
    """
    append template to prompt and negative prompt
    """
    def _strip(s):
        return s.rstrip().rstrip(',').rstrip()

    def _strip_search(s):
        return s.replace('{', '').replace('}', '')

    if template is None:
        template = DEFAULT_TEMPLATE
    
    prompt_list = [_strip(prompt), template['prompt']]
    negative_prompt_list = [_strip(negative_prompt), template['negative_prompt']]

    # special case handling for default nai prompt negatives
    # only add the when input prompt does not contain the tags
    
    search_prompt = prompt.replace('_', ' ').replace('  ', ' ')
    add_to_neg_tags = ['{{black hair}}', '2girls']
    
    for tag in add_to_neg_tags:
        if _strip_search(tag) not in search_prompt:
            negative_prompt_list.append(tag)

    ret_prompt = ", ".join(i for i in prompt_list if i)
    ret_negative_prompt = ", ".join(i for i in negative_prompt_list if i)
    
    return ret_prompt, ret_negative_prompt


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_novelai_inference()