import gradio as gr
import os
import time
import random
import re
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

from dynamicprompts.generators import RandomPromptGenerator
from .novelai_wrapper import NovelAIInference, DEFAULT_TEMPLATE, apply_template

def generate_with_novelai(
    prompts: List[str],
    api_token: str,
    model: str,
    save_dir: str,
    negative_prompt: str = None,
    width: int = 832,
    height: int = 1216,
    steps: int = 28,
    sampler: str = "k_euler_ancestral",
    seeds: List[int] = None
) -> List[Dict[str, Any]]:
    """
    Generate images using NovelAI with the provided prompts
    
    Args:
        prompts: List of text prompts to generate images from
        api_token: NovelAI API token
        model: Model to use for generation
        save_dir: Directory to save images to
        negative_prompt: Negative prompt to use for all images
        width: Image width
        height: Image height
        steps: Number of diffusion steps
        sampler: Sampler to use
        seeds: Optional list of seeds (one per prompt)
        
    Returns:
        List of dictionaries containing prompt, seed, image path and PIL image
    """
    # Set up the NovelAI inference
    inference = NovelAIInference(
        persistent_token=api_token,
        default_model=model,
        default_width=width,
        default_height=height,
        default_steps=steps,
        default_sampler=sampler
    )
    
    # Apply default templates to each prompt if negative_prompt is None
    if negative_prompt is None:
        processed_prompts = []
        for prompt in prompts:
            enhanced_prompt, default_neg = apply_template(prompt, "", DEFAULT_TEMPLATE)
            processed_prompts.append(enhanced_prompt)
        prompts = processed_prompts
        negative_prompt = default_neg
    
    # Generate images with NovelAI
    results = inference.generate_multiple(
        prompts=prompts,
        negative_prompt=negative_prompt,
        seeds=seeds,
        output_dir=save_dir,
        wait=1.0  # Wait 1 second between requests
    )
    
    return results

def generate_images(
    api_token,
    api_url,
    model,
    artist_list,
    prompt_template,
    save_location,
    num_images,
    negative_prompt=None,
    width=832,
    height=1216,
    steps=28,
    sampler="k_euler_ancestral"
):
    """
    Generate images using dynamic prompts with sequential artist switching
    and random general/elements.
    
    Returns a tuple of:
      - list of image paths (for Gradio gallery)
      - string of generated prompts (for Textbox)
      - string of logs (for the logs output)
    """
    logs = []
    gallery_images = []
    generated_prompts_text = []

    # Use token from environment if available and not provided in UI
    if not api_token and os.environ.get("NOVELAI_PST_TOKEN"):
        api_token = os.environ.get("NOVELAI_PST_TOKEN")
    
    # If still no token, we can’t proceed
    if not api_token:
        error_msg = "Error: NovelAI API token not provided."
        print(error_msg)
        logs.append(error_msg)
        return [], "", "\n".join(logs)
    
    # Create save directory if it doesn't exist
    save_path = Path(save_location)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Parse artist list
    artists = [artist.strip() for artist in artist_list.strip().split('\n') if artist.strip()]
    if not artists:
        error_msg = "No artists provided."
        print(error_msg)
        logs.append(error_msg)
        return [], "", "\n".join(logs)
    
    # Create the dynamic prompt with sequential artists and random general/elements
    # Use the @ prefix for cyclical sampling of artists
    artist_variant = "{@" + "|".join(artists) + "}"
    prompt_with_artists = prompt_template.replace("{artist}", artist_variant)
    
    # Use RandomPromptGenerator which will handle the mixed sampling methods
    generator = RandomPromptGenerator()
    
    # Generate the list of prompts
    generated_prompts = generator.generate(prompt_with_artists, num_images)
    print(f"Created {len(generated_prompts)} prompts to generate")
    logs.append(f"Created {len(generated_prompts)} prompts to generate")

    # Generate images with NovelAI
    try:
        # Generate random seeds for reproducibility
        seeds = [random.randint(0, 4294967295) for _ in range(num_images)]
        
        results = generate_with_novelai(
            prompts=generated_prompts,
            api_token=api_token,
            model=model,
            save_dir=str(save_path),
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            sampler=sampler,
            seeds=seeds
        )
        
        # Prepare output data
        for i, result in enumerate(results):
            prompt = result["prompt"]
            seed = result["seed"]
            path_to_image = result["path"]
            
            generated_prompts_text.append(f"Image {i+1}: {prompt} (seed: {seed})")
            gallery_images.append(path_to_image)
        
        print("Image generation successful.")
        logs.append("Image generation successful.")

    except Exception as e:
        error_msg = f"Error generating images: {str(e)}"
        print(error_msg)
        logs.append(error_msg)
        return [], "", "\n".join(logs)

    # Return gallery, text, and logs
    return gallery_images, "\n".join(generated_prompts_text), "\n".join(logs)

def create_ui():
    """Create Gradio UI for the image generator."""
    # Get default save location from environment
    default_save_dir = os.environ.get("NAI_GENERATE_SAVE_DIR", "./generated_images")
    
    # Check if token is already in environment
    has_token_in_env = bool(os.environ.get("NOVELAI_PST_TOKEN"))
    
    with gr.Blocks(title="Dynamic Prompts Image Generator") as app:
        gr.Markdown("# Dynamic Prompts Image Generator with NovelAI")
        
        if not has_token_in_env:
            gr.Markdown("""
            This application uses the dynamicprompts library to generate images with sequential artist cycling 
            and random general/elements using NovelAI.
            
            **Note:** Please provide your NovelAI API token below or set it in config.toml.
            """)
        else:
            gr.Markdown("""
            This application uses the dynamicprompts library to generate images with sequential artist cycling 
            and random general/elements using NovelAI.
            
            **Note:** NovelAI API token is loaded from configuration.
            """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Always create the API token input, but hide it if token is in env
                api_token = gr.Textbox(
                    label="NovelAI API Token (Optional if configured)", 
                    placeholder="pst-...",
                    type="password",
                    info="Your NovelAI API token (starts with pst-)",
                    visible=not has_token_in_env,
                    value=""  # Empty string
                )
                
                api_url = gr.Textbox(
                    label="API URL", 
                    placeholder="https://api.novelai.net",
                    value="https://api.novelai.net",
                    visible=False  # Hidden since we’re using NovelAI’s built-in URL
                )
                
                model = gr.Dropdown(
                    label="Model", 
                    choices=["nai-diffusion-4-full", "nai-diffusion-3", "nai-diffusion-2", "nai-diffusion"],
                    value="nai-diffusion-4-full",
                    info="The NovelAI model to use for image generation"
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt", 
                        value="blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, multiple views, logo, too many watermarks, white blank page, blank page, blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, multiple views, logo, too many watermarks, white blank page, blank page",
                        placeholder="Low quality, bad anatomy, worst quality, low quality",
                        info="Leave empty to use default NovelAI negative prompt"
                    )
                    
                    with gr.Row():
                        width = gr.Slider(
                            label="Width", 
                            minimum=512, 
                            maximum=1024, 
                            value=832, 
                            step=64,
                            info="Image width"
                        )
                        height = gr.Slider(
                            label="Height", 
                            minimum=512, 
                            maximum=1024, 
                            value=1216, 
                            step=64,
                            info="Image height"
                        )
                    
                    with gr.Row():
                        steps = gr.Slider(
                            label="Steps", 
                            minimum=20, 
                            maximum=50, 
                            value=28, 
                            step=1,
                            info="Diffusion steps"
                        )
                        sampler = gr.Dropdown(
                            label="Sampler", 
                            choices=["k_euler", "k_euler_ancestral", "k_dpmpp_2s_ancestral", "k_dpmpp_2m"],
                            value="k_euler_ancestral",
                            info="Sampling method"
                        )
                prompt_template = gr.Textbox(
                    label="Prompt Template", 
                    placeholder="{artist}, {general}, {elements}",
                    value="{artist}, {landscape|portrait|still life}, {vibrant colors|muted tones|black and white}, 1girl, long hair, eating, no text, best quality, very aesthetic, absurdres",
                    lines=4,
                    info="Use {artist} for artist names, and variants like {option1|option2} for random selection"
                )

                artist_list = gr.Textbox(
                    label="Artist List (one per line)", 
                    value="sy4\nnaga_u\nwlop\nsyuri22\nsoresaki\npumpkinspicelatte",
                    placeholder="sy4\nnaga_u\nwlop\nsyuri22\nsoresaki\npumpkinspicelatte",
                    lines=8,
                    info="Enter artist names, one per line. These will be cycled through sequentially."
                )

                save_location = gr.Textbox(
                    label="Save Location", 
                    placeholder="./generated_images",
                    value=default_save_dir,
                    info="Directory where images will be saved"
                )
                
                num_images = gr.Number(
                    label="Total number of Images to Generate", 
                    minimum=1, 
                    maximum=100000, 
                    value=6, 
                    step=1,
                    info="How many images to generate"
                )
                
                generate_button = gr.Button("Generate Images", variant="primary")
            
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    label="Generated Images",
                    columns=2,
                    rows=3,
                    object_fit="contain",
                    height="600px"
                )
                prompt_output = gr.Textbox(
                    label="Generated Prompts", 
                    lines=10,
                    info="The prompts used to generate each image"
                )

        # LOGS accordion
        with gr.Accordion("📂 Logs (Click to Expand)", open=False):
            log_output = gr.Textbox(
                label="Logs",
                lines=8,
                interactive=False
            )

        # When we click the generate button, we get three outputs:
        # - gallery (list of images)
        # - prompt_output (generated prompts text)
        # - log_output (logs)
        inputs = [
            api_token, 
            api_url, 
            model, 
            artist_list, 
            prompt_template, 
            save_location, 
            num_images,
            negative_prompt, 
            width, 
            height, 
            steps, 
            sampler
        ]
        outputs = [gallery, prompt_output, log_output]

        generate_button.click(
            fn=generate_images,
            inputs=inputs,
            outputs=outputs
        )
    
    return app

# Main function
if __name__ == "__main__":
    app = create_ui()
    app.launch()
