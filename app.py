import gradio as gr
import os
import requests
import json
from PIL import Image
import io
import time
import base64
from pathlib import Path
from dynamicprompts.generators import RandomPromptGenerator

def generate_images(api_url, api_key, model, artist_list, prompt_template, save_location, num_images):
    """
    Generate images using dynamic prompts with sequential artist switching
    and random general/elements.
    """
    # Create save directory if it doesn't exist
    save_path = Path(save_location)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Parse artist list
    artists = [artist.strip() for artist in artist_list.strip().split('\n') if artist.strip()]
    if not artists:
        return [], "No artists provided"
    
    # Create the dynamic prompt with sequential artists and random general/elements
    # Use the @ prefix for cyclical sampling of artists
    artist_variant = "{@" + "|".join(artists) + "}"
    prompt_with_artists = prompt_template.replace("{artist}", artist_variant)
    
    # Use RandomPromptGenerator which will handle the mixed sampling methods
    generator = RandomPromptGenerator()
    
    # Generate prompts
    generated_prompts = generator.generate(prompt_with_artists, num_images)
    
    gallery_images = []
    generated_prompts_text = []
    
    for i, prompt in enumerate(generated_prompts):
        generated_prompts_text.append(f"Image {i+1}: {prompt}")
        
        try:
            # Call API to generate image
            payload = {
                "prompt": prompt,
                "model": model,
                # Add other API parameters as needed
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Process response based on content type
            content_type = response.headers.get('Content-Type', '')
            
            if 'application/json' in content_type:
                # API returned JSON
                json_data = response.json()
                
                # Try to extract image data or URL from JSON (adjust based on API)
                if 'data' in json_data and isinstance(json_data['data'], list) and 'url' in json_data['data'][0]:
                    # If API returns URL to image
                    image_url = json_data['data'][0]['url']
                    image_response = requests.get(image_url)
                    image = Image.open(io.BytesIO(image_response.content))
                elif 'data' in json_data and isinstance(json_data['data'], list) and 'b64_json' in json_data['data'][0]:
                    # If API returns base64 encoded image
                    image_data = base64.b64decode(json_data['data'][0]['b64_json'])
                    image = Image.open(io.BytesIO(image_data))
                else:
                    # API structure varies - try to handle common formats
                    image_content = None
                    if 'image' in json_data:
                        # Direct image data
                        image_content = base64.b64decode(json_data['image'])
                    elif 'images' in json_data and len(json_data['images']) > 0:
                        image_content = base64.b64decode(json_data['images'][0])
                    
                    if image_content:
                        image = Image.open(io.BytesIO(image_content))
                    else:
                        raise ValueError(f"Couldn't extract image from JSON response: {json_data}")
            else:
                # Assume API returned image directly
                image = Image.open(io.BytesIO(response.content))
            
            # Save the image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = save_path / f"image_{timestamp}_{i}.png"
            
            image.save(str(filename))
            gallery_images.append(str(filename))
            
        except Exception as e:
            error_msg = f"Error generating image {i+1}: {str(e)}"
            generated_prompts_text.append(error_msg)
            print(error_msg)
    
    return gallery_images, "\n".join(generated_prompts_text)

def create_ui():
    """Create Gradio UI for the image generator."""
    with gr.Blocks(title="Dynamic Prompts Image Generator") as app:
        gr.Markdown("# Dynamic Prompts Image Generator")
        gr.Markdown("""
        This application uses the dynamicprompts library to generate images with sequential artist cycling 
        and random general/elements. 
        
        **Instructions:**
        1. Enter your API credentials and model
        2. Paste a list of artists (one per line)
        3. Enter a prompt template using {artist}, {general}, and {elements}
        4. Click Generate Images
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                api_url = gr.Textbox(
                    label="API URL", 
                    placeholder="https://api.example.com/v1/images/generations",
                    info="The URL endpoint for your image generation API"
                )
                api_key = gr.Textbox(
                    label="API Key", 
                    type="password",
                    info="Your API key for authentication"
                )
                model = gr.Textbox(
                    label="Model", 
                    placeholder="stable-diffusion-xl-1.0",
                    info="The model to use for image generation"
                )
                
                artist_list = gr.Textbox(
                    label="Artist List (one per line)", 
                    placeholder="van Gogh\nPicasso\nDa Vinci\nMonet",
                    lines=8,
                    info="Enter artist names, one per line. These will be cycled through sequentially."
                )
                
                prompt_template = gr.Textbox(
                    label="Prompt Template", 
                    placeholder="{artist}, {general}, {elements}",
                    value="A painting by {artist}, {landscape|portrait|still life}, {vibrant colors|muted tones|black and white}",
                    lines=4,
                    info="Use {artist} for artist names, and variants like {option1|option2} for random selection"
                )
                
                save_location = gr.Textbox(
                    label="Save Location", 
                    placeholder="./generated_images",
                    value="./generated_images",
                    info="Directory where images will be saved"
                )
                
                num_images = gr.Slider(
                    label="Number of Images to Generate", 
                    minimum=1, 
                    maximum=20, 
                    value=5, 
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
        
        generate_button.click(
            fn=generate_images,
            inputs=[api_url, api_key, model, artist_list, prompt_template, save_location, num_images],
            outputs=[gallery, prompt_output]
        )
    
    return app

# Main function
if __name__ == "__main__":
    app = create_ui()
    app.launch()