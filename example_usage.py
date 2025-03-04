"""
This is an example showing how to use dynamicprompts for sequential artist switching
with random general and elements without the Gradio UI.
"""

from dynamicprompts.generators import RandomPromptGenerator

def demonstrate_dynamic_prompts():
    # List of artists
    artists = [
        "Vincent van Gogh",
        "Claude Monet", 
        "Salvador Dali",
        "Pablo Picasso",
        "Leonardo da Vinci"
    ]
    
    # Create the prompt template with sequential artists and random general/elements
    # Use the @ prefix for cyclical (sequential) sampling of artists
    artist_variant = "{@" + "|".join(artists) + "}"
    
    # Example 1: Basic template
    template = f"A painting by {artist_variant}, {{landscape|portrait|still life}}, {{vibrant colors|muted tones|black and white}}"
    
    # Use RandomPromptGenerator which will handle the mixed sampling methods
    generator = RandomPromptGenerator()
    
    # Generate 10 prompts
    prompts = generator.generate(template, 10)
    
    print("Example 1: Basic template with sequential artists and random general/elements")
    for i, prompt in enumerate(prompts):
        print(f"{i+1}. {prompt}")
    print()
    
    # Example 2: More complex template
    template = f"Artwork in the style of {artist_variant}, {{fantasy scene|urban landscape|nature|abstract}}, {{oil painting|watercolor|digital art}}, {{colorful|monochromatic|high contrast}}"
    
    # Generate 10 prompts
    prompts = generator.generate(template, 10)
    
    print("Example 2: More complex template")
    for i, prompt in enumerate(prompts):
        print(f"{i+1}. {prompt}")

if __name__ == "__main__":
    demonstrate_dynamic_prompts()