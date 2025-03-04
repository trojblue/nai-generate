import argparse
import toml
from .app import create_ui


import os
import toml
from pathlib import Path



def main() -> None:
    """Main entry point for nai-generate CLI."""
    parser = argparse.ArgumentParser(description="Generate images using dynamic prompts")
    parser.add_argument("--share", action="store_true", help="Launch the Gradio UI with share=True")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration from file
    config_path = Path(args.config)
    if config_path.exists():
        try:
            config = toml.load(config_path)
            
            # Set environment variables from config
            if "novelai" in config:
                novelai_config = config["novelai"]
                if "endpoint" in novelai_config:
                    os.environ["NOVELAI_ENDPOINT"] = novelai_config["endpoint"]
                if "token" in novelai_config:
                    os.environ["NOVELAI_PST_TOKEN"] = novelai_config["token"]
            
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
    else:
        print(f"Config file {config_path} not found. Using environment variables.")
    
    # Create and launch the UI
    app = create_ui()
    
    if args.share:
        app.launch(share=True)    
    else:
        app.launch()

if __name__ == "__main__":
    main()
