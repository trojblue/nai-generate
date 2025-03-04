# Dynamic Prompts Image Generator

This application integrates the [dynamicprompts](https://github.com/adieyal/dynamicprompts) library with a Gradio UI to generate images with sequential artist cycling and random general/elements.

## Features

- Sequential artist switching (cyclical sampler)
- Random general and elements selection
- Custom save location for generated images
- Easy-to-use Gradio interface

## How It Works

The application uses the dynamicprompts library to handle prompt generation with different sampling methods:

- **Sequential Artist Switching**: Uses the cyclical sampler (`@`) to cycle through artists one by one
- **Random General and Elements**: Uses the default random sampler to randomly select from general and element options

For example, when you provide a template like:
```
A painting by {artist}, {landscape|portrait|still life}, {vibrant colors|muted tones|black and white}
```

The application will:
1. Replace `{artist}` with `{@artist1|artist2|artist3}` (using the cyclical sampler)
2. Leave the other variant groups with random sampling
3. Generate images with sequential artists and random general/elements

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/trojblue/nai-generate
   cd nai-generate
   ```

2. Install the project:
   ```
   pip install -e .
   ```

## Usage

1. Create a configuration file (e.g., `config.toml`) with the following format:

   - see: [config.toml.example](config.toml.example) for an example configuration file

   ```toml
   [novelai]
   endpoint = "https://image.novelai.net/ai/generate-image"
   token = "pst-XXXX"

   [app]
   save_dir = "./generated_images"
   ```

2. Run the application (in the terminal):
   ```bashw
   # reads config in current directory
   nai-generate 

   # full config path + gradio share
   nai-generate --config config.toml --share
   ```

3. Access the UI in your browser (typically at http://127.0.0.1:7860)

4. Click "Generate Images" to start the generation process

## Example

**Artist List:**
```
Vincent van Gogh
Claude Monet
Salvador Dali
```

**Prompt Template:**
```
A painting by {artist}, {landscape|portrait|still life}, {vibrant colors|muted tones|black and white}
```

This will generate sequential images with:
1. "A painting by Vincent van Gogh, [random selection], [random selection]"
2. "A painting by Claude Monet, [random selection], [random selection]"  
3. "A painting by Salvador Dali, [random selection], [random selection]"
4. "A painting by Vincent van Gogh, [random selection], [random selection]"
5. And so on...

## Notes


To build wheels manually, run the following commands:

```bash
python -m pip install build twine
python -m build
twine check dist/*
twine upload dist/*
```