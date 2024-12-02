# Quantization Energy Efficiency

This repository provides a script and resources for analyzing the energy efficiency of different quantization methods applied to the [Stable Diffusion 3.5 model](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) during image generation. It aims to support articles and discussions around energy-efficient AI, particularly in the context of generative models.

## Overview

The script evaluates the effects of quantization on:
- **Inference Performance**: Time required for image generation.
- **GPU Memory Usage**: Peak memory allocated during inference.
- **Energy Efficiency**: Power consumption and energy usage during inference.

The repository uses creative prompts to measure the impact of quantization on the quality and energy efficiency of image generation.

## Quantization Methods

The following quantization methods are evaluated in the script:

| Quantization Type  | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| **NF4**            | 4-bit quantization using the `nf4` type with computation in `torch.bfloat16`. |
| **FP4**            | 4-bit quantization using the `fp4` type with computation in `torch.bfloat16`. |
| **8-bit**          | 8-bit quantization for efficient inference with minimal quality degradation. |
| **Mixed Precision**| Default mode using `torch.float16` without any explicit quantization.       |

## Prompts

The script tests multiple creative prompts to ensure a comprehensive analysis. Each prompt is paired with a negative prompt to avoid undesirable characteristics like low quality or unrealistic imagery.

| Prompt | Negative Prompt |
|--------|------------------|
| **Prompt 1**: A creative image of a waffle-hippopotamus hybrid basking in melted butter, with pancake-like foliage in the background. | blurry, low-quality, unrealistic, oversaturated, distorted proportions |
| **Prompt 2**: An enchanting modern oil painting of a transparent glass unicorn with iridescent mane, nibbling vibrant flowers in a magical meadow at sunset. | blurry, low-quality, unrealistic, oversaturated, distorted proportions |
| **Prompt 3**: A girl with an umbrella and sky-blue eyes looking directly at the viewer. | blurry, low-quality, unrealistic, oversaturated, distorted proportions |
| **Prompt 4**: A dreamy forest pattern in doodle style featuring a majestic dragon with intricate, swirling designs and vibrant colors. | blurry, low-quality, unrealistic, oversaturated, distorted proportions |
| **Prompt 5**: An abstract artwork titled 'brake eggs' exploring vibrant hues and textured strokes, evoking sensory experience and synesthesia. | blurry, low-quality, unrealistic, oversaturated, distorted proportions |
| **Prompt 6**: A romantic scene of a fallen angel and a saint on a date at a trendy rooftop café, blending urban nightlife with celestial elements. | blurry, low-quality, unrealistic, oversaturated, distorted proportions |
| **Prompt 7**: An extremely minimalist, symmetrical, almost transparent flower with simple lines on a pale palette, focusing on elegance and balance. | blurry, low-quality, unrealistic, oversaturated, distorted proportions |
| **Prompt 8**: Little Red Riding Hood converses with a detailed robotic wolf in an enchanted forest, surrounded by lifelike fuzzy bunnies hopping around. | blurry, low-quality, unrealistic, oversaturated, distorted proportions |

## Results

Generated results are saved in the `results` directory:

### Files

1. **[intermediate_power_readings.csv](results/intermediate_power_readings.csv)**:
   - Contains detailed power readings during inference, recorded every 10 milliseconds.
   - Columns include:
     - Quantization type
     - Prompt index
     - Run number
     - Power readings (in Watts)

2. **[quantization_results_with_energy.csv](results/quantization_results_with_energy.csv)**:
   - Summarized metrics for each quantization type and prompt.
   - Columns include:
     - Quantization type
     - Prompt index
     - Average memory usage (in MB)
     - Average inference time (in seconds)
     - Average power consumption (in Watts)
     - Total energy consumption (in Joules)
     - List of generated image filenames.

## Model

This project uses the [Stable Diffusion 3.5 large model](https://huggingface.co/stabilityai/stable-diffusion-3.5-large), a powerful generative model capable of producing high-quality images from textual descriptions.

## How to Use

1. Clone the repository:
   git clone https://github.com/leybzon/Quantization-Energy-Efficiency.git
   cd Quantization-Energy-Efficiency
2. Install dependencies:
   pip install torch transformers diffusers pynvml
3. Run the script:
   python stable-diffusion-3.5-large-test-with-power.py
4. Check outputs:
   Generated images
   CSV files with detailed and summarized results

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to enhance the script, add prompts, or suggest new quantization methods.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
