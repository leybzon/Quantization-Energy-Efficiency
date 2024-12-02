import torch
import csv
import time
import threading
from transformers import CLIPTokenizerFast
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown

# Initialize NVML
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)  # Assumes using GPU 0

# Model ID for Stable Diffusion 3.5
model_id = "stabilityai/stable-diffusion-3.5-large"

# Define quantization configurations
quantization_configs = [
    {"type": "NF4", "config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)},
    {"type": "FP4", "config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.bfloat16)},
    {"type": "8-bit", "config": BitsAndBytesConfig(load_in_8bit=True)},
    {"type": "Mixed Precision", "config": None},  # No quantization, use torch.float16
]

# Define the prompts (rephrased and condensed to less than 77 tokens)
prompts = [
    {
        "prompt": "A creative image of a waffle-hippopotamus hybrid basking in melted butter, with pancake-like foliage in the background.",
        "negative_prompt": "blurry, low-quality, unrealistic, oversaturated, distorted proportions"
    },
    {
        "prompt": "An enchanting modern oil painting of a transparent glass unicorn with iridescent mane, nibbling vibrant flowers in a magical meadow at sunset.",
        "negative_prompt": "blurry, low-quality, unrealistic, oversaturated, distorted proportions"
    },
    {
        "prompt": "A girl with an umbrella and sky-blue eyes looking directly at the viewer.",
        "negative_prompt": "blurry, low-quality, unrealistic, oversaturated, distorted proportions"
    },
    {
        "prompt": "A dreamy forest pattern in doodle style featuring a majestic dragon with intricate, swirling designs and vibrant colors.",
        "negative_prompt": "blurry, low-quality, unrealistic, oversaturated, distorted proportions"
    },
    {
        "prompt": "An abstract artwork titled 'brake eggs' exploring vibrant hues and textured strokes, evoking sensory experience and synesthesia.",
        "negative_prompt": "blurry, low-quality, unrealistic, oversaturated, distorted proportions"
    },
    {
        "prompt": "A romantic scene of a fallen angel and a saint on a date at a trendy rooftop café, blending urban nightlife with celestial elements.",
        "negative_prompt": "blurry, low-quality, unrealistic, oversaturated, distorted proportions"
    },
    {
        "prompt": "An extremely minimalist, symmetrical, almost transparent flower with simple lines on a pale palette, focusing on elegance and balance.",
        "negative_prompt": "blurry, low-quality, unrealistic, oversaturated, distorted proportions"
    },
    {
        "prompt": "Little Red Riding Hood converses with a detailed robotic wolf in an enchanted forest, surrounded by lifelike fuzzy bunnies hopping around.",
        "negative_prompt": "blurry, low-quality, unrealistic, oversaturated, distorted proportions"
    },
]

# CSV files
results_csv = "quantization_results_with_energy.csv"
power_csv = "intermediate_power_readings.csv"

# Use the fast tokenizer to improve tokenization performance
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")

# Suppress warnings for expected behavior
import warnings
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast")

# Function to measure power during inference using threading
def measure_power_during_inference(handle, inference_fn):
    power_readings = []
    sample_interval = 0.01  # 10ms
    stop_event = threading.Event()

    # Function to measure power readings
    def power_measurement_thread():
        while not stop_event.is_set():
            power_readings.append(nvmlDeviceGetPowerUsage(handle) / 1000)  # Power in watts
            time.sleep(sample_interval)

    # Start power measurement thread
    power_thread = threading.Thread(target=power_measurement_thread)
    power_thread.start()

    # Synchronize before starting inference to ensure accurate timing
    torch.cuda.synchronize()
    start_time = time.time()

    # Perform inference
    inference_fn()

    # Synchronize after inference to ensure all GPU operations are completed
    torch.cuda.synchronize()
    end_time = time.time()

    # Signal the thread to stop and wait for it to finish
    stop_event.set()
    power_thread.join()

    # Calculate metrics
    avg_power = sum(power_readings) / len(power_readings) if power_readings else 0
    total_energy = sum(power_readings) * sample_interval  # Energy in Joules
    inference_time = end_time - start_time

    return avg_power, total_energy, power_readings, inference_time

# Function to run tests
def run_test(quant_type, quant_config):
    print(f"Running test for Quantization Type: {quant_type}")

    # Configure model and pipeline
    if quant_config:
        model = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quant_config["config"],
            torch_dtype=torch.bfloat16
        )
    else:
        model = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.float16
        )

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=model,
        torch_dtype=torch.bfloat16 if quant_config else torch.float16
    )

    # Ensure quantized models stay on GPU (don't call `.to("cpu")`)
    pipeline.enable_model_cpu_offload()

    # Use the fast tokenizer
    pipeline.tokenizer = tokenizer

    results = []

    for prompt_index, prompt_data in enumerate(prompts):
        prompt = prompt_data["prompt"]
        negative_prompt = prompt_data["negative_prompt"]

        print(f"  Running prompt {prompt_index + 1}/{len(prompts)}")

        # Lists to store metrics over multiple runs
        memory_usages = []
        inference_times = []
        avg_powers = []
        total_energies = []

        for run in range(1, 11):  # Run each test 10 times
            print(f"    Run {run}/10")
            # Clear GPU memory
            torch.cuda.empty_cache()

            # Reset peak memory tracker
            torch.cuda.reset_peak_memory_stats()

            # Define the inference function
            def inference_fn():
                image = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=28,
                    guidance_scale=4.5
                ).images[0]
                image.save(f"test-{quant_type}-prompt{prompt_index+1}-run{run}.png")

            # Measure power, energy, and time
            avg_power, total_energy, power_readings, inference_time = measure_power_during_inference(gpu_handle, inference_fn)

            # Measure peak GPU memory usage
            peak_memory = torch.cuda.max_memory_allocated() // (1024 * 1024)  # Convert bytes to MB

            # Save intermediate power readings for debugging
            with open(power_csv, mode="a", newline="") as power_file:
                writer = csv.writer(power_file)
                for reading in power_readings:
                    writer.writerow([quant_type, f"Prompt {prompt_index+1}", f"Run {run}", reading])

            # Collect metrics
            memory_usages.append(peak_memory)
            inference_times.append(inference_time)
            avg_powers.append(avg_power)
            total_energies.append(total_energy)

        # Compute averages over the 10 runs
        avg_memory_usage = sum(memory_usages) / len(memory_usages)
        avg_inference_time = sum(inference_times) / len(inference_times)
        avg_avg_power = sum(avg_powers) / len(avg_powers)
        avg_total_energy = sum(total_energies) / len(total_energies)

        results.append({
            "Quantization Type": quant_type,
            "Prompt Index": prompt_index + 1,
            "Memory Usage (MB)": round(avg_memory_usage, 2),
            "Time (s)": round(avg_inference_time, 2),
            "Power Usage (Watts)": round(avg_avg_power, 2),
            "Energy Consumption (J)": round(avg_total_energy, 2),
            "Image Names": [f"test-{quant_type}-prompt{prompt_index+1}-run{run}.png" for run in range(1, 11)]
        })

    return results

# Main script execution
if __name__ == "__main__":
    # Write headers for debugging power CSV
    with open(power_csv, mode="w", newline="") as power_file:
        writer = csv.writer(power_file)
        writer.writerow(["Quantization Type", "Prompt", "Run", "Power Reading (W)"])

    all_results = []

    for config in quantization_configs:
        try:
            test_results = run_test(config["type"], config if config["config"] else None)
            all_results.extend(test_results)
        except Exception as e:
            print(f"Error with Quantization Type {config['type']}: {e}")

    # Write results to CSV
    fieldnames = ["Quantization Type", "Prompt Index", "Memory Usage (MB)", "Time (s)", "Power Usage (Watts)", "Energy Consumption (J)", "Image Names"]

    with open(results_csv, mode="w", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            # Flatten the Image Names list into a single string
            result["Image Names"] = ', '.join(result["Image Names"])
            writer.writerow(result)

    # Shut down NVML
    nvmlShutdown()

    print(f"Results saved to {results_csv}")
    print(f"Intermediate power readings saved to {power_csv}")
