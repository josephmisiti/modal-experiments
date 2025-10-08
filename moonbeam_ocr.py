import modal
import sys
from pathlib import Path

app = modal.App("moonbeam-ocr")

# Create volumes for inputs and outputs
input_volume = modal.Volume.from_name("modal-experiments-inputs", create_if_missing=True)
output_volume = modal.Volume.from_name("modal-experiments-outputs", create_if_missing=True)

# Define the image with all required dependencies
MODEL_ID = "vikhyatk/moondream2"

def download_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Download and cache the model and tokenizer
    AutoTokenizer.from_pretrained(MODEL_ID)
    AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "pillow",
        "accelerate",
        "einops",
    )
    .run_function(download_model)
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={
        "/inputs": input_volume,
        "/outputs": output_volume,
    },
    timeout=600,
)
def ocr_with_bounding_boxes(image_path: str):
    import json
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to("cuda")

    # Load the image
    input_file = Path("/inputs") / Path(image_path).name
    if not input_file.exists():
        raise FileNotFoundError(f"Image not found: {input_file}")

    image = Image.open(input_file)

    # Encode the image
    enc_image = model.encode_image(image)

    # Perform OCR - detect text with bounding boxes
    ocr_prompt = "Extract all text from this image and provide the pixel coordinates (bounding boxes) for each text element. Return the results in a structured format with text content and coordinates."

    # Get OCR results
    ocr_result = model.answer_question(enc_image, ocr_prompt, tokenizer)

    # Also get objects with bounding boxes
    detect_prompt = "Detect all text regions in this image and provide their bounding box coordinates in pixels."
    detection_result = model.answer_question(enc_image, detect_prompt, tokenizer)

    # Try to get point-based detection for text
    text_objects_prompt = "What text can you see in this image? For each text element, describe its location."
    text_objects = model.answer_question(enc_image, text_objects_prompt, tokenizer)

    # Compile results
    results = {
        "image_path": image_path,
        "image_size": {
            "width": image.width,
            "height": image.height
        },
        "ocr_result": ocr_result,
        "detection_result": detection_result,
        "text_objects": text_objects,
        "detections": []
    }

    # Try to query specific bounding boxes using the model's point detection
    # Moondream supports pointing queries
    try:
        # Ask model to identify text regions
        query_result = model.query(enc_image, "text", tokenizer)
        if query_result:
            results["raw_detections"] = query_result
    except Exception as e:
        results["query_error"] = str(e)

    # Save output
    output_filename = Path(image_path).stem + "_ocr_results.json"
    output_path = Path("/outputs") / output_filename

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Commit the volume changes
    output_volume.commit()

    print(f"OCR results saved to: {output_path}")
    print(json.dumps(results, indent=2))

    return results


@app.local_entrypoint()
def main(image: str):
    """
    Run OCR on an image using Moondream3.

    Usage: modal run moonbeam_ocr.py --image <path-to-image>
    """
    from pathlib import Path
    import shutil

    # Copy image to input volume
    image_path = Path(image)
    if not image_path.exists():
        print(f"Error: Image not found at {image}")
        sys.exit(1)

    print(f"Processing image: {image}")
    print(f"Copying to input volume...")

    # Copy file to inputs directory (Modal will handle volume sync)
    input_dir = Path("/tmp/modal_inputs")
    input_dir.mkdir(exist_ok=True)
    dest_path = input_dir / image_path.name
    shutil.copy(image_path, dest_path)

    # Upload to volume (force=True to overwrite existing files)
    with input_volume.batch_upload(force=True) as batch:
        batch.put_file(str(dest_path), image_path.name)

    print(f"Running OCR with Moondream...")
    results = ocr_with_bounding_boxes.remote(image_path.name)

    print("\n" + "="*50)
    print("OCR COMPLETE")
    print("="*50)
    print(f"\nResults saved to outputs volume")

    return results
