import torch
from transformers import ConvNextImageProcessor, ConvNextForImageClassification, BlipForConditionalGeneration, BlipProcessor
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the ConvNext model and image processor
convnext_processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-base-224")
convnext_model = ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224")

# Load BLIP-2 processor and model for caption generation
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Image preprocessing function
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Ensure the image is RGB
    inputs = convnext_processor(images=img, return_tensors="pt")  # Use processor to transform image
    return inputs

# Feature extraction function using ConvNext
def extract_features(image_path):
    inputs = preprocess_image(image_path)
    with torch.no_grad():
        outputs = convnext_model(**inputs)
        features = outputs.logits
    return features

# Caption generation function using BLIP-2
def generate_caption(image_path):
    # Load and process the image
    img = Image.open(image_path).convert("RGB")
    inputs = blip_processor(img, return_tensors="pt").to("cpu")  # Adjust to your device if necessary

    # Generate caption
    with torch.no_grad():
        caption_ids = blip_model.generate(**inputs)
        caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption

# Main function
if __name__ == "__main__":
    image_path = "static\gr2.jpeg"  # Replace with the path to your test image
    logging.info("Generating features and caption for the test image...")

    try:
        features = extract_features(image_path)
        logging.info("Features generated successfully.")
        print("Extracted Features:", features)
        
        # Generate the caption
        caption = generate_caption(image_path)
        logging.info("Caption generated successfully.")
        print("Generated Caption:", caption)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
