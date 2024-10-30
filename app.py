import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, ConvNextImageProcessor, ConvNextForImageClassification
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the models and processors
convnext_processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-large-224")
convnext_model = ConvNextForImageClassification.from_pretrained("facebook/convnext-large-224")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Move models to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
convnext_model.to(device)
blip_model.to(device)

# Streamlit interface setup
st.set_page_config(page_title="Image Caption Generator", page_icon=":camera:", layout="centered")

# Style for the app
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3, h4, p {
            color: black;  /* Change all text to black */
        }
        h1 {
            font-weight: 700;
        }
        .button {
            font-size: 18px;
            background-color: #f63366;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
        }
        .button:hover {
            background-color: #e0245e;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app interface
st.title("Image Caption Generator")
st.write("")

# Image upload
uploaded_image = st.file_uploader("Choose an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

# Define functions for preprocessing, feature extraction, and caption generation
def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    inputs = convnext_processor(images=img, return_tensors="pt").to(device)
    return inputs

def extract_features(image):
    inputs = preprocess_image(image)
    with torch.no_grad():
        outputs = convnext_model(**inputs)
        features = outputs.logits
    return features

def generate_caption(image, max_length=50, num_beams=5):
    image = Image.open(image).convert("RGB")
    blip_inputs = blip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        caption_ids = blip_model.generate(
            **blip_inputs, 
            max_new_tokens=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        caption = blip_processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
    return caption

# Generate caption button
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write("Click below to generate a caption for your image.")

    if st.button("ENTER", key="generate_button"):
        try:
            with st.spinner("Extracting features..."):
                features = extract_features(uploaded_image)
            
            with st.spinner("Generating caption..."):
                caption = generate_caption(uploaded_image)
                
                # Convert the caption to uppercase
                formatted_caption = caption.upper()
                
                # Display the caption in bold with a larger font size and in black color
                st.markdown(f"<h2 style='font-weight:bold; font-size:32px; color:black;'>{formatted_caption}</h2>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
