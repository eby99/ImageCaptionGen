import streamlit as st
from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the models and processors
convnext_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
convnext_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Move models to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
convnext_model.to(device)

# Create a Flask API
app = Flask(__name__)

# Define API endpoints
@app.route('/caption', methods=['POST'])
def generate_caption_api():
    image = request.files['image']
    caption = generate_caption(image)
    return jsonify({'caption': caption})

def generate_caption(image):
    img = Image.open(image).convert("RGB")
    inputs = convnext_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = convnext_model.generate(**inputs)
        caption = convnext_processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return caption

# Streamlit interface setup
st.set_page_config(page_title="Image Caption Generator", page_icon=":camera:", layout="centered")

# Style for the app
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6; /* Light gray background */
        }
        .main {
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.2);
            background-color: white; /* White card background */
            max-width: 800px; /* Limit width */
            margin: auto; /* Center the card */
            overflow-y: auto; /* Enable scrolling */
            max-height: 80vh; /* Limit height */
        }
        h1 {
            color: red; /* Title color */
            font-weight: 700;
            font-size: 36px; /* Increase font size */
            text-align: center; /* Center align title */
        }
        h2 {
            font-weight: bold;
            font-size: 32px;
            color: #0033cc; /* Dark blue for caption */
            text-align: center; /* Center align caption */
        }
        .button {
            font-size: 18px;
            background-color: #f63366; /* Button background color */
            color: white; /* Button text color */
            padding: 12px 24px; /* Padding for the button */
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease; /* Smooth transition */
            display: block; /* Make button a block element */
            margin: 20px auto; /* Center align button */
        }
        .button:hover {
            background-color: #e0245e; /* Button hover color */
        }
        .file-uploader {
            margin: 20px 0; /* Add margin to file uploader */
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app interface
st.title("Image Caption Generator")
st.write("")

# Scrollable container for main content
with st.container():
    # Image upload
    uploaded_image = st.file_uploader("Choose an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    # Define functions for preprocessing, feature extraction, and caption generation
    def preprocess_image(image):
        img = Image.open(image).convert("RGB")
        inputs = convnext_processor(images=img, return_tensors="pt").to(device)
        return inputs

    def generate_caption(image):
        inputs = preprocess_image(image)
        with torch.no_grad():
            outputs = convnext_model.generate(**inputs)
            caption = convnext_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return caption

    # Generate caption button
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write("Click below to generate a caption for your image.")

        if st.button("GENERATE CAPTION", key="generate_button", help="Generate a caption for the uploaded image"):
            try:
                with st.spinner("Generating caption..."):  # No text_color argument
                    caption = generate_caption(uploaded_image)
                    
                    # Convert the caption to uppercase
                    formatted_caption = caption.upper()
                    
                    # Display the caption in darker blue
                    st.markdown(f"<h2 style='font-weight:bold; font-size:32px; color:#0033cc;'>{formatted_caption}</h2>", unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)
