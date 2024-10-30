import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration, ConvNextImageProcessor, ConvNextForImageClassification
from PIL import Image
import os

# Hyperparameters
batch_size = 4
learning_rate = 1e-5
num_epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define paths for the image and caption dataset (assuming you have a folder of images and a .txt file for captions)
image_folder = "path_to_your_image_folder"
caption_file = "path_to_your_caption_file"

# Load the BLIP and ConvNext models
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
convnext_processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-large-224")
convnext_model = ConvNextForImageClassification.from_pretrained("facebook/convnext-large-224").to(device)

# Dataset class to handle loading images and captions
class ImageCaptionDataset(Dataset):
    def __init__(self, image_folder, caption_file, convnext_processor, blip_processor):
        self.image_folder = image_folder
        self.convnext_processor = convnext_processor
        self.blip_processor = blip_processor

        # Load image paths and captions
        with open(caption_file, "r") as file:
            lines = file.readlines()
            self.image_caption_pairs = [
                (line.split("\t")[0], line.split("\t")[1].strip()) for line in lines
            ]

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        image_name, caption = self.image_caption_pairs[idx]
        image_path = os.path.join(self.image_folder, image_name)

        # Open and preprocess image
        image = Image.open(image_path).convert("RGB")
        convnext_inputs = self.convnext_processor(images=image, return_tensors="pt")
        blip_inputs = self.blip_processor(text=caption, return_tensors="pt")

        return convnext_inputs, blip_inputs["input_ids"].squeeze(), blip_inputs["attention_mask"].squeeze()

# Instantiate the dataset and dataloader
dataset = ImageCaptionDataset(image_folder, caption_file, convnext_processor, blip_processor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define optimizer and loss function for fine-tuning
optimizer = torch.optim.Adam(list(convnext_model.parameters()) + list(blip_model.parameters()), lr=learning_rate)

# Fine-tuning loop
for epoch in range(num_epochs):
    blip_model.train()
    convnext_model.train()
    
    total_loss = 0
    
    for convnext_inputs, input_ids, attention_mask in dataloader:
        # Move data to the appropriate device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Forward pass through ConvNext for feature extraction
        convnext_outputs = convnext_model(**convnext_inputs.to(device))
        features = convnext_outputs.logits
        
        # Forward pass through BLIP with extracted features and captions
        outputs = blip_model(input_ids=input_ids, attention_mask=attention_mask, encoder_outputs=features)
        
        # Calculate loss (using the model's built-in loss function)
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the fine-tuned models
blip_model.save_pretrained("fine_tuned_blip_model")
convnext_model.save_pretrained("fine_tuned_convnext_model")
