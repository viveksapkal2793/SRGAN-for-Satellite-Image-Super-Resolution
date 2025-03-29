import io
import streamlit as st
import torch
import time
import numpy as np
import os
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, CenterCrop, Resize
from model import Generator
from math import log10

# Set page configuration
st.set_page_config(
    page_title="Satellite Image Super-Resolution", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Calculate valid crop size (from utils.py)
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

# Function to calculate PSNR
def calculate_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return 100
    return 10 * log10(1.0 / mse)

# Load model with caching
@st.cache_resource
def load_model(model_path):
    model = Generator().eval()
    if torch.cuda.is_available():
        model.cuda()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, 
                                      map_location=device,
                                      weights_only=True))
    return model, device

# Function for processing the image
def process_image(image, model, device, use_center_crop=True, crop_size=128, upscale_factor=4):
    # Apply proper preprocessing
    if use_center_crop:
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        hr_transform = CenterCrop(crop_size)
        image = hr_transform(image)
    
    # Convert to tensor and prepare for model
    img_tensor = ToTensor()(image).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # Generate super-resolution image
    with torch.no_grad():
        start_time = time.time()
        output = model(img_tensor)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_time = time.time() - start_time

    # Convert back to PIL Image
    sr_image = ToPILImage()(output[0].data.cpu())
    
    # Create bicubic upscaled version for comparison
    if use_center_crop:
        lr_scale = Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        lr_image = lr_scale(image)
        bicubic_image = hr_scale(lr_image)
    else:
        # Simple 4x bicubic upscaling
        w, h = image.size
        lr_image = image.resize((w//upscale_factor, h//upscale_factor), Image.BICUBIC)
        bicubic_image = lr_image.resize((w, h), Image.BICUBIC)
    
    return sr_image, bicubic_image, lr_image, inference_time

# Main application UI
st.title("🛰️ Satellite Image Super-Resolution")
st.write("""
Upload a satellite image and enhance its resolution using SRGAN (Super-Resolution GAN).
This application uses a deep learning model specifically trained for satellite imagery.
""")

# Sidebar for options
st.sidebar.header("Model Options")
model_option = st.sidebar.selectbox(
    "Select model",
    ["1000-epoch model", "Baseline model"]
)

model_path = 'cp/netG_epoch_1000_gpu.pth' if model_option == "1000-epoch model" else 'cp/netG_baseline_gpu.pth'

preprocessing_option = st.sidebar.checkbox("Use center cropping", value=True,
                                        help="Crops image from center before processing")
crop_size = st.sidebar.slider("Crop size", 96, 256, 128, 
                             help="Size of the center crop (if enabled)")
                        
# Display model info
st.sidebar.header("Model Information")
if st.sidebar.button("Show model details"):
    model, device = load_model(model_path)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    st.sidebar.write(f"Parameters: {num_params:,}")
    st.sidebar.write(f"Device: {device}")
    st.sidebar.write(f"Upscaling factor: 4×")

# Upload image section
uploaded_file = st.file_uploader("Choose a satellite image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load model
    with st.spinner("Loading model..."):
        model, device = load_model(model_path)
    
    # Load and display original image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Create columns for display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Original Image**")
        st.image(image, use_column_width=True)
        width, height = image.size
        st.write(f"Dimensions: {width}x{height}")
    
    # Process the image
    with st.spinner("Generating super-resolution image..."):
        sr_image, bicubic_image, lr_image, inference_time = process_image(
            image, model, device, preprocessing_option, crop_size, 4
        )
    
    # Display results
    with col2:
        st.write("**Bicubic Upsampling**")
        st.image(bicubic_image, use_column_width=True)
        st.write(f"Simple interpolation")
    
    with col3:
        st.write("**SRGAN Output**")
        st.image(sr_image, use_column_width=True)
        st.write(f"Processing time: {inference_time:.3f}s")
    
    # Comparison and metrics
    st.subheader("Visual Comparison")
    
    # Add download button
    buf = io.BytesIO()
    sr_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    st.download_button(
        label="Download Super-Resolution Image",
        data=byte_im,
        file_name="sr_satellite_image.png",
        mime="image/png"
    )
    
    # Additional information about the model
    st.subheader("About this model")
    st.write("""
    This SRGAN model was specifically trained for satellite imagery super-resolution with a 4× upscaling factor.
    The model uses a deep residual network with skip connections to generate high-resolution details.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
Made with Streamlit and PyTorch | SRGAN for Satellite Image Super-Resolution
</div>
""", unsafe_allow_html=True)