import io
import streamlit as st
import torch
import time
import os
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from model import Generator
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Satellite Image Super-Resolution", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Function for processing the image - simplified to match sr.py
def process_image(image, model, device):
    # Convert to tensor and prepare for model (matching sr.py)
    img_tensor = Variable(ToTensor()(image)).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # Generate super-resolution image
    with torch.no_grad():
        start_time = time.time()
        output = model(img_tensor)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_time = time.time() - start_time

    # Convert back to PIL Image (matching sr.py)
    sr_image = ToPILImage()(output[0].data.cpu())
    
    return sr_image, inference_time

# Fixed function to extract a zoomed crop from an image
def get_zoomed_crop(image, center_x, center_y, crop_size, zoom_factor=2):
    """Extract a region around center_x, center_y and zoom it"""
    # Calculate crop boundaries
    half_size = crop_size // 2
    left = max(0, center_x - half_size)
    top = max(0, center_y - half_size)
    right = min(image.width, center_x + half_size)
    bottom = min(image.height, center_y + half_size)
    
    # Ensure valid crop coordinates
    if right <= left:
        right = left + 1
    if bottom <= top:
        bottom = top + 1
    
    # Crop and zoom
    cropped = image.crop((left, top, right, bottom))
    
    # Only resize if not already at original size
    if crop_size < image.width or crop_size < image.height:
        # Make it larger to see details better
        zoomed = cropped.resize((int(cropped.width * zoom_factor), 
                                int(cropped.height * zoom_factor)), 
                               Image.LANCZOS)
        return zoomed
    return cropped

# Main application UI
st.title("🛰️ Satellite Image Super-Resolution")
st.write("""
Upload a satellite image and enhance its resolution using SRGAN.
""")

# Sidebar for model selection
st.sidebar.header("Model Options")
model_option = st.sidebar.selectbox(
    "Select model",
    ["1000-epoch model", "Baseline model"]
)

model_path = 'cp/netG_epoch_1000_gpu.pth' if model_option == "1000-epoch model" else 'cp/netG_baseline_gpu.pth'
                        
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
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Image**")
        st.image(image, use_column_width=True)
        width, height = image.size
        st.write(f"Dimensions: {width}×{height}")
    
    # Process the image - simplified to match sr.py
    with st.spinner("Generating super-resolution image..."):
        sr_image, inference_time = process_image(image, model, device)
    
    # Display results
    with col2:
        st.write("**SRGAN Output**")
        st.image(sr_image, use_column_width=True)
        width, height = sr_image.size
        st.write(f"Dimensions: {width}×{height}")
        st.write(f"Processing time: {inference_time:.3f}s")
    
    # Add zoom controls
    st.subheader("Compare Details (Zoom View)")
    
    # Add zoom controls to sidebar
    st.sidebar.header("Zoom Controls")
    
    # Default to center of image
    default_x = width // 2
    default_y = height // 2
    
    # Determine max crop size (20% of image width by default)
    default_crop_size = min(width, height) // 5
    max_crop_size = min(width, height) // 2
    
    # Add sliders to control zoom position and size
    zoom_col1, zoom_col2 = st.sidebar.columns(2)
    with zoom_col1:
        center_x = st.slider("X position", 0, width, default_x)
    with zoom_col2:
        center_y = st.slider("Y position", 0, height, default_y)
    
    crop_size = st.sidebar.slider("Crop size", 2, max_crop_size, default_crop_size)
    zoom_factor = st.sidebar.slider("Zoom factor", 1, 5, 2)
    
    # Get zoomed crops
    original_crop = get_zoomed_crop(image, center_x, center_y, crop_size, zoom_factor)
    sr_crop = get_zoomed_crop(sr_image, center_x, center_y, crop_size, zoom_factor)
    
    # Display zoomed crops
    zoom_col1, zoom_col2 = st.columns(2)
    with zoom_col1:
        st.write("**Original (Zoomed)**")
        st.image(original_crop, use_column_width=True)
    
    with zoom_col2:
        st.write("**SRGAN (Zoomed)**")
        st.image(sr_crop, use_column_width=True)
        
    # Add "crosshair" indicator on main images
    st.write("""
    *The zoom region is centered at ({}, {}) with size {}×{} pixels*
    """.format(center_x, center_y, crop_size, crop_size))
    
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
    st.write("""
    This SRGAN model was specifically trained for satellite imagery super-resolution with a 4× upscaling factor.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
Made with Streamlit and PyTorch | SRGAN for Satellite Image Super-Resolution
</div>
""", unsafe_allow_html=True)