import os
from os import listdir, makedirs
from os.path import join, exists
from PIL import Image
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def ensure_divisible_size(image, upscale_factor=4, method='minimal_crop'):
    """
    Ensure image dimensions are divisible by upscale_factor using different methods
    
    Args:
        image: PIL Image
        upscale_factor: Factor to ensure divisibility by
        method: One of 'minimal_crop', 'no_crop', or 'pad'
    
    Returns:
        Processed PIL Image
    """
    width, height = image.size
    
    if method == 'no_crop':
        # Resize to nearest dimensions divisible by upscale_factor
        new_width = width - (width % upscale_factor)
        new_height = height - (height % upscale_factor)
        if new_width == 0:
            new_width = upscale_factor
        if new_height == 0:
            new_height = upscale_factor
        return image.resize((new_width, new_height), Image.BICUBIC)
        
    elif method == 'pad':
        # Pad image to make dimensions divisible by upscale_factor
        new_width = width + (upscale_factor - width % upscale_factor) % upscale_factor
        new_height = height + (upscale_factor - height % upscale_factor) % upscale_factor
        new_img = Image.new('RGB', (new_width, new_height), color=(0, 0, 0))
        new_img.paste(image, (0, 0))
        return new_img
        
    else:  # minimal_crop (default)
        # Only crop the minimum pixels needed
        new_width = width - (width % upscale_factor)
        new_height = height - (height % upscale_factor)
        if new_width == 0:
            new_width = upscale_factor
        if new_height == 0:
            new_height = upscale_factor
        
        # Calculate crop box (centered)
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        
        return image.crop((left, top, right, bottom))

def generate_lowres_images(hr_dir, lr_dir, upscale_factor=4, method='minimal_crop'):
    """
    Generate low-resolution versions of images with minimal or no cropping
    
    Args:
        hr_dir: Directory with high-resolution images
        lr_dir: Directory to save low-resolution images
        upscale_factor: Downscaling factor (default: 4)
        method: How to handle non-divisible dimensions:
                'minimal_crop' - crop only what's necessary
                'no_crop' - resize without cropping
                'pad' - add padding instead of cropping
    """
    if not exists(lr_dir):
        makedirs(lr_dir)
    
    hr_image_filenames = [join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)]
    
    print(f"Processing {len(hr_image_filenames)} images from {hr_dir} to {lr_dir}")
    print(f"Method: {method}, Downscale factor: {upscale_factor}x")
    
    # Track how much is cropped on average
    crop_percentages = []
    
    for i, hr_path in enumerate(hr_image_filenames):
        filename = os.path.basename(hr_path)
        
        # Load HR image
        hr_image = Image.open(hr_path).convert('RGB')
        original_size = hr_image.size
        
        # Make dimensions divisible by upscale_factor
        hr_processed = ensure_divisible_size(hr_image, upscale_factor, method)
        processed_size = hr_processed.size
        
        # Calculate how much was cropped (percentage of pixels)
        if method == 'minimal_crop':
            original_pixels = original_size[0] * original_size[1]
            processed_pixels = processed_size[0] * processed_size[1]
            crop_percentage = 100 * (1 - processed_pixels / original_pixels)
            crop_percentages.append(crop_percentage)
        
        # Create low-resolution version
        lr_width = processed_size[0] // upscale_factor
        lr_height = processed_size[1] // upscale_factor
        lr_image = hr_processed.resize((lr_width, lr_height), Image.BICUBIC)
        
        # Save LR image
        lr_path = join(lr_dir, filename)
        lr_image.save(lr_path)
        
        # Print progress
        if (i + 1) % 10 == 0 or i == len(hr_image_filenames) - 1:
            print(f"Processed {i+1}/{len(hr_image_filenames)} images")
    
    # Print statistics on cropping
    if method == 'minimal_crop' and crop_percentages:
        avg_crop = sum(crop_percentages) / len(crop_percentages)
        print(f"Average pixels cropped: {avg_crop:.2f}%")

if __name__ == "__main__":
    # Paths
    hr_dir = "data\\val"
    lr_dir = "data\\low_res"
    
    # Choose one of: 'minimal_crop', 'no_crop', 'pad'
    processing_method = 'minimal_crop'
    
    # Generate low-res images
    generate_lowres_images(hr_dir, lr_dir, upscale_factor=4, method=processing_method)
    
    print("Done!")
    
    # Display example image dimensions
    try:
        example_hr = Image.open(join(hr_dir, listdir(hr_dir)[0]))
        example_lr = Image.open(join(lr_dir, listdir(lr_dir)[0]))
        print(f"Example HR image size: {example_hr.size}")
        print(f"Example LR image size: {example_lr.size}")
        print(f"Downscale factor: {example_hr.size[0]/example_lr.size[0]:.2f}x")
    except IndexError:
        print("No images found.")
    except Exception as e:
        print(f"Error checking example images: {e}")