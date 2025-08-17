"""
Create a sample dataset for testing the image classification system
"""

import os
import numpy as np
from PIL import Image
import random

def create_sample_dataset():
    """Create a sample dataset with synthetic images"""
    
    # Create dataset directory structure
    dataset_dir = 'dataset'
    classes = ['normal', 'cheating', 'looking_around']
    
    # Create directories
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # Generate sample images for each class
    image_size = (128, 128)
    num_images_per_class = 50
    
    for i, class_name in enumerate(classes):
        print(f"Creating {num_images_per_class} sample images for class '{class_name}'...")
        
        for j in range(num_images_per_class):
            # Create a synthetic image with different patterns for each class
            img = np.zeros((*image_size, 3), dtype=np.uint8)
            
            if class_name == 'normal':
                # Normal: Blue-ish with some noise
                img[:, :, 2] = 150 + np.random.randint(0, 50, image_size)  # Blue channel
                img[:, :, 1] = 50 + np.random.randint(0, 30, image_size)   # Green channel
                img[:, :, 0] = 30 + np.random.randint(0, 20, image_size)   # Red channel
                
            elif class_name == 'cheating':
                # Cheating: Red-ish with specific patterns
                img[:, :, 0] = 150 + np.random.randint(0, 50, image_size)  # Red channel
                img[:, :, 1] = 30 + np.random.randint(0, 20, image_size)   # Green channel
                img[:, :, 2] = 30 + np.random.randint(0, 20, image_size)   # Blue channel
                
                # Add some diagonal patterns
                for k in range(0, image_size[0], 10):
                    img[k:k+2, :, :] = 255
                
            elif class_name == 'looking_around':
                # Looking around: Green-ish with circular patterns
                img[:, :, 1] = 150 + np.random.randint(0, 50, image_size)  # Green channel
                img[:, :, 0] = 30 + np.random.randint(0, 20, image_size)   # Red channel
                img[:, :, 2] = 30 + np.random.randint(0, 20, image_size)   # Blue channel
                
                # Add some circular patterns
                center = (image_size[0] // 2, image_size[1] // 2)
                for radius in range(10, 50, 10):
                    for angle in range(0, 360, 10):
                        x = int(center[0] + radius * np.cos(np.radians(angle)))
                        y = int(center[1] + radius * np.sin(np.radians(angle)))
                        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                            img[x-1:x+2, y-1:y+2, :] = 255
            
            # Convert to PIL Image and save as PNG
            pil_img = Image.fromarray(img)
            filename = f"{class_name}_{j+1:03d}.png"
            filepath = os.path.join(dataset_dir, class_name, filename)
            pil_img.save(filepath)
    
    print(f"✅ Sample dataset created successfully!")
    print(f"📁 Dataset location: {dataset_dir}")
    print(f"📊 Classes: {classes}")
    print(f"🖼️  Images per class: {num_images_per_class}")
    print(f"📏 Image size: {image_size}")

if __name__ == "__main__":
    create_sample_dataset()