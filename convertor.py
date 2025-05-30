import os
import pandas as pd
import numpy as np
from PIL import Image

# Load the dataset
df = pd.read_csv('IOTmalw.csv')

# Create directories for storing images
output_directory = 'cell_images/training_img'
os.makedirs(output_directory, exist_ok=True)

# Directories for malicious and benign images
malicious_dir = os.path.join(output_directory, 'malicious')
benign_dir = os.path.join(output_directory, 'benign')
os.makedirs(malicious_dir, exist_ok=True)
os.makedirs(benign_dir, exist_ok=True)

# Normalize the data to fit into the range 0-255 for RGB images
def normalize_column(column):
    min_val = column.min()
    max_val = column.max()
    return ((column - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Normalize each column except for the IP addresses and target_encoded
for col in df.columns:
    if col not in ['id.orig_h', 'id.resp_h', 'Target_encoded']:
        df[col] = normalize_column(df[col])

# Function to convert IP address to pixel values
def ip_to_pixels(ip):
    return [int(part) for part in ip.split('.')]

# Set image dimensions
image_width = 15  # Adjust as needed
image_height = 1  # Adjust as needed

# Iterate through each row in the dataframe
for index, row in df.iterrows():
    # Extract IP addresses and convert to pixel values
    orig_h_pixels = ip_to_pixels(row['id.orig_h'])
    resp_h_pixels = ip_to_pixels(row['id.resp_h'])
    
    # Extract all the remaining columns except the target_encoded column
    other_columns = row.drop(['id.orig_h', 'id.resp_h', 'Target_encoded'])
    
    # Normalize IP address values to 0-255 range
    ip_values = np.array(orig_h_pixels + resp_h_pixels, dtype=np.uint8)
    
    # Combine IP pixels and other columns
    data = np.concatenate((ip_values, other_columns.values)).astype(np.uint8)
    
    # Create a flat image with the specified width and height
    image_data = data.reshape((image_height, image_width))
    
    # Create a PIL Image object from the image data
    image = Image.fromarray(np.stack([image_data]*3, axis=-1))
    
    # Determine the folder based on target_encoded
    folder = malicious_dir if row['Target_encoded'] == 1 else benign_dir
    
    # Save the image
    image_filename = os.path.join(folder, f'image_{index}.png')
    image.save(image_filename)
