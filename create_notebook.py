import nbformat as nbf

nb = nbf.v4.new_notebook()
text = """\
# Shelf Row Detection Pipeline (Deep Hough Transform)
**IMPORTANT:** This notebook MUST be run on a machine with an NVIDIA GPU (e.g., your lab computer or Google Colab). It relies on CUDA C++ extensions which are incompatible with Apple Silicon (M1/M2/M3) Macs.

This notebook will guide you through formatting the SHARD dataset and training the Deep Hough Transform (DHT) model for shelf row detection.
"""

setup_md = """\
## 1. Environment Setup
Clone the Deep Hough Transform repository and compile the custom CUDA extensions.
"""

setup_code = """\
!git clone https://github.com/Hanqer/deep-hough-transform.git
%cd deep-hough-transform

# Install dependencies
!pip install -r requirements.txt
!pip install opencv-python scipy yacs tensorboardX

# Compile the CUDA C++ extensions
%cd model/_cdht
!python setup.py build_ext --inplace
%cd ../../
"""

data_md = """\
## 2. Dataset Preparation
The SHARD dataset provides horizontal line positions as Y-coordinate percentages (e.g., `0.12`). Deep Hough Transform typically expects line segment coordinates `(x1, y1, x2, y2)` or custom JSON formats. 
Run this cell to generate the formatted dataset directory structure that DHT expects.
"""

data_code = """\
import os
import csv
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Update this path to where you extracted shelf_detection.7z on your lab computer
SHARD_DATA_DIR = '../data/24100695'
ANNOTATION_FILE = os.path.join(SHARD_DATA_DIR, 'annotation.csv')
IMAGE_DIR = os.path.join(SHARD_DATA_DIR, 'shelf_detection/images') # Adjust if 7z extracts differently

OUTPUT_DIR = 'data/shard'
os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'annotations'), exist_ok=True)

with open(ANNOTATION_FILE, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    next(reader) # Skip header
    for row in tqdm(reader, desc="Processing SHARD annotations"):
        filename = row[0]
        y_coords = [float(y) for y in row[1].split(',')]
        
        # We need image dimensions to convert percentages to pixels
        img_path = os.path.join(IMAGE_DIR, filename)
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        # Deep Hough Transform expects line segments: [x1, y1, x2, y2]
        lines = []
        for y_pct in y_coords:
            y_pixel = int(y_pct * h)
            # A horizontal line from the left edge (x=0) to the right edge (x=w)
            lines.append([0, y_pixel, w, y_pixel])
        
        # Save annotation in NKL/JSON format for the dataloader
        ann_path = os.path.join(OUTPUT_DIR, 'annotations', filename.replace('.jpg', '.json'))
        with open(ann_path, 'w') as out_f:
            json.dump({"lines": lines, "height": h, "width": w}, out_f)
            
print("Dataset conversion complete. Make sure to copy the images to data/shard/images/")
"""

train_md = """\
## 3. Training the Model
Configure the training script to use your newly formatted SHARD dataset.
"""

train_code = """\
# You will likely need to adjust the configuration file (e.g., config/nkl.yaml) 
# to point DATA.DIR to 'data/shard' before running this.

!python train.py --config config/nkl.yaml
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_markdown_cell(setup_md),
    nbf.v4.new_code_cell(setup_code),
    nbf.v4.new_markdown_cell(data_md),
    nbf.v4.new_code_cell(data_code),
    nbf.v4.new_markdown_cell(train_md),
    nbf.v4.new_code_cell(train_code)
]

with open('Shelf_Row_Detection_Pipeline.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
