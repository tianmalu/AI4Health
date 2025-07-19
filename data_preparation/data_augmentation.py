
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib
import csv
from PIL import Image
import cv2
matplotlib.use('Agg')  
import glob


def read_labels(label_file):
    label_dict = {}
    with open(label_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        for row in tqdm(rows, desc="Loading labels"):
            label_dict[row["file_name"]] = row["Cold (upper respiratory tract infection)"]
    return label_dict

def search_in_labels(filename, label_dict):
    base_name = os.path.splitext(filename)[0]
    
    if "_logmel" in base_name:
        base_name = base_name.replace("_logmel", "")
    
    parts = base_name.split("_")
    if len(parts) >= 2:
        key = f"{parts[0]}_{parts[1]}.wav"
    else:
        key = f"{base_name}.wav"
    
    if key in label_dict:
        return label_dict[key]
    else:
        print(f"Warning: {key} not found in labels for file {filename}")
        return None

def augment_image_mirror(img_dir, filename, label_dict):    
    try:
        if not filename.endswith('.png'):
            return None
            
        image_path = os.path.join(img_dir, filename)
        img = Image.open(image_path)
        base_name = os.path.splitext(filename)[0]

        if search_in_labels(filename, label_dict) == "C":
            img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_path = os.path.join(img_dir, f"{base_name}_flipped.png")
            img_flipped.save(flipped_path)
            
            return {
                'original_file': filename,
                'flipped_file': f"{base_name}_flipped.png",
                'status': 'success'
            }
        else:
            return {
                'original_file': filename,
                'status': 'skipped'
            }

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {'original_file': filename, 'status': 'failed', 'error': str(e)}

def check_image_sizes(image_paths):
    sizes = set()
    
    for path in image_paths:
        try:
            with Image.open(path) as img:
                sizes.add(img.size)  
        except Exception as e:
            print(f"Failed to open {path}: {e}")
    
    if len(sizes) == 1:
        print(f"✅ Size Consistent：{sizes.pop()}")
    else:
        print(f"❌ Image Sizes are not consistent：")
        for size in sizes:
            print(f"  - {size}")


if __name__ == "__main__":
    print("📊 Loading labels...")
    label_dict = read_labels("../ComParE2017_Cold_4students/lab/ComParE2017_Cold.tsv")
    print(f"Loaded {len(label_dict)} labels")
    
    data_split = ["train_files", "devel_files"]
    image_dir = "../spectrograms_variable_width"
    
    results = []
    
    for split in data_split:
        img_dir = os.path.join(image_dir, split)
        print(f"\n🎯 Processing {split}...")
        
        if not os.path.exists(img_dir):
            print(f"Directory {img_dir} not found, skipping...")
            continue
            
        png_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        print(f"Found {len(png_files)} PNG files in {split}")
        
        for filename in tqdm(png_files, desc=f"Processing {split} PNG images"):
            result = augment_image_mirror(img_dir, filename, label_dict)
            if result:
                results.append(result)
    
    if results:
        df = pd.DataFrame(results)
        
        print(f"\n📊 Processing Summary:")
        print(f"Total files processed: {len(results)}")
        print(f"Status distribution:")
        print(df['status'].value_counts())
        
        augmented_count = len(df[df['status'] == 'success'])
        print(f"🎯 Generated {augmented_count} flipped images for cold samples")
    else:
        print("❌ No files processed")
