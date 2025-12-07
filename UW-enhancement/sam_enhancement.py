import sys
import os
import torch
import cv2
import numpy as np
import glob
import shutil
import json
import copy
from tqdm import tqdm
import os.path as osp

# --- Path Setup (Keep your existing path logic) ---
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, curPath)

try:
    from utils import Config
    from core.Models import build_network
    from utils import load
except ImportError:
    print("--- ERROR ---")
    print("Could not import 'utils' or 'core'.")
    sys.exit(1)

# --- Helper Functions (Same as before) ---

def normimage_test(image_tensor, save_cfg=True, usebytescale=False):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    if save_cfg:
        image_numpy = (image_numpy * 0.5 + 0.5)
    if usebytescale:
        image_numpy = image_numpy * 255.0
    else:
        image_numpy = np.clip(image_numpy, 0, 1) * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    return image_numpy

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_float = frame_rgb.astype(np.float32) / 255.0
    frame_normalized = (frame_float - 0.5) / 0.5
    frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
    return frame_tensor

def load_enhancement_model(config_path, checkpoint_path):
    print(f"[1/3] Loading config from: {config_path}")
    cfg = Config.fromfile(config_path)
    print(f"[2/3] Building model...")
    model = build_network(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    print(f"[3/3] Loading checkpoint from: {checkpoint_path}")
    load(checkpoint_path, model, None)
    
    device = 'cpu'
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
        print(f"--- Model loaded on GPU (CUDA) ---")
    else:
        print(f"--- Model loaded on CPU ---")
    model.eval()
    
    save_cfg = False
    for i in range(len(cfg.test_pipeling)):
        if 'Normalize' == cfg.test_pipeling[i].type:
            save_cfg = True
    usebytescale = cfg.usebytescale if hasattr(cfg, 'usebytescale') else False
    return model, device, save_cfg, usebytescale

def enhance_image(model, device, frame, save_cfg, usebytescale):
    frame_tensor = preprocess_frame(frame)
    if device == 'cuda':
        frame_tensor = frame_tensor.cuda()
    with torch.no_grad():
        output_tensor = model(frame_tensor)
    output_frame = normimage_test(output_tensor, save_cfg=save_cfg, usebytescale=usebytescale)
    output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
    return output_frame_bgr

# --- NEW FUNCTION FOR COCO FORMAT ---

def process_coco_split(input_dir, output_dir, split_name, model, device, save_cfg, usebytescale):
    """
    Reads COCO JSON, enhances images, duplicates annotations for enhanced images,
    and saves a NEW merged JSON.
    """
    
    # Define paths based on your screenshot structure
    # Input: deepfish_sam_coco/train/
    input_split_path = osp.join(input_dir, split_name)
    output_split_path = osp.join(output_dir, split_name)
    
    # Make output directory
    os.makedirs(output_split_path, exist_ok=True)

    # Path to the JSON file
    json_file_name = "_annotations.coco.json"
    input_json_path = osp.join(input_split_path, json_file_name)
    output_json_path = osp.join(output_split_path, json_file_name)

    if not osp.exists(input_json_path):
        print(f"Skipping {split_name} (No JSON found at {input_json_path})")
        return

    print(f"\nProcessing split: {split_name}")
    print(f"Loading JSON: {input_json_path}")

    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    # Lists to store NEW enhanced data
    new_images_list = []
    new_annotations_list = []

    # Find the highest current IDs so we don't overlap
    # If list is empty, start at 0 or 1
    max_img_id = 0
    if coco_data['images']:
        max_img_id = max(img['id'] for img in coco_data['images'])
    
    max_ann_id = 0
    if coco_data['annotations']:
        max_ann_id = max(ann['id'] for ann in coco_data['annotations'])

    next_img_id = max_img_id + 1
    next_ann_id = max_ann_id + 1

    # Loop through every image in the JSON
    for img_info in tqdm(coco_data['images'], desc=f"Enhancing {split_name}"):
        
        file_name = img_info['file_name']
        original_id = img_info['id']
        
        # 1. Read Original Image
        img_path = osp.join(input_split_path, file_name)
        
        # Just copy the original image to output folder first
        output_orig_path = osp.join(output_split_path, file_name)
        try:
            shutil.copy(img_path, output_orig_path)
        except FileNotFoundError:
            print(f"Image not found: {img_path}")
            continue

        # 2. Enhance Image
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        enhanced_frame = enhance_image(model, device, frame, save_cfg, usebytescale)

        # 3. Save Enhanced Image
        filename_no_ext, ext = os.path.splitext(file_name)
        new_filename = f"{filename_no_ext}_enhanced{ext}"
        output_enhanced_path = osp.join(output_split_path, new_filename)
        cv2.imwrite(output_enhanced_path, enhanced_frame)

        # 4. Add Enhanced Image to JSON Structure
        # Create a copy of the image info, but change ID and filename
        new_img_info = copy.deepcopy(img_info)
        new_img_info['id'] = next_img_id
        new_img_info['file_name'] = new_filename
        new_images_list.append(new_img_info)

        # 5. Duplicate Annotations
        # Find all annotations belonging to the ORIGINAL image
        related_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == original_id]

        for ann in related_anns:
            new_ann = copy.deepcopy(ann)
            new_ann['id'] = next_ann_id
            new_ann['image_id'] = next_img_id # Link to the NEW enhanced image
            new_annotations_list.append(new_ann)
            next_ann_id += 1

        # Increment image ID for next loop
        next_img_id += 1

    # Add the new data to the main lists
    coco_data['images'].extend(new_images_list)
    coco_data['annotations'].extend(new_annotations_list)

    # Save the modified JSON
    print(f"Saving updated JSON to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f)

def main():
    # --- ⬇️ CONFIGURATION ⬇️ ---
    
    # Point this to the FOLDER containing 'train', 'test'
    INPUT_DATASET_DIR = "deepfish_sam_coco"  
    
    OUTPUT_DATASET_DIR = "deepfish_sam_coco_enhanced"
    
    CONFIG_PATH = './config/UIEC2Net.py'
    CHECKPOINT_PATH = 'UIEC2Net.pth'
    
    # --- ⬆️ END CONFIGURATION ⬆️ ---
    
    if not osp.exists(INPUT_DATASET_DIR):
        print(f"Error: Input directory not found: {INPUT_DATASET_DIR}")
        return

    # Load Model
    try:
        model, device, save_cfg, usebytescale = load_enhancement_model(CONFIG_PATH, CHECKPOINT_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Process each split (train, test, valid)
    splits = ['train', 'test', 'valid']
    
    for split in splits:
        full_split_path = osp.join(INPUT_DATASET_DIR, split)
        if osp.exists(full_split_path):
            process_coco_split(INPUT_DATASET_DIR, OUTPUT_DATASET_DIR, split, model, device, save_cfg, usebytescale)
        else:
            print(f"Split folder not found, skipping: {split}")

    print("\n" + "="*60)
    print("Done! New COCO dataset created.")
    print(f"Location: {OUTPUT_DATASET_DIR}")
    print("="*60)

if __name__ == '__main__':
    main()