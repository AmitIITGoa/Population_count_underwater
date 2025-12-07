import sys
import os
import torch
import cv2
import numpy as np
import glob
import shutil
from tqdm import tqdm
import os.path as osp

# Add current path to sys.path to find 'utils' and 'core'
# This assumes 'utils' and 'core' are findable from this script's location
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, curPath)

# Try to import from your project structure
try:
    from utils import Config
    from core.Models import build_network
    from utils import load
except ImportError:
    print("--- ERROR ---")
    print("Could not import 'utils' or 'core'.")
    print(f"Please make sure 'utils' and 'core' directories are in: {curPath}")
    print("Or that your PYTHONPATH is set up correctly.")
    print("This script should be in the same folder as your test_video.py")
    sys.exit(1)


# --- Start: Functions copied directly from your reference code ---
# These are needed to process the images exactly as your model expects.

def normimage_test(image_tensor, save_cfg=True, usebytescale=False):
    """Convert tensor to numpy array for saving"""
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    
    if save_cfg:
        # Denormalize if normalized with mean=0.5, std=0.5
        image_numpy = (image_numpy * 0.5 + 0.5)
    
    if usebytescale:
        image_numpy = image_numpy * 255.0
    else:
        image_numpy = np.clip(image_numpy, 0, 1) * 255.0
    
    image_numpy = image_numpy.astype(np.uint8)
    return image_numpy


def preprocess_frame(frame):
    """Preprocess frame for model input"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to float and normalize to [0, 1]
    frame_float = frame_rgb.astype(np.float32) / 255.0
    
    # Normalize with mean=0.5, std=0.5
    frame_normalized = (frame_float - 0.5) / 0.5
    
    # Convert to tensor: H x W x C -> C x H x W
    frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return frame_tensor

# --- End: Copied functions ---


def load_enhancement_model(config_path, checkpoint_path):
    """Loads the enhancement model from config and checkpoint."""
    
    print(f"[1/3] Loading config from: {config_path}")
    cfg = Config.fromfile(config_path)
    
    print(f"[2/3] Building model...")
    model = build_network(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    
    print(f"[3/3] Loading checkpoint from: {checkpoint_path}")
    load(checkpoint_path, model, None)
    
    # Set up device (GPU or CPU)
    device = 'cpu'
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
        print(f"--- Model loaded on GPU (CUDA) ---")
    else:
        print(f"--- Model loaded on CPU ---")
        
    model.eval()
    
    # Get parameters needed for saving images from config
    save_cfg = False
    for i in range(len(cfg.test_pipeling)):
        if 'Normalize' == cfg.test_pipeling[i].type:
            save_cfg = True
    
    usebytescale = cfg.usebytescale if hasattr(cfg, 'usebytescale') else False
    
    return model, device, save_cfg, usebytescale


def enhance_image(model, device, frame, save_cfg, usebytescale):
    """Enhances a single image frame (numpy array)."""
    
    # 1. Preprocess the image
    frame_tensor = preprocess_frame(frame)
    
    if device == 'cuda':
        frame_tensor = frame_tensor.cuda()
        
    # 2. Run inference
    with torch.no_grad():
        output_tensor = model(frame_tensor)
        
    # 3. Post-process output back to a displayable image
    output_frame = normimage_test(output_tensor, save_cfg=save_cfg, usebytescale=usebytescale)
    
    # 4. Convert RGB back to BGR for OpenCV
    output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
    
    return output_frame_bgr


def process_dataset(input_dir, output_dir, model, device, save_cfg, usebytescale):
    """
    Processes the entire dataset, creating a new one with both
    original and enhanced images and their corresponding labels.
    """
    
    splits = ['train', 'test', 'valid']
    
    for split in splits:
        print("\n" + "="*60)
        print(f"Processing split: {split}")
        print("="*60)
        
        input_img_dir = osp.join(input_dir, split, 'images')
        input_lbl_dir = osp.join(input_dir, split, 'labels')
        
        output_img_dir = osp.join(output_dir, split, 'image')
        output_lbl_dir = osp.join(output_dir, split, 'label')
        
        # Create output directories
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_lbl_dir, exist_ok=True)
        
        # Check if input directories exist
        if not osp.exists(input_img_dir):
            print(f"Warning: Input image directory not found, skipping: {input_img_dir}")
            continue
            
        if not osp.exists(input_lbl_dir):
            print(f"Warning: Input label directory not found: {input_lbl_dir}")
            # We can still proceed, but will not copy labels
        
        # Find all image files
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            image_files.extend(glob.glob(osp.join(input_img_dir, ext)))
            
        print(f"Found {len(image_files)} images in {input_img_dir}")
        
        for img_path in tqdm(image_files, desc=f"Processing {split} images"):
            try:
                basename = os.path.basename(img_path)
                filename_no_ext, ext = os.path.splitext(basename)
                
                # --- 1. Handle Original Image and Label ---
                
                original_label_path = osp.join(input_lbl_dir, filename_no_ext + '.txt')
                output_original_img_path = osp.join(output_img_dir, basename)
                output_original_label_path = osp.join(output_lbl_dir, filename_no_ext + '.txt')
                
                # Copy original image
                shutil.copy(img_path, output_original_img_path)
                
                # Copy original label if it exists
                if osp.exists(original_label_path):
                    shutil.copy(original_label_path, output_original_label_path)
                
                # --- 2. Handle Enhanced Image and Label ---
                
                # Read the image for processing
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Warning: Could not read image {img_path}, skipping enhancement.")
                    continue
                
                # Enhance the image
                enhanced_frame = enhance_image(model, device, frame, save_cfg, usebytescale)
                
                # Define new paths for the enhanced files
                enhanced_img_name = f"{filename_no_ext}_enhanced{ext}"
                enhanced_lbl_name = f"{filename_no_ext}_enhanced.txt"
                output_enhanced_img_path = osp.join(output_img_dir, enhanced_img_name)
                output_enhanced_label_path = osp.join(output_lbl_dir, enhanced_lbl_name)
                
                # Save the enhanced image
                cv2.imwrite(output_enhanced_img_path, enhanced_frame)
                
                # Copy the *same* original label for the enhanced image
                if osp.exists(original_label_path):
                    shutil.copy(original_label_path, output_enhanced_label_path)
                    
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                
    print("\n" + "="*60)
    print("✨ Dataset processing complete! ✨")
    print(f"New dataset created at: {output_dir}")
    print("="*60)


def main():
    # --- ⬇️ PLEASE CONFIGURE THESE PATHS ⬇️ ---
    
    # 1. Your original dataset
    INPUT_DATASET_DIR = "/home/amit/experiment_comparing_model/ADVENCE_DETECTION/UWEnhancement/URPC_dataset"
    
    # 2. Where to save the new dataset
    OUTPUT_DATASET_DIR = "PROPER_URPC2020_custom_dataset"
    
    # 3. Paths to your model config and weights
    # (These are from your reference code's defaults, update them if they are different)
    CONFIG_PATH = './config/UIEC2Net.py'
    CHECKPOINT_PATH = 'UIEC2Net.pth'
    
    # --- ⬆️ END OF CONFIGURATION ⬆️ ---
    
    
    # Verify paths before starting
    if not osp.exists(INPUT_DATASET_DIR):
        print(f"Error: Input dataset directory not found: {INPUT_DATASET_DIR}")
        return
        
    if not osp.exists(CONFIG_PATH):
        print(f"Error: Config file not found: {CONFIG_PATH}")
        print("Please update the 'CONFIG_PATH' variable in the script.")
        return
        
    if not osp.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found: {CHECKPOINT_PATH}")
        print("Please update the 'CHECKPOINT_PATH' variable in the script.")
        return

    # Load the model
    try:
        model, device, save_cfg, usebytescale = load_enhancement_model(CONFIG_PATH, CHECKPOINT_PATH)
    except Exception as e:
        print(f"--- FATAL ERROR: Could not load model ---")
        print(f"{e}")
        print("Please check your config and checkpoint paths and project imports.")
        return
        
    # Run the dataset processing
    process_dataset(INPUT_DATASET_DIR, OUTPUT_DATASET_DIR, model, device, save_cfg, usebytescale)


if __name__ == '__main__':
    main()