import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, curPath)

import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import os.path as osp
from utils import Config
from core.Models import build_network
from utils import mkdir_or_exist, load
from utils.save_image import save_image


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


def load_image(image_path):
    """Load and preprocess image"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image from {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to float and normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Normalize with mean=0.5, std=0.5
    img = (img - 0.5) / 0.5
    
    # Convert to tensor: H x W x C -> C x H x W
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    return img


def parse_args():
    parser = argparse.ArgumentParser(description='Test single image enhancement')
    parser.add_argument('--config', type=str,
                        default='./config/UIEC2Net.py',
                        help='config file path')
    parser.add_argument('--checkpoint',
                        default='../UIEC2Net.pth',
                        help='checkpoint file path')
    parser.add_argument('--input', type=str,
                        default='./input.jpg',
                        help='input image path')
    parser.add_argument('--output', type=str,
                        default='./enhanced.jpg',
                        help='output image path')
    parser.add_argument('--no-gpu', action='store_true',
                        help='force CPU inference (default: use GPU if available)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Check if input image exists
    if not osp.exists(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")
    
    # Check if config file exists
    if not osp.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Check if checkpoint exists
    if not osp.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    print(f"Loading config from: {args.config}")
    cfg = Config.fromfile(args.config)
    
    print(f"Building model...")
    model = build_network(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    load(args.checkpoint, model, None)
    
    # Put model on GPU if available (unless --no-gpu is specified)
    device = 'cpu'
    if torch.cuda.is_available() and not args.no_gpu:
        model = model.cuda()
        device = 'cuda'
        print(f"Using GPU (CUDA available)")
    else:
        if args.no_gpu:
            print(f"Using CPU (forced by --no-gpu flag)")
        else:
            print(f"Using CPU (CUDA not available)")
    
    model.eval()
    
    print(f"Loading image from: {args.input}")
    input_image = load_image(args.input)
    
    if device == 'cuda':
        input_image = input_image.cuda()
    
    print(f"Running inference...")
    with torch.no_grad():
        output_image = model(input_image)
    
    # Convert to numpy and save
    print(f"Processing output...")
    save_cfg = False
    for i in range(len(cfg.test_pipeling)):
        if 'Normalize' == cfg.test_pipeling[i].type:
            save_cfg = True
    
    output_numpy = normimage_test(output_image, save_cfg=save_cfg, 
                                   usebytescale=cfg.usebytescale if hasattr(cfg, 'usebytescale') else False)
    
    print(f"Saving enhanced image to: {args.output}")
    save_image(output_numpy, args.output, 
               usebytescale=cfg.usebytescale if hasattr(cfg, 'usebytescale') else False)
    
    print(f"Done! Enhanced image saved to: {args.output}")


if __name__ == '__main__':
    main()
