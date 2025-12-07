import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, curPath)

import argparse
import torch
import cv2
import numpy as np
import time
from tqdm import tqdm
import os.path as osp
from utils import Config
from core.Models import build_network
from utils import mkdir_or_exist, load


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


def parse_args():
    parser = argparse.ArgumentParser(description='Video enhancement with underwater image enhancement models')
    parser.add_argument('--config', type=str,
                        default='./config/UIEC2Net.py',
                        help='config file path')
    parser.add_argument('--checkpoint',
                        default='../UIEC2Net.pth',
                        help='checkpoint file path')
    parser.add_argument('--input', type=str,
                        default='./output_video1.mp4',
                        help='input video path')
    parser.add_argument('--output', type=str,
                        default='./output_video1_enhanced.mp4',
                        help='output video path')
    parser.add_argument('--display', action='store_true',
                        help='display real-time processing (press q to quit)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='force CPU inference (default: use GPU if available)')
    parser.add_argument('--fps', type=int, default=None,
                        help='output video FPS (default: same as input)')
    parser.add_argument('--codec', type=str, default='mp4v',
                        help='video codec (default: mp4v, options: mp4v, h264, xvid)')
    parser.add_argument('--skip-frames', type=int, default=1,
                        help='process every Nth frame (default: 1, means process all frames)')
    args = parser.parse_args()
    return args


def get_video_codec(codec_name):
    """Get fourcc code for video codec"""
    codec_map = {
        'mp4v': cv2.VideoWriter_fourcc(*'mp4v'),
        'h264': cv2.VideoWriter_fourcc(*'H264'),
        'x264': cv2.VideoWriter_fourcc(*'X264'),
        'xvid': cv2.VideoWriter_fourcc(*'XVID'),
        'mjpg': cv2.VideoWriter_fourcc(*'MJPG'),
        'avc1': cv2.VideoWriter_fourcc(*'avc1'),
    }
    return codec_map.get(codec_name.lower(), cv2.VideoWriter_fourcc(*'mp4v'))


def main():
    args = parse_args()
    
    # Check if input video exists
    if not osp.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
    
    # Check if config file exists
    if not osp.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Check if checkpoint exists
    if not osp.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    print("="*60)
    print("VIDEO ENHANCEMENT - UNDERWATER IMAGE ENHANCEMENT")
    print("="*60)
    
    print(f"\n[1/6] Loading config from: {args.config}")
    cfg = Config.fromfile(args.config)
    
    print(f"[2/6] Building model...")
    model = build_network(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    
    print(f"[3/6] Loading checkpoint from: {args.checkpoint}")
    load(args.checkpoint, model, None)
    
    # Put model on GPU if available (unless --no-gpu is specified)
    device = 'cpu'
    if torch.cuda.is_available() and not args.no_gpu:
        model = model.cuda()
        device = 'cuda'
        print(f"[4/6] Using GPU (CUDA available)")
    else:
        if args.no_gpu:
            print(f"[4/6] Using CPU (forced by --no-gpu flag)")
        else:
            print(f"[4/6] Using CPU (CUDA not available)")
    
    model.eval()
    
    # Open input video
    print(f"\n[5/6] Opening video: {args.input}")
    cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {args.input}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use custom fps if provided
    output_fps = args.fps if args.fps is not None else fps
    
    print(f"   Video properties:")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - FPS: {fps:.2f} (output: {output_fps:.2f})")
    print(f"   - Total frames: {total_frames}")
    print(f"   - Duration: {total_frames/fps:.2f} seconds")
    if args.skip_frames > 1:
        print(f"   - Processing every {args.skip_frames} frame(s)")
    
    # Create output directory if needed
    output_dir = osp.dirname(args.output)
    if output_dir and not osp.exists(output_dir):
        mkdir_or_exist(output_dir)
    
    # Initialize video writer
    fourcc = get_video_codec(args.codec)
    out = cv2.VideoWriter(args.output, fourcc, output_fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create output video file: {args.output}")
    
    print(f"\n[6/6] Processing video...")
    print(f"   Output: {args.output}")
    print(f"   Codec: {args.codec}")
    
    # Check save_cfg from config
    save_cfg = False
    for i in range(len(cfg.test_pipeling)):
        if 'Normalize' == cfg.test_pipeling[i].type:
            save_cfg = True
    
    usebytescale = cfg.usebytescale if hasattr(cfg, 'usebytescale') else False
    
    # Process video
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    # Create progress bar
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames if needed
            if args.skip_frames > 1 and (frame_count - 1) % args.skip_frames != 0:
                # For skipped frames, just write the original frame
                out.write(frame)
                pbar.update(1)
                continue
            
            # Preprocess frame
            frame_tensor = preprocess_frame(frame)
            
            if device == 'cuda':
                frame_tensor = frame_tensor.cuda()
            
            # Run inference
            with torch.no_grad():
                output_tensor = model(frame_tensor)
            
            # Post-process output
            output_frame = normimage_test(output_tensor, save_cfg=save_cfg, usebytescale=usebytescale)
            
            # Convert RGB back to BGR for OpenCV
            output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            
            # Write frame to output video
            out.write(output_frame_bgr)
            
            processed_count += 1
            
            # Display if requested
            if args.display:
                # Create side-by-side comparison
                comparison = np.hstack([frame, output_frame_bgr])
                
                # Add text labels
                cv2.putText(comparison, 'Original', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(comparison, 'Enhanced', (width + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Video Enhancement (Press Q to quit)', comparison)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\nQuitting on user request...")
                    break
            
            # Update progress bar with FPS info
            elapsed_time = time.time() - start_time
            processing_fps = processed_count / elapsed_time if elapsed_time > 0 else 0
            pbar.set_postfix({'FPS': f'{processing_fps:.2f}'})
            pbar.update(1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user...")
    
    finally:
        pbar.close()
        
        # Release resources
        cap.release()
        out.release()
        
        if args.display:
            cv2.destroyAllWindows()
        
        # Print summary
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Total frames: {frame_count}")
        print(f"Processed frames: {processed_count}")
        print(f"Skipped frames: {frame_count - processed_count}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Average FPS: {processed_count / elapsed_time:.2f}")
        print(f"Output saved to: {args.output}")
        print("="*60)


if __name__ == '__main__':
    main()
