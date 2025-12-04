#!/usr/bin/env python3
"""
main_global_feature_edge.py - Version BEFORE anti-ID-swap changes

This version includes:
- Edge detection with direction consistency (fish entering vs exiting)
- Mid-area appearance handling
- Proximity-based matching at edges
- Color coding: Yellow (new) -> Green (confirmed)

Does NOT include:
- Trajectory history tracking
- Swap protection mechanisms
- Close pair detection
- Trajectory deviation costs
"""

import os, time, math, cv2, torch, numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO
import torchvision.transforms as T
import torchvision.models as models
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from PIL import Image
from typing import List, Tuple, Optional, Dict

# ---------------- USER CONFIG ----------------
VIDEO_PATH = "output_video2_enhanced.mp4"
OUTPUT_PATH = "output_video2_edge_fixed.mp4"
STATS_OUTPUT = "population_statistics_edge_fixed.txt"
MODEL_PATH = "best.pt"

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REID_DEVICE = "cpu"

# --- TUNING ---
CONF_THRESHOLD = 0.45
NMS_IOU = 0.45
BASE_COSINE_THRESH = 0.40
IOU_THRESHOLD = 0.30

# PHYSICS LIMITS
MAX_PIXEL_SPEED = 40.0

# Timeouts
MAX_MISS_FRAMES = 80
MAX_GAP_FRAMES = 120
INACTIVE_AFTER_MISSES = 45

# Memory & Features
MEMORY_BANK_SIZE = 120
LONG_TERM_SIZE = 120
EMA_ALPHA = 0.8
DEDUP_THRESHOLD = 0.15

# Resurrection & Edge - ENHANCED
RESURRECT_AGE_FRAMES = 80
EDGE_MARGIN = 30
EDGE_REID_DISTANCE_THRESHOLD = 0.20
EDGE_DIRECTION_CONSISTENCY_REQUIRED = True

# Occlusion Handling
OCCLUSION_IOU_THRESHOLD = 0.3  # IOU overlap indicating potential occlusion
OCCLUSION_BUFFER_FRAMES = 50   # Keep occluded tracks alive longer
RESURRECTION_APPEARANCE_THRESHOLD = 0.30  # Stricter for resurrection
OCCLUSION_POSITION_RADIUS = 120  # Radius around occlusion point to check
OCCLUSION_MEMORY_FRAMES = 40   # How long to remember occlusion zones

# Crowd Handling
CROWD_DISTANCE_THRESHOLD = 100  # Distance to consider fish in crowd
CROWD_MIN_NEIGHBORS = 2  # Minimum neighbors to be considered in crowd
CROWD_MATCH_THRESHOLD = 0.30  # Stricter matching threshold in crowds
TRAJECTORY_HISTORY_LENGTH = 20  # Track position history for trajectory
TRAJECTORY_DEVIATION_THRESHOLD = 80  # Max deviation from expected trajectory
SWAP_PROTECTION_DISTANCE = 80  # Distance to activate swap protection

# Multicrop
MULTICROP = True
CROP_SCALES = [1.0, 0.9, 0.7]

# Histograms
HIST_BINS = 32
HIST_COMP_METHOD = cv2.HISTCMP_CORREL

# Confirmation & Locking
N_CONFIRM = 5
HIL_LOCK_HITS = 5
OSM_MAX_SURVIVAL = 120

# Weights
RBM_EMA_WEIGHT = 0.5
RBM_MEMORY_WEIGHT = 0.3
RBM_SHORT_WEIGHT = 0.2

HIL_APP_WEIGHT = 0.7
HIL_MOTION_WEIGHT = 0.25
HIL_SHAPE_WEIGHT = 0.05

DEBUG = False

# ---------------- UTILITIES ----------------
def iou(a: np.ndarray, b: np.ndarray) -> float:
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    areaB = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def centroid(b: np.ndarray) -> Tuple[float, float]:
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)

def xywh_from_xyxy(b: np.ndarray) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0; cy = y1 + h / 2.0
    return cx, cy, w, h

def L2_normalize(x: np.ndarray) -> Optional[np.ndarray]:
    if x is None: return None
    n = np.linalg.norm(x)
    return x / (n + 1e-8) if n > 0 else np.zeros_like(x)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 1.0
    a = a / (np.linalg.norm(a) + 1e-8); b = b / (np.linalg.norm(b) + 1e-8)
    cos_sim = float(np.dot(a, b))
    return (1.0 - cos_sim) / 2.0

def euclidean_distance(p1: Tuple[float,float], p2: Tuple[float,float]) -> float:
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def compute_histogram_rgb(img_bgr, bbox):
    if img_bgr is None: return None
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0: return None
    chans = cv2.split(roi)
    hist = []
    for ch in chans:
        hst = cv2.calcHist([ch], [0], None, [HIST_BINS], [0,256])
        cv2.normalize(hst, hst)
        hist.append(hst)
    return np.vstack(hist).flatten()

def hist_similarity(h1, h2):
    if h1 is None or h2 is None: return 0.0
    try:
        return float(cv2.compareHist(h1.astype('float32'), h2.astype('float32'), HIST_COMP_METHOD))
    except Exception:
        return 0.0

def motion_distance_gate(det_centroid, track_centroid, track_velocity, max_distance=140.0):
    predicted_pos = np.array(track_centroid) + track_velocity * 2.0
    dist = euclidean_distance(det_centroid, predicted_pos)
    if dist > max_distance: return 1.0
    return min(dist / max_distance, 1.0)

def get_edge_zone(bbox, frame_w, frame_h, margin=EDGE_MARGIN):
    """
    Returns which edge zone a bbox is in: 'left', 'right', 'top', 'bottom', or None
    """
    x1, y1, x2, y2 = map(float, bbox)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    if x1 < margin: return 'left'
    if x2 > (frame_w - margin): return 'right'
    if y1 < margin: return 'top'
    if y2 > (frame_h - margin): return 'bottom'
    return None

def check_direction_consistency(track_exit_zone, detection_entry_zone):
    """
    Check if the detection appears from the same edge where track disappeared.
    This prevents assigning old IDs to new fish entering from different edges.
    """
    if track_exit_zone is None or detection_entry_zone is None:
        return False
    return track_exit_zone == detection_entry_zone

def is_entering_from_edge(bbox, velocity, edge_zone, frame_w, frame_h):
    """
    Check if fish is ENTERING from edge (moving inward), not exiting.
    Returns True only if velocity direction indicates entering motion.
    """
    if edge_zone is None or velocity is None:
        return False
    
    vx, vy = velocity
    vel_magnitude = np.linalg.norm(velocity)
    
    # If velocity is too small, can't determine direction reliably
    if vel_magnitude < 2.0:
        return True  # Allow if stationary at edge (might be just appearing)
    
    # Check if moving INWARD from the edge
    if edge_zone == 'left' and vx > 0:  # Moving right (inward)
        return True
    elif edge_zone == 'right' and vx < 0:  # Moving left (inward)
        return True
    elif edge_zone == 'top' and vy > 0:  # Moving down (inward)
        return True
    elif edge_zone == 'bottom' and vy < 0:  # Moving up (inward)
        return True
    
    return False  # Moving outward or wrong direction

def is_mid_area_appearance(bbox, frame_w, frame_h, margin=EDGE_MARGIN*2):
    """
    Check if detection appears in mid-area (not near edges).
    Fish appearing far from edges should be treated as new entries.
    """
    x1, y1, x2, y2 = map(float, bbox)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    # Check if center is far from all edges
    far_from_left = cx > margin
    far_from_right = cx < (frame_w - margin)
    far_from_top = cy > margin
    far_from_bottom = cy < (frame_h - margin)
    
    return far_from_left and far_from_right and far_from_top and far_from_bottom

# ---------------- KALMAN FILTER ----------------
class KalmanFilterCV:
    def __init__(self, cx: float, cy: float, dt: float = 1.0):
        self.dt = dt
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 10.0
        self.Q = np.eye(4) * 1.0
        self.R = np.eye(2) * 20.0
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
    def predict(self, dt: float = 1.0) -> np.ndarray:
        self.dt = dt
        self.F[0,2]=dt; self.F[1,3]=dt
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.x
    def update(self, cx: float, cy: float):
        z = np.array([cx, cy], dtype=float)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(z - self.H.dot(self.x))
        self.P = (np.eye(len(self.x)) - K.dot(self.H)).dot(self.P)
    def get_velocity(self): return self.x[2:4]

# ---------------- ReID model ----------------
class ReIDModel:
    def __init__(self, device: str = REID_DEVICE):
        self.device = REID_DEVICE
        self.input_size = (256,128)
        backbone = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
        self.model.eval()
        self.norm = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        print("ReID Model Loaded on CPU.")

    def _letterbox(self, roi_bgr):
        roi_pil = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
        target_h, target_w = self.input_size
        old_size = roi_pil.size
        ratio = min(float(target_w)/old_size[0], float(target_h)/old_size[1])
        new_size = tuple([int(x*ratio) for x in old_size])
        roi_pil = roi_pil.resize(new_size, Image.Resampling.BILINEAR)
        new_im = Image.new("RGB", (target_w, target_h), (0,0,0))
        new_im.paste(roi_pil, ((target_w-new_size[0])//2, (target_h-new_size[1])//2))
        return new_im

    def _crop_variants(self, frame_bgr, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h_img, w_img = frame_bgr.shape[:2]
        x1, y1 = max(0,x1), max(0,y1); x2, y2 = min(w_img,x2), min(h_img,y2)
        w_box = max(1, x2-x1); h_box = max(1, y2-y1)
        cx = x1 + w_box/2.0; cy = y1 + h_box/2.0
        crops = []
        for scale in CROP_SCALES:
            nw = max(1, int(w_box*scale)); nh = max(1, int(h_box*scale))
            nx1 = int(cx - nw/2.0); ny1 = int(cy - nh/2.0); nx2 = nx1+nw; ny2 = ny1+nh
            nx1, ny1 = max(0,nx1), max(0,ny1); nx2, ny2 = min(w_img,nx2), min(h_img,ny2)
            roi = frame_bgr[ny1:ny2, nx1:nx2]
            if roi.size > 0: crops.append(self._letterbox(roi))
        return crops

    def extract_batch(self, frame_bgr: np.ndarray, bboxes: List[List[float]]) -> List[Optional[np.ndarray]]:
        all_features = [None]*len(bboxes)
        imgs = []; idx_map=[]
        for i,bbox in enumerate(bboxes):
            crops = self._crop_variants(frame_bgr, bbox) if MULTICROP else []
            if not crops:
                x1,y1,x2,y2 = map(int,bbox)
                roi = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                if roi.size > 0: crops=[self._letterbox(roi)]
            for pil in crops:
                imgs.append(self.norm(pil)); idx_map.append(i)
                imgs.append(self.norm(pil.transpose(Image.FLIP_LEFT_RIGHT))); idx_map.append(i)
        if not imgs: return [None]*len(bboxes)
        batch = torch.stack(imgs, dim=0).to(self.device)
        with torch.no_grad():
            feats = self.model(batch).reshape(len(batch), -1).cpu().numpy()
        per_idx = defaultdict(list)
        for idx,f in zip(idx_map, feats): per_idx[idx].append(f)
        for i in range(len(bboxes)):
            if i in per_idx:
                all_features[i] = L2_normalize(np.mean(np.stack(per_idx[i],0), axis=0))
        return all_features

# ---------------- Memory & Track ----------------
class EnhancedFeatureMemoryBank:
    def __init__(self):
        self.long_term = deque(maxlen=LONG_TERM_SIZE)
    def add(self, feat):
        if feat is None: return
        if len(self.long_term)>0:
            dists = cdist([feat], list(self.long_term), metric='cosine')[0]
            if dists.min() < DEDUP_THRESHOLD: return
        self.long_term.append(feat)
    def mean_long_term(self):
        if not self.long_term: return None
        return L2_normalize(np.mean(np.array(list(self.long_term)), axis=0))

class Track:
    def __init__(self, pid, bbox, frame_idx, feat=None, hist=None):
        self.pid = pid
        self.bbox = np.array(bbox, dtype=float)
        self.kf = KalmanFilterCV(*centroid(bbox))
        self.last_seen = frame_idx
        self.miss_count = 0
        self.hits = 1
        self.consecutive_hits = 1
        self.memory = EnhancedFeatureMemoryBank()
        if feat is not None: self.memory.add(feat)
        self.smoothed_feat = feat.copy() if feat is not None else None
        self.short_term = deque(maxlen=5)
        if feat is not None: self.short_term.append(feat)
        self.hist = hist
        self.is_confirmed = False
        self.confirm_count = 1
        self.locked = False
        self.occluded = False
        self.occlusion_timer = 0
        self.counted = False
        
        # Track edge exit information
        self.last_edge_zone = None
        self.velocity_history = deque(maxlen=10)
        
        # Occlusion tracking
        self.occlusion_position = None  # Store position where fish was last seen before occlusion
        self.occlusion_feature = None   # Store strong feature before occlusion
        self.was_occluded = False       # Track if fish was recently occluded
        
        # Trajectory and crowd tracking
        self.trajectory_history = deque(maxlen=TRAJECTORY_HISTORY_LENGTH)
        self.trajectory_history.append(centroid(bbox))
        self.swap_protection_active = False  # Flag when near other fish
        self.nearby_track_ids = set()  # IDs of nearby fish

    def predict(self): self.kf.predict()
    
    def update(self, bbox, frame_idx, feat=None, hist=None):
        # Store velocity and trajectory before update
        old_cent = centroid(self.bbox)
        new_cent = centroid(bbox)
        velocity = np.array([new_cent[0] - old_cent[0], new_cent[1] - old_cent[1]])
        self.velocity_history.append(velocity)
        self.trajectory_history.append(new_cent)
        
        self.bbox = np.array(bbox, dtype=float)
        cx, cy, _, _ = xywh_from_xyxy(bbox)
        self.kf.update(cx, cy)
        self.last_seen = frame_idx
        self.hits += 1
        self.consecutive_hits += 1
        self.miss_count = 0
        if feat is not None:
            if self.smoothed_feat is None: self.smoothed_feat = feat.copy()
            else: self.smoothed_feat = EMA_ALPHA * self.smoothed_feat + (1.0-EMA_ALPHA) * feat
            self.short_term.append(feat)
            self.memory.add(self.smoothed_feat)
        if hist is not None: self.hist = hist
        
        # If recovering from occlusion, mark it
        if self.was_occluded:
            self.was_occluded = False
        
        self.confirm_count += 1
        if self.confirm_count >= N_CONFIRM: self.is_confirmed = True
        
        if not self.occluded and self.consecutive_hits >= HIL_LOCK_HITS:
            self.locked = True
        self.occluded = False; self.occlusion_timer = 0

    def increment_miss(self):
        self.miss_count += 1
        self.consecutive_hits = 0
        self.occlusion_timer += 1
        
        # Store occlusion info on first miss
        if self.occlusion_timer == 1:
            self.occlusion_position = centroid(self.bbox)
            self.occlusion_feature = self.smoothed_feat.copy() if self.smoothed_feat is not None else None
            self.was_occluded = True
            
        if self.occlusion_timer > 0:
            self.occluded = True

    def get_reid_feature(self):
        if self.smoothed_feat is not None: return L2_normalize(self.smoothed_feat)
        return self.memory.mean_long_term()
    
    def get_short_term_mean(self):
        if not self.short_term: return None
        return L2_normalize(np.mean(np.array(list(self.short_term)), axis=0))
    
    def get_avg_velocity(self):
        """Get average velocity direction over recent frames"""
        if not self.velocity_history: return np.array([0.0, 0.0])
        return np.mean(np.array(list(self.velocity_history)), axis=0)
    
    def predict_trajectory(self, frames_ahead=3):
        """Predict future position based on trajectory"""
        if len(self.trajectory_history) < 3:
            return centroid(self.bbox)
        
        # Use recent positions to predict
        recent = list(self.trajectory_history)[-5:]
        velocities = []
        for i in range(1, len(recent)):
            vx = recent[i][0] - recent[i-1][0]
            vy = recent[i][1] - recent[i-1][1]
            velocities.append((vx, vy))
        
        if velocities:
            avg_vx = np.mean([v[0] for v in velocities])
            avg_vy = np.mean([v[1] for v in velocities])
            curr = recent[-1]
            predicted = (curr[0] + avg_vx * frames_ahead, curr[1] + avg_vy * frames_ahead)
            return predicted
        return centroid(self.bbox)
    
    def get_trajectory_deviation(self, new_centroid):
        """Calculate how much a new position deviates from expected trajectory"""
        if len(self.trajectory_history) < 3:
            return 0.0
        
        predicted = self.predict_trajectory(1)
        deviation = euclidean_distance(predicted, new_centroid)
        return deviation

# ---------------- TRACKER ----------------
class EnhancedDeepSORTTracker:
    def __init__(self, device="cuda"):
        self.tracks: Dict[int, Track] = {}
        self.next_pid = 0
        self.reid = ReIDModel(REID_DEVICE)
        self.total_unique_fishes = 0
        self.deleted_tracks = deque(maxlen=1500)
        self.frame_w = None; self.frame_h = None
        # Track occlusion zones: {pid: (position, frame_idx, feature)}
        self.occlusion_zones = {}

    def set_frame_size(self, w:int, h:int):
        self.frame_w = w; self.frame_h = h

    def update_count(self):
        for pid, track in self.tracks.items():
            if track.is_confirmed and not track.counted:
                self.total_unique_fishes += 1
                track.counted = True

    def _rbm_fuse(self, det_feat, track: Track):
        ema = track.smoothed_feat; mem = track.memory.mean_long_term(); short = track.get_short_term_mean()
        parts = []; weights = []
        if ema is not None: parts.append(ema); weights.append(RBM_EMA_WEIGHT)
        if mem is not None: parts.append(mem); weights.append(RBM_MEMORY_WEIGHT)
        if short is not None: parts.append(short); weights.append(RBM_SHORT_WEIGHT)
        if not parts: return det_feat
        wsum = sum(weights)
        agg = sum(w * p for w,p in zip(weights, parts)) / (wsum + 1e-12)
        return L2_normalize(agg)
    
    def _check_occlusion_with_objects(self, detections):
        """
        Check if any tracks might be occluded by detecting spatial overlaps
        """
        occluded_tracks = set()
        
        for pid, track in self.tracks.items():
            if track.miss_count == 0:
                continue  # Track is currently matched
                
            track_bbox = track.bbox
            
            # Check if track area overlaps with any current detection
            for det in detections:
                iou_val = iou(track_bbox, det)
                if iou_val > OCCLUSION_IOU_THRESHOLD:
                    occluded_tracks.add(pid)
                    break
                    
        return occluded_tracks
    
    def _is_in_occlusion_zone(self, bbox, feat, current_frame_idx):
        """
        Check if detection is in a recently occluded zone and might be the occluded fish
        Returns (should_avoid_new_id, best_occluded_pid)
        """
        det_cent = centroid(bbox)
        best_pid = None
        best_score = 1e9
        
        # Clean old occlusion zones
        pids_to_remove = []
        for pid, (pos, frame, occlusion_feat) in self.occlusion_zones.items():
            if current_frame_idx - frame > OCCLUSION_MEMORY_FRAMES:
                pids_to_remove.append(pid)
        for pid in pids_to_remove:
            del self.occlusion_zones[pid]
        
        # Check if detection is near any occlusion zone
        for pid, (pos, frame, occlusion_feat) in self.occlusion_zones.items():
            dist = euclidean_distance(det_cent, pos)
            if dist < OCCLUSION_POSITION_RADIUS:
                # This detection is near an occlusion zone
                if feat is not None and occlusion_feat is not None:
                    appearance_dist = cosine_distance(feat, occlusion_feat)
                    if appearance_dist < best_score:
                        best_score = appearance_dist
                        best_pid = pid
        
        # If very similar appearance in occlusion zone, it's likely the same fish
        if best_pid is not None and best_score < RESURRECTION_APPEARANCE_THRESHOLD:
            return True, best_pid
        
        return False, None
    
    def _detect_crowds_and_close_pairs(self, frame_idx):
        """
        Detect which tracks are in crowded areas or close to each other.
        Activates swap protection for tracks in crowds.
        """
        track_list = list(self.tracks.items())
        
        # Reset nearby tracking
        for pid, track in self.tracks.items():
            track.nearby_track_ids = set()
            track.swap_protection_active = False
        
        # Find close pairs and crowds
        for i, (pid1, track1) in enumerate(track_list):
            if frame_idx - track1.last_seen > 5:
                continue  # Skip inactive tracks
            
            c1 = centroid(track1.bbox)
            neighbor_count = 0
            
            for j, (pid2, track2) in enumerate(track_list):
                if i >= j or frame_idx - track2.last_seen > 5:
                    continue
                
                c2 = centroid(track2.bbox)
                dist = euclidean_distance(c1, c2)
                
                if dist < CROWD_DISTANCE_THRESHOLD:
                    neighbor_count += 1
                    track1.nearby_track_ids.add(pid2)
                    track2.nearby_track_ids.add(pid1)
                    
                    # Activate swap protection for close pairs
                    if dist < SWAP_PROTECTION_DISTANCE:
                        track1.swap_protection_active = True
                        track2.swap_protection_active = True
            
            # Mark as in crowd if enough neighbors
            if neighbor_count >= CROWD_MIN_NEIGHBORS:
                track1.swap_protection_active = True

    def _edge_active_match(self, feat, bbox, hist, current_frame_idx):
        """
        Enhanced edge matching: Prioritizes existing nearby tracks over entry/exit logic.
        If a track is very close, it's likely the same fish (even if exiting).
        """
        if feat is None or self.frame_w is None: return None
        
        # Check if detection is at edge
        det_edge_zone = get_edge_zone(bbox, self.frame_w, self.frame_h, EDGE_MARGIN)
        if det_edge_zone is None: return None
        
        det_cent = centroid(bbox)
        best_pid = None; best_score = 1e9
        nearby_track_pid = None; min_distance = float('inf')
        
        for pid, tr in self.tracks.items():
            time_gap = current_frame_idx - tr.last_seen
            if time_gap > INACTIVE_AFTER_MISSES: continue
            
            # --- PHYSICS GATE ---
            tr_cent = centroid(tr.bbox)
            dist_px = euclidean_distance(det_cent, tr_cent)
            speed = dist_px / (time_gap + 1e-5)
            if speed > MAX_PIXEL_SPEED: continue
            # --------------------
            
            # Track nearby detections separately (within 50px = likely same fish)
            if dist_px < 50 and dist_px < min_distance:
                min_distance = dist_px
                nearby_track_pid = pid
            
            # --- DIRECTION CONSISTENCY CHECK ---
            tr_exit_zone = get_edge_zone(tr.bbox, self.frame_w, self.frame_h, EDGE_MARGIN)
            
            if EDGE_DIRECTION_CONSISTENCY_REQUIRED:
                if not check_direction_consistency(tr_exit_zone, det_edge_zone):
                    continue
                
                # Compute velocity from last known position to current detection
                velocity = np.array([det_cent[0] - tr_cent[0], det_cent[1] - tr_cent[1]])
                
                # If track is nearby (< 50px), allow both entering AND exiting
                if dist_px > 50:
                    # For distant matches, verify the fish is moving INWARD from the edge
                    if not is_entering_from_edge(bbox, velocity, det_edge_zone, self.frame_w, self.frame_h):
                        continue  # Skip if fish is exiting or moving wrong direction
            # ----------------------------------------
            
            tr_feat = tr.get_reid_feature(); tr_hist = tr.hist
            if tr_feat is None: continue
            
            fd = cosine_distance(feat, tr_feat)
            hs = hist_similarity(hist, tr_hist) if (hist is not None and tr_hist is not None) else 0.0
            
            # Stricter threshold for edge cases
            score = 0.75*fd + 0.25*(1.0-hs)
            if score < best_score: 
                best_score = score
                best_pid = pid
        
        # PRIORITY 1: If there's a very close track (<50px), use it regardless of direction
        if nearby_track_pid is not None and min_distance < 50:
            return nearby_track_pid
        
        # PRIORITY 2: Use best appearance match if good enough
        if best_score < EDGE_REID_DISTANCE_THRESHOLD: 
            return best_pid
        return None

    def _check_deleted_resurrect(self, feat, bbox, hist, current_frame_idx):
        if feat is None or not self.deleted_tracks: return None
        det_cent = centroid(bbox)
        
        # Check if detection is in mid-area (far from edges)
        is_mid_area = is_mid_area_appearance(bbox, self.frame_w, self.frame_h)
        
        # Check if at edge
        det_edge_zone = get_edge_zone(bbox, self.frame_w, self.frame_h, EDGE_MARGIN)
        is_at_edge = det_edge_zone is not None
        
        # Mid-area appearances from long distances should NOT resurrect old IDs
        if is_mid_area:
            return None  # Treat as new fish
        
        best = None; best_score = 1e9
        nearby_pid = None; min_distance = float('inf')
        
        for entry in reversed(self.deleted_tracks):
            # Handle both old (7-tuple) and new (8-tuple) format
            if len(entry) == 8:
                del_feat, del_bbox, del_frame, del_hist, del_vel, del_pid, del_edge_zone, del_occluded = entry
            else:
                del_feat, del_bbox, del_frame, del_hist, del_vel, del_pid, del_edge_zone = entry
                del_occluded = False
            
            age = current_frame_idx - del_frame
            if age > RESURRECT_AGE_FRAMES: continue
            
            # --- PHYSICS GATE ---
            del_cent = centroid(del_bbox)
            dist_px = euclidean_distance(det_cent, del_cent)
            speed = dist_px / (age + 1e-5)
            if speed > MAX_PIXEL_SPEED: continue
            # --------------------
            
            # ENHANCED: For occluded tracks, be more lenient with distance but stricter with appearance
            if del_occluded:
                # Occluded fish should reappear near occlusion point
                if dist_px < 200 and age < OCCLUSION_BUFFER_FRAMES:
                    # Very strict appearance matching for occluded tracks
                    fd = cosine_distance(feat, del_feat) if del_feat is not None else 1.0
                    if fd < RESURRECTION_APPEARANCE_THRESHOLD:
                        # Strong match for occluded track
                        if dist_px < min_distance:
                            min_distance = dist_px
                            nearby_pid = del_pid
            else:
                # Normal resurrection logic
                # Track very close deleted tracks (within 80px)
                if dist_px < 80 and age < 15 and dist_px < min_distance:
                    min_distance = dist_px
                    nearby_pid = del_pid
            
            # --- EDGE DIRECTION CHECK for resurrection ---
            if is_at_edge and EDGE_DIRECTION_CONSISTENCY_REQUIRED:
                # Only resurrect if edge zones are consistent
                if not check_direction_consistency(del_edge_zone, det_edge_zone):
                    continue
                
                # For nearby deleted tracks, allow both directions
                if dist_px > 80 or age > 10:
                    # CRITICAL: Verify fish is ENTERING from edge (not exiting)
                    velocity = np.array([det_cent[0] - del_cent[0], det_cent[1] - del_cent[1]])
                    if not is_entering_from_edge(bbox, velocity, det_edge_zone, self.frame_w, self.frame_h):
                        continue  # Skip if exiting or wrong direction
            # ----------------------------------------------
            
            fd = cosine_distance(feat, del_feat) if del_feat is not None else 1.0
            hs = hist_similarity(hist, del_hist) if (hist is not None and del_hist is not None) else 0.0
            iouv = iou(bbox, del_bbox)
            
            # Stricter scoring for occluded tracks
            if del_occluded:
                score = 0.8*fd + 0.15*(1.0 - hs) + 0.05*(1.0 - iouv)
            else:
                score = 0.6*fd + 0.25*(1.0 - hs) + 0.15*(1.0 - iouv)
            
            if score < best_score:
                best_score = score
                best = (del_pid, del_frame, fd, iouv, hs, dist_px, age, del_occluded)

        # PRIORITY 1: Nearby occluded track with good appearance match
        if nearby_pid is not None and min_distance < 150:
            # Extra validation: check if this makes sense trajectory-wise
            for entry in reversed(self.deleted_tracks):
                entry_pid = entry[5] if len(entry) >= 6 else None
                if entry_pid == nearby_pid:
                    del_feat = entry[0] if len(entry) > 0 else None
                    if del_feat is not None and feat is not None:
                        appearance_check = cosine_distance(feat, del_feat)
                        if appearance_check < RESURRECTION_APPEARANCE_THRESHOLD:
                            return nearby_pid
                    break
            return nearby_pid

        if best is None: return None
        pid, del_frame, fd, iouv, hs, dist_px, age, was_occluded = best
        
        # Different thresholds based on occlusion state
        if was_occluded:
            threshold = RESURRECTION_APPEARANCE_THRESHOLD
            cond = (fd < threshold) and (age < OCCLUSION_BUFFER_FRAMES) and (dist_px < 200)
        else:
            threshold = 0.18 if is_at_edge else 0.20
            cond = (fd < threshold) and ((current_frame_idx - del_frame) < RESURRECT_AGE_FRAMES) and (iouv > 0.2 or hs > 0.3)
        
        if cond: return pid
        return None

    def match(self, detections, feats, frame_idx, frame_wh: Tuple[int,int]=None, frame_img=None):
        if frame_wh is not None: self.set_frame_size(*frame_wh)
        det_hists = [compute_histogram_rgb(frame_img, d) if frame_img is not None else None for d in detections]
        
        # Update edge zones for active tracks
        for pid, track in self.tracks.items():
            track.last_edge_zone = get_edge_zone(track.bbox, self.frame_w, self.frame_h, EDGE_MARGIN)
        
        # Check for occlusions
        occluded_pids = self._check_occlusion_with_objects(detections)
        
        # Detect crowds and close pairs
        self._detect_crowds_and_close_pairs(frame_idx)
        
        # Predict
        for t in self.tracks.values(): t.predict()
        
        # Empty tracks -> Initialize all
        if not self.tracks:
            assigns=[]
            for i,d in enumerate(detections):
                pid=self.next_pid; self.next_pid+=1
                tr=Track(pid,d,frame_idx,feat=feats[i],hist=det_hists[i])
                self.tracks[pid]=tr; assigns.append((pid,d))
            return assigns

        # Cost Matrix
        track_ids = list(self.tracks.keys())
        cost_matrix = np.ones((len(detections), len(track_ids)), dtype=float) * 1e6
        
        for d_idx, (det, feat, hist) in enumerate(zip(detections, feats, det_hists)):
            det_c = centroid(det)
            
            for t_idx, pid in enumerate(track_ids):
                track = self.tracks[pid]
                if (frame_idx - track.last_seen > MAX_GAP_FRAMES) and not track.is_confirmed: continue

                # ENHANCED: Use occlusion feature if track was recently occluded
                track_feat = track.occlusion_feature if (track.was_occluded and track.occlusion_feature is not None) else track.get_reid_feature()
                
                fused = self._rbm_fuse(feat, track)
                app_cost = cosine_distance(feat, fused) if (feat is not None and fused is not None) else 0.9
                hist_cost = 1.0 - hist_similarity(hist, track.hist)
                
                tr_c = centroid(track.bbox)
                dist_cost = motion_distance_gate(det_c, tr_c, track.kf.get_velocity())
                
                # Calculate trajectory deviation cost
                trajectory_cost = 0.0
                if len(track.trajectory_history) >= 3:
                    deviation = track.get_trajectory_deviation(det_c)
                    if deviation > TRAJECTORY_DEVIATION_THRESHOLD:
                        trajectory_cost = 0.20  # Heavy penalty for unexpected movement
                    else:
                        trajectory_cost = deviation / TRAJECTORY_DEVIATION_THRESHOLD * 0.15
                
                # ENHANCED: Check if detection is near occlusion position
                if track.was_occluded and track.occlusion_position is not None:
                    dist_from_occlusion = euclidean_distance(det_c, track.occlusion_position)
                    if dist_from_occlusion < OCCLUSION_POSITION_RADIUS:
                        # Strongly boost this match - likely the same fish reappearing
                        app_cost *= 0.5  # More aggressive boost
                        dist_cost *= 0.6  # Also reduce distance cost
                        trajectory_cost *= 0.5  # Reduce trajectory penalty
                
                # CROWD HANDLING: Stricter matching when in crowd or swap protection active
                if track.swap_protection_active:
                    # In crowd: prioritize appearance over motion
                    total = 0.60*app_cost + 0.15*dist_cost + 0.10*hist_cost + 0.15*trajectory_cost
                elif track.locked:
                    total = (HIL_APP_WEIGHT*app_cost) + (HIL_MOTION_WEIGHT*dist_cost) + 0.1*hist_cost + 0.05*trajectory_cost
                else:
                    total = 0.45*app_cost + 0.2*dist_cost + 0.15*hist_cost + 0.1*trajectory_cost + 0.1*(1.0-iou(det, track.bbox))
                cost_matrix[d_idx, t_idx] = total

        # Hungarian
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_dets = set(); matched_tracks = set(); assignments = []; assigned_pids = set()
        
        for r, c in zip(row_ind, col_ind):
            pid = track_ids[c]
            track = self.tracks[pid]
            
            # Apply stricter threshold for tracks in crowds or with swap protection
            threshold = BASE_COSINE_THRESH
            if track.swap_protection_active:
                threshold = CROWD_MATCH_THRESHOLD  # Stricter for crowds
            
            if cost_matrix[r, c] < threshold:
                d_idx = r
                det_cent = centroid(detections[d_idx])
                
                # SWAP PREVENTION: If track has nearby tracks, verify this isn't a swap
                if track.swap_protection_active and track.nearby_track_ids:
                    # Check trajectory deviation for this match
                    deviation = track.get_trajectory_deviation(det_cent)
                    
                    # If large deviation, might be swapping - verify appearance more strictly
                    if deviation > TRAJECTORY_DEVIATION_THRESHOLD * 0.7:
                        feat_det = feats[d_idx]
                        track_feat = track.get_reid_feature()
                        if feat_det is not None and track_feat is not None:
                            app_dist = cosine_distance(feat_det, track_feat)
                            # Reject if appearance doesn't strongly match
                            if app_dist > CROWD_MATCH_THRESHOLD * 0.8:
                                continue  # Skip this match - likely a swap
                
                self.tracks[pid].update(detections[d_idx], frame_idx, feat=feats[d_idx], hist=det_hists[d_idx])
                assignments.append((pid, detections[d_idx]))
                matched_dets.add(d_idx); matched_tracks.add(pid); assigned_pids.add(pid)

        # Unmatched Detections -> Recover or Create
        for d_idx in range(len(detections)):
            if d_idx in matched_dets: continue
            feat = feats[d_idx]; hist = det_hists[d_idx]; bbox = detections[d_idx]

            # Check if this is a mid-area appearance (should be treated as new)
            is_mid_area = is_mid_area_appearance(bbox, self.frame_w, self.frame_h)
            
            if not is_mid_area:
                # 1. Try Active Edge Recovery (With Direction Check - ENTERING only)
                pid_act = self._edge_active_match(feat, bbox, hist, frame_idx)
                if pid_act is not None:
                    self.tracks[pid_act].update(bbox, frame_idx, feat=feat, hist=hist)
                    assignments.append((pid_act, bbox)); assigned_pids.add(pid_act); continue

                # 2. Try Deleted Resurrection (With Enhanced Occlusion Handling)
                pid_del = self._check_deleted_resurrect(feat, bbox, hist, frame_idx)
                if pid_del is not None:
                    # Clean from deleted
                    for idx, entry in enumerate(self.deleted_tracks):
                        # Check both positions where pid might be (index -2 for 7-tuple, -3 for 8-tuple)
                        entry_pid = entry[5] if len(entry) >= 6 else None
                        if entry_pid == pid_del:
                            del self.deleted_tracks[idx]
                            break
                    tr = Track(pid_del, bbox, frame_idx, feat=feat, hist=hist)
                    tr.is_confirmed = True; tr.locked = True
                    self.tracks[pid_del] = tr
                    assignments.append((pid_del, bbox)); assigned_pids.add(pid_del); continue

            # 3. Check nearby area for similar fish before creating new ID (prevent crowd swaps)
            det_cent = centroid(bbox)
            best_nearby_deleted = None
            best_nearby_score = 1e9
            
            for entry in reversed(self.deleted_tracks):
                if len(entry) >= 6:
                    del_feat, del_bbox, del_frame, del_hist, del_vel, del_pid = entry[:6]
                    age = frame_idx - del_frame
                    
                    if age < 30:  # Only consider recent deletions
                        del_cent = centroid(del_bbox)
                        dist = euclidean_distance(det_cent, del_cent)
                        
                        if dist < 150:  # In crowded area
                            if feat is not None and del_feat is not None:
                                app_dist = cosine_distance(feat, del_feat)
                                if app_dist < best_nearby_score:
                                    best_nearby_score = app_dist
                                    best_nearby_deleted = del_pid
            
            # If strong appearance match with recently deleted nearby fish, resurrect it
            if best_nearby_deleted is not None and best_nearby_score < CROWD_MATCH_THRESHOLD:
                # Remove from deleted
                for idx, entry in enumerate(self.deleted_tracks):
                    entry_pid = entry[5] if len(entry) >= 6 else None
                    if entry_pid == best_nearby_deleted:
                        del self.deleted_tracks[idx]
                        break
                
                if best_nearby_deleted in self.occlusion_zones:
                    del self.occlusion_zones[best_nearby_deleted]
                
                tr = Track(best_nearby_deleted, bbox, frame_idx, feat=feat, hist=hist)
                tr.is_confirmed = True
                tr.locked = True
                self.tracks[best_nearby_deleted] = tr
                assignments.append((best_nearby_deleted, bbox))
                assigned_pids.add(best_nearby_deleted)
                continue
            
            # 4. Check if in occlusion zone before creating new track
            in_occlusion_zone, occluded_pid = self._is_in_occlusion_zone(bbox, feat, frame_idx)
            
            if in_occlusion_zone and occluded_pid is not None:
                # This might be the occluded fish reappearing - resurrect it
                if occluded_pid in self.occlusion_zones:
                    del self.occlusion_zones[occluded_pid]
                
                # Remove from deleted if present
                for idx, entry in enumerate(self.deleted_tracks):
                    entry_pid = entry[5] if len(entry) >= 6 else None
                    if entry_pid == occluded_pid:
                        del self.deleted_tracks[idx]
                        break
                
                tr = Track(occluded_pid, bbox, frame_idx, feat=feat, hist=hist)
                tr.is_confirmed = True
                tr.locked = True
                self.tracks[occluded_pid] = tr
                assignments.append((occluded_pid, bbox))
                assigned_pids.add(occluded_pid)
                continue
            
            # 5. Create New (for mid-area or unmatched edge cases)
            pid = self.next_pid; self.next_pid += 1
            tr = Track(pid, bbox, frame_idx, feat=feat, hist=hist)
            self.tracks[pid] = tr
            assignments.append((pid, bbox)); assigned_pids.add(pid)

        # Cleanup - Enhanced for occlusion
        for pid in list(self.tracks.keys()):
            if pid not in assigned_pids:
                track = self.tracks[pid]
                track.increment_miss()
                
                # ENHANCED: Keep occluded tracks alive longer
                if pid in occluded_pids and track.occlusion_timer < OCCLUSION_BUFFER_FRAMES:
                    continue
                    
                if track.occluded and track.occlusion_timer < OSM_MAX_SURVIVAL:
                    continue
                    
                if track.miss_count > MAX_MISS_FRAMES:
                    mf = track.get_reid_feature()
                    hist = track.hist
                    vel = track.kf.get_velocity()
                    edge_zone = track.last_edge_zone
                    was_occluded = track.was_occluded or track.occluded
                    
                    # Store with occlusion info (8-tuple format)
                    self.deleted_tracks.append((mf, track.bbox.copy(), frame_idx, hist, vel, pid, edge_zone, was_occluded))
                    
                    # If occluded, store occlusion zone to prevent new IDs
                    if was_occluded and track.occlusion_position is not None and track.occlusion_feature is not None:
                        self.occlusion_zones[pid] = (track.occlusion_position, frame_idx, track.occlusion_feature)
                    
                    del self.tracks[pid]
        
        return assignments

    def get_active_count(self):
        return sum(1 for t in self.tracks.values() if t.miss_count < INACTIVE_AFTER_MISSES and t.is_confirmed)

# ---------------- MAIN RUN ----------------
def run(video_path, output_path, model_path, stats_output):
    print("Starting EDGE-FIXED Tracker (BEFORE Anti-ID-Swap)...")
    yolo = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))
    
    tracker = EnhancedDeepSORTTracker(DEVICE)
    tracker.set_frame_size(W,H)
    frame_idx = 0
    if os.path.exists(stats_output): os.remove(stats_output)
    start = time.time()
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        
        results = yolo(frame, conf=CONF_THRESHOLD, iou=NMS_IOU, verbose=False)[0]
        dets, feats = [], []
        if results.boxes:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            valid_indices = [i for i, (box, conf) in enumerate(zip(boxes, confs)) 
                             if conf > CONF_THRESHOLD and (box[2]-box[0])*(box[3]-box[1]) > 60]
            if valid_indices:
                bboxes = boxes[valid_indices].tolist()
                raw_feats = tracker.reid.extract_batch(frame, bboxes)
                for bb, f in zip(bboxes, raw_feats):
                    dets.append(np.array(bb)); feats.append(f)
                    
        assigns = tracker.match(dets, feats, frame_idx, (W,H), frame_img=frame)
        
        # Update count
        tracker.update_count()

        # Draw with color coding
        for pid, box in assigns:
            x1,y1,x2,y2 = map(int, box)
            tr = tracker.tracks[pid]
            
            # Color coding: Yellow (new) -> Green (confirmed/old)
            if not tr.is_confirmed:
                color = (0,255,255)  # Yellow for newly detected fish
            else:
                color = (0,255,0)  # Green for confirmed and old fish
            
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            
            label = f"ID {pid}"
            cv2.putText(frame, label, (x1, max(12,y1-5)), 0, 0.6, (255,255,255), 2)

        count = tracker.total_unique_fishes
        active = tracker.get_active_count()
        fps_now = frame_idx / (time.time() - start + 1e-9)
        
        cv2.putText(frame, f"Active: {active} | Total Confirmed: {count}", (10,30), 0, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"Frame: {frame_idx} | FPS: {fps_now:.1f}", (10,60), 0, 0.6, (255,255,255), 2)
        out.write(frame)
        
        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx} | Active: {active} | Total: {count} | FPS: {fps_now:.1f}")
            with open(stats_output, "a") as f:
                f.write(f"{frame_idx},{active},{count},{fps_now:.2f}\n")

    cap.release(); out.release()
    print(f"\n{'='*50}")
    print(f"Processing Complete!")
    print(f"Total Unique Fish Counted: {tracker.total_unique_fishes}")
    print(f"Output saved to: {output_path}")
    print(f"Stats saved to: {stats_output}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    run(VIDEO_PATH, OUTPUT_PATH, MODEL_PATH, STATS_OUTPUT)










