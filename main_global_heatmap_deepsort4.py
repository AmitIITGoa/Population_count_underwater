#!/usr/bin/env python3
"""
main_global_heatmap_deepsort4.py

Quality Restoration Version (v4).
Based on v2 (High Quality) but with GPU ReID (Safe Speedup).
Reverts the "bad" optimizations from v3 (Frame Skipping, Downscaling).

Improvements:
 1. GPU ReID: ResNet18 on CUDA (Fast & Accurate).
 2. VECTORIZED RECOVERY: Kept from v2.
 3. FULL RESOLUTION: Kept from v2 (No downscaling).
 4. NO FRAME SKIPPING: Smooth tracking.
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
VIDEO_PATH = "output_video2_enhanced_trim.mp4"
OUTPUT_PATH = "output_video2_enhanced_trim_v4.mp4"
HEATMAP_VIDEO_PATH = "output_heatmap_only_v4.mp4"
STATS_OUTPUT = "population_statistics_v4.txt"
MODEL_PATH = "best.pt"

# --- SPEED CONTROL ---
PLAYBACK_SPEED = 1.0 

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# OPTIMIZATION: Use GPU for ReID (Safe speedup)
REID_DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

# MATCHING WEIGHTS
CONF_THRESHOLD = 0.35      
NMS_IOU = 0.50
IOU_STICKY_THRESHOLD = 0.5 
REID_VETO_THRESHOLD = 0.30 
BASE_COSINE_THRESH = 0.30  

# PHYSICS GATES
MAX_MOTION_DISTANCE = 200.0 
MAX_PIXEL_SPEED = 40.0      

# QUALITY GATES
SMART_UPDATE_THRESH = 0.70 
CROWD_RADIUS = 80.0 

# Timeouts
MAX_MISS_FRAMES = 300      
MAX_GAP_FRAMES = 300
DEDUP_THRESHOLD = 0.10
KF_DT = 1.0

# Visuals
HEATMAP_INTENSITY = 0.7 
HEATMAP_DECAY = False

# System
MEMORY_BANK_SIZE = 120     
LONG_TERM_SIZE = 120
INACTIVE_AFTER_MISSES = 60
EMA_ALPHA = 0.9            
MULTICROP = True
CROP_SCALES = [1.0]
HIST_BINS = 32
HIST_COMP_METHOD = cv2.HISTCMP_CORREL
N_CONFIRM = 3
EDGE_MARGIN = 40
RESURRECT_AGE_FRAMES = 300 
HIL_LOCK_HITS = 5
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

def get_position_label(bbox, W, H, margin=EDGE_MARGIN):
    if W is None or H is None: return 'Inside'
    x1, y1, x2, y2 = map(float, bbox)
    if x1 < margin: return 'Left'
    if x2 > (W - margin): return 'Right'
    if y1 < margin: return 'Top'
    if y2 > (H - margin): return 'Bottom'
    return 'Inside'

def are_opposite_sides(side1, side2):
    if side1 == 'Inside' or side2 == 'Inside': return False
    if (side1 == 'Left' and side2 == 'Right') or (side1 == 'Right' and side2 == 'Left'): return True
    if (side1 == 'Top' and side2 == 'Bottom') or (side1 == 'Bottom' and side2 == 'Top'): return True
    return False

def motion_distance_gate(det_centroid, track_centroid, track_velocity, max_distance=140.0):
    predicted_pos = np.array(track_centroid) + track_velocity * 2.0
    dist = euclidean_distance(det_centroid, predicted_pos)
    if dist > max_distance: return 1.0
    return min(dist / max_distance, 1.0)

# ---------------- HEATMAP (OPTIMIZED) ----------------
class MotionHeatmap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.mask = np.zeros((height, width), dtype=np.float32)

    def update(self, tracks):
        if HEATMAP_DECAY: self.mask *= 0.98 
        for t in tracks:
            if not t.is_confirmed: continue
            cx, cy, _, _ = xywh_from_xyxy(t.bbox)
            cv2.circle(self.mask, (int(cx), int(cy)), radius=8, color=(5.0), thickness=-1)

    def get_score(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.mask[int(y), int(x)]
        return 0.0

    def get_heatmap_image(self):
        display_mask = np.clip(self.mask * 4, 0, 255).astype(np.uint8)
        display_mask = cv2.GaussianBlur(display_mask, (15, 15), 0)
        heatmap_img = cv2.applyColorMap(display_mask, cv2.COLORMAP_JET)
        return heatmap_img

    def apply_overlay(self, frame):
        heatmap_img = self.get_heatmap_image()
        overlay = cv2.addWeighted(frame, 1.0, heatmap_img, HEATMAP_INTENSITY, 0)
        return overlay

# ---------------- NSA KALMAN FILTER ----------------
class NSAKalmanFilter:
    def __init__(self, cx: float, cy: float, dt: float = KF_DT):
        self.dt = dt
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 10.0 
        self.Q = np.eye(4) * 0.5 
        self.Q[2,2] *= 4.0; self.Q[3,3] *= 4.0
        self.R_base = np.eye(2) * 5.0
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)

    def predict(self, dt: float = 1.0) -> np.ndarray:
        self.dt = dt
        self.F[0,2]=dt; self.F[1,3]=dt
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.x

    def update(self, cx: float, cy: float, conf: float = 1.0):
        z = np.array([cx, cy], dtype=float)
        prediction = self.H.dot(self.x)
        innovation = z - prediction
        dist_innov = np.linalg.norm(innovation)
        if dist_innov > 20.0:
            self.P += np.eye(4) * (dist_innov / 5.0)

        safe_conf = max(0.1, min(0.99, conf))
        scale_factor = (1.0 - safe_conf) * 10.0 
        R_adaptive = self.R_base * max(1.0, scale_factor)
        S = self.H.dot(self.P).dot(self.H.T) + R_adaptive
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(z - self.H.dot(self.x))
        self.P = (np.eye(len(self.x)) - K.dot(self.H)).dot(self.P)
    
    def get_velocity(self): return self.x[2:4]
    def get_future_trajectory(self, steps=10):
        future_pts = []
        temp_x = self.x.copy()
        temp_F = self.F.copy()
        for _ in range(steps):
            temp_x = temp_F.dot(temp_x)
            future_pts.append((int(temp_x[0]), int(temp_x[1])))
        return future_pts

# ---------------- ReID model (GPU OPTIMIZED) ----------------
class ReIDModel:
    def __init__(self, device: str = REID_DEVICE):
        self.device = REID_DEVICE
        self.input_size = (224, 224)
        # ResNet18 on GPU
        backbone = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
        self.model.eval()
        self.norm = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    
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

    def extract_batch(self, frame_bgr: np.ndarray, bboxes: List[List[float]]) -> List[Optional[np.ndarray]]:
        all_features = [None]*len(bboxes)
        imgs = []; idx_map=[]
        for i,bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            h_img, w_img = frame_bgr.shape[:2]
            cx = (x1+x2)/2; cy = (y1+y2)/2
            w_box = max(1, x2-x1); h_box = max(1, y2-y1)
            crops_pil = []
            scales = CROP_SCALES if MULTICROP else [1.0]
            for scale in scales:
                nw = int(w_box*scale); nh = int(h_box*scale)
                nx1 = int(cx - nw/2); ny1 = int(cy - nh/2)
                nx2 = nx1 + nw; ny2 = ny1 + nh
                nx1, ny1 = max(0, nx1), max(0, ny1)
                nx2, ny2 = min(w_img, nx2), min(h_img, ny2)
                roi = frame_bgr[ny1:ny2, nx1:nx2]
                if roi.size > 0: crops_pil.append(self._letterbox(roi))
            for pil in crops_pil:
                imgs.append(self.norm(pil)); idx_map.append(i)
                imgs.append(self.norm(pil.transpose(Image.FLIP_LEFT_RIGHT))); idx_map.append(i)
        if not imgs: return [None]*len(bboxes)
        
        # GPU BATCH PROCESSING
        batch = torch.stack(imgs, dim=0).to(self.device)
        with torch.no_grad():
            feats = self.model(batch).reshape(len(batch), -1).cpu().numpy()
            
        per_idx = defaultdict(list)
        for idx, f in zip(idx_map, feats):
            per_idx[idx].append(f)
        for i in range(len(bboxes)):
            if i in per_idx:
                arr = np.stack(per_idx[i], axis=0)
                all_features[i] = L2_normalize(np.mean(arr, axis=0))
        return all_features

# ---------------- Memory and Track ----------------
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
    def best_match_score(self, query_feat):
        if not self.long_term or query_feat is None: return 1.0
        dists = cdist([query_feat], list(self.long_term), metric='cosine')[0]
        return dists.min() 

class Track:
    def __init__(self, pid, bbox, frame_idx, feat=None, hist=None, conf=0.0):
        self.pid = pid
        self.bbox = np.array(bbox, dtype=float)
        self.kf = NSAKalmanFilter(*centroid(bbox))
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
        self.is_crowded = False
        self.velocity_ema = np.array([0.0, 0.0])
        self.heatmap_score = 0.0

    def predict(self): self.kf.predict()
    
    def update(self, bbox, frame_idx, feat=None, hist=None, freeze_feature=False, conf=1.0, heatmap_score=0.0):
        self.bbox = np.array(bbox, dtype=float)
        cx, cy, _, _ = xywh_from_xyxy(bbox)
        self.kf.update(cx, cy, conf=conf)
        raw_vel = self.kf.get_velocity()
        self.velocity_ema = 0.6 * self.velocity_ema + 0.4 * raw_vel
        self.last_seen = frame_idx
        self.hits += 1
        self.consecutive_hits += 1
        self.miss_count = 0
        self.heatmap_score = heatmap_score
        
        is_good_quality = (conf > SMART_UPDATE_THRESH)
        if feat is not None and not freeze_feature and is_good_quality:
            if self.smoothed_feat is None: self.smoothed_feat = feat.copy()
            else: self.smoothed_feat = EMA_ALPHA * self.smoothed_feat + (1.0-EMA_ALPHA) * feat
            self.short_term.append(feat)
            self.memory.add(self.smoothed_feat)
            
        if hist is not None:
            if self.hist is None: self.hist = hist
            else:
                self.hist = 0.8 * self.hist + 0.2 * hist
                cv2.normalize(self.hist, self.hist)

        self.confirm_count += 1
        if self.confirm_count >= N_CONFIRM: self.is_confirmed = True
        if not self.occluded and self.consecutive_hits >= HIL_LOCK_HITS: self.locked = True
        self.occluded = False; self.occlusion_timer = 0

    def increment_miss(self):
        self.miss_count += 1; self.consecutive_hits = 0
        self.occlusion_timer += 1
        if self.occlusion_timer > 0: self.occluded = True
    def get_reid_feature(self):
        if self.smoothed_feat is not None: return L2_normalize(self.smoothed_feat)
        return self.memory.mean_long_term()
    def get_prediction_points(self):
        return self.kf.get_future_trajectory(steps=10)

# ---------------- TRACKER (VECTORIZED) ----------------
class EnhancedDeepSORTTracker:
    def __init__(self, device="cuda"):
        self.tracks: Dict[int, Track] = {}
        self.next_pid = 0
        self.reid = ReIDModel(REID_DEVICE)
        self.total_unique_fishes = 0
        self.deleted_tracks_data = [] 
        self.frame_w = None; self.frame_h = None

    def set_frame_size(self, w:int, h:int):
        self.frame_w = w; self.frame_h = h

    def _recover_id_with_position_logic(self, feat, bbox, hist, current_frame_idx, heatmap=None):
        if feat is None: return None, None
        det_cent = centroid(bbox)
        new_side = get_position_label(bbox, self.frame_w, self.frame_h, EDGE_MARGIN)
        required_similarity = 0.60 if (new_side != 'Inside') else 0.35 
        best_pid = None; best_sim = -1.0; source_type = None 

        for pid, tr in self.tracks.items():
            if (current_frame_idx - tr.last_seen) > INACTIVE_AFTER_MISSES: continue
            if tr.get_reid_feature() is None: continue
            
            old_side = get_position_label(tr.bbox, self.frame_w, self.frame_h, EDGE_MARGIN)
            if are_opposite_sides(old_side, new_side): continue
            
            cand_cent = centroid(tr.bbox)
            dist_px = euclidean_distance(det_cent, cand_cent)
            time_gap = current_frame_idx - tr.last_seen
            max_allowed_dist = max(100.0, float(time_gap) * MAX_PIXEL_SPEED)
            
            is_proximity_boost = False
            if old_side == 'Inside' and new_side == 'Inside':
                if dist_px < 250.0: is_proximity_boost = True 
            if not is_proximity_boost and dist_px > max_allowed_dist: continue

            d = tr.memory.best_match_score(feat)
            sim = 1.0 - (2.0 * d) 
            if is_proximity_boost: sim += 0.2 
            if heatmap is not None:
                cx, cy = centroid(tr.bbox)
                score = heatmap.get_score(cx, cy)
                if score > 10.0: sim += 0.15
            
            if sim > best_sim: best_sim = sim; best_pid = pid; source_type = 'active'

        if self.deleted_tracks_data:
            del_feats = []
            del_bboxes = []
            del_frames = []
            del_hists = []
            del_pids = []
            
            for i, item in enumerate(self.deleted_tracks_data):
                age = current_frame_idx - item[2]
                if age <= RESURRECT_AGE_FRAMES:
                    del_feats.append(item[0])
                    del_bboxes.append(item[1])
                    del_frames.append(item[2])
                    del_hists.append(item[3])
                    del_pids.append(item[5])
            
            if del_feats:
                del_feats = np.array(del_feats)
                del_frames = np.array(del_frames)
                del_pids = np.array(del_pids)
                
                del_cx = (np.array([b[0] for b in del_bboxes]) + np.array([b[2] for b in del_bboxes])) / 2.0
                del_cy = (np.array([b[1] for b in del_bboxes]) + np.array([b[3] for b in del_bboxes])) / 2.0
                del_cents = np.stack([del_cx, del_cy], axis=1)
                
                dists_px = np.linalg.norm(del_cents - np.array(det_cent), axis=1)
                time_gaps = current_frame_idx - del_frames
                max_allowed = np.maximum(100.0, time_gaps * MAX_PIXEL_SPEED)
                valid_spatial = dists_px <= max_allowed
                
                if np.any(valid_spatial):
                    cand_feats = del_feats[valid_spatial]
                    cand_hists = [del_hists[i] for i, v in enumerate(valid_spatial) if v]
                    cand_pids = del_pids[valid_spatial]
                    
                    dists_cos = cdist([feat], cand_feats, metric='cosine')[0]
                    sims = 1.0 - (2.0 * dists_cos)
                    
                    if hist is not None:
                        for idx, h_cand in enumerate(cand_hists):
                            if h_cand is not None:
                                hs = hist_similarity(hist, h_cand)
                                if hs > 0.85: sims[idx] += 0.15
                                elif hs > 0.65: sims[idx] += 0.08
                    
                    best_idx = np.argmax(sims)
                    best_val = sims[best_idx]
                    if best_val > best_sim:
                        best_sim = best_val
                        best_pid = cand_pids[best_idx]
                        source_type = 'deleted'

        if best_sim > required_similarity: return best_pid, source_type
        return None, None

    def match(self, detections, feats, frame_idx, confs, frame_wh: Tuple[int,int]=None, frame_img=None, heatmap=None):
        if frame_wh is not None: self.set_frame_size(*frame_wh)
        det_hists=[compute_histogram_rgb(frame_img, d) if frame_img is not None else None for d in detections]
        for t in self.tracks.values(): t.predict()
        track_ids=list(self.tracks.keys())
        confirmed_ids = [pid for pid in track_ids if self.tracks[pid].is_confirmed and self.tracks[pid].miss_count < 10]
        unconfirmed_ids = [pid for pid in track_ids if pid not in confirmed_ids]
        matched_dets=set(); matched_tracks=set(); matches=[]

        if len(track_ids) > 1:
            track_centroids = [centroid(self.tracks[pid].bbox) for pid in track_ids]
            dist_mat = cdist(track_centroids, track_centroids)
            np.fill_diagonal(dist_mat, 9999.0)
            min_dists = dist_mat.min(axis=1)
            for i, d in enumerate(min_dists):
                if d < CROWD_RADIUS: self.tracks[track_ids[i]].is_crowded = True
                else: self.tracks[track_ids[i]].is_crowded = False
        
        if heatmap is not None:
            for pid in track_ids:
                tr = self.tracks[pid]
                cx, cy, _, _ = xywh_from_xyxy(tr.bbox)
                if heatmap.get_score(cx, cy) > 10.0: tr.is_crowded = True

        # --- CASCADE STAGE 1 ---
        if confirmed_ids:
            cost_matrix = np.ones((len(detections), len(confirmed_ids)), dtype=float) * 1e6
            for d_idx, det in enumerate(detections):
                for t_idx, pid in enumerate(confirmed_ids):
                    track = self.tracks[pid]
                    dist_feat = track.memory.best_match_score(feats[d_idx])
                    dist_hist = 0.5
                    if track.hist is not None and det_hists[d_idx] is not None:
                        sim_h = hist_similarity(track.hist, det_hists[d_idx])
                        dist_hist = 0.5 * (1.0 - sim_h)
                    combined_appearance = 0.6 * dist_feat + 0.4 * dist_hist
                    
                    val_iou = iou(det, track.bbox)
                    tr_cent = centroid(track.bbox); det_cent = centroid(det)
                    
                    time_gap = frame_idx - track.last_seen
                    allowed_dist = max(50.0, float(time_gap) * MAX_PIXEL_SPEED)
                    if euclidean_distance(det_cent, tr_cent) > allowed_dist: 
                        cost_matrix[d_idx, t_idx] = 1e6; continue

                    dist_move = motion_distance_gate(det_cent, tr_cent, track.kf.get_velocity())
                    
                    dir_cost = 0.5
                    vel_smooth = track.velocity_ema 
                    speed = np.linalg.norm(vel_smooth)
                    if speed > 1.0:
                        disp = np.array(det_cent) - np.array(tr_cent)
                        if np.linalg.norm(disp) > 1e-3:
                            v_norm = vel_smooth / speed
                            d_norm = disp / np.linalg.norm(disp)
                            dir_score = np.dot(v_norm, d_norm)
                            if dir_score < -0.5: dir_cost = 0.8 
                            else: dir_cost = (1.0 - dir_score) / 2.0

                    sim_feat = 1.0 - (2.0 * dist_feat)
                    if val_iou > IOU_STICKY_THRESHOLD and sim_feat > REID_VETO_THRESHOLD:
                        cost_matrix[d_idx, t_idx] = 0.001; continue

                    if track.is_crowded:
                        combined_cost = 0.3*combined_appearance + 0.4*dir_cost + 0.3*dist_move
                    else:
                        combined_cost = 0.5*combined_appearance + 0.3*dist_move + 0.2*(1.0-val_iou)
                    cost_matrix[d_idx, t_idx] = combined_cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r,c in zip(row_ind, col_ind):
                if cost_matrix[r,c] < BASE_COSINE_THRESH or cost_matrix[r,c] < 0.01:
                    matches.append((r, confirmed_ids[c]))
                    matched_dets.add(r); matched_tracks.add(confirmed_ids[c])

        # --- CASCADE STAGE 2 ---
        if unconfirmed_ids:
            cost_matrix = np.ones((len(detections), len(unconfirmed_ids)), dtype=float) * 1e6
            for d_idx, det in enumerate(detections):
                if d_idx in matched_dets: continue
                for t_idx, pid in enumerate(unconfirmed_ids):
                    track = self.tracks[pid]
                    if (frame_idx - track.last_seen > MAX_GAP_FRAMES): continue
                    dist_feat = track.memory.best_match_score(feats[d_idx])
                    dist_hist = 0.5
                    if track.hist is not None and det_hists[d_idx] is not None:
                        sim_h = hist_similarity(track.hist, det_hists[d_idx])
                        dist_hist = 0.5 * (1.0 - sim_h)
                    combined_appearance = 0.6 * dist_feat + 0.4 * dist_hist
                    tr_cent = centroid(track.bbox); det_cent = centroid(det)
                    time_gap = frame_idx - track.last_seen
                    dynamic_max_dist = MAX_MOTION_DISTANCE + (float(time_gap) * 10.0)
                    dist_move = motion_distance_gate(det_cent, tr_cent, track.kf.get_velocity(), max_distance=dynamic_max_dist)
                    combined_cost = 0.7*combined_appearance + 0.3*dist_move
                    cost_matrix[d_idx, t_idx] = combined_cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r,c in zip(row_ind, col_ind):
                if cost_matrix[r,c] < BASE_COSINE_THRESH:
                    matches.append((r, unconfirmed_ids[c]))
                    matched_dets.add(r); matched_tracks.add(unconfirmed_ids[c])

        # UPDATE
        assignments=[]; assigned_pids=set()
        for d_idx, pid in matches:
            tr = self.tracks[pid]
            h_score = 0.0
            if heatmap is not None:
                cx, cy = centroid(detections[d_idx])
                h_score = heatmap.get_score(cx, cy)
            tr.update(detections[d_idx], frame_idx, feat=feats[d_idx], hist=det_hists[d_idx], 
                      freeze_feature=tr.is_crowded, conf=confs[d_idx], heatmap_score=h_score)
            assignments.append((pid, detections[d_idx])); assigned_pids.add(pid)

        # RECOVERY
        for d_idx in range(len(detections)):
            if d_idx in matched_dets: continue
            recovered_pid, source_type = self._recover_id_with_position_logic(
                feats[d_idx], detections[d_idx], det_hists[d_idx], frame_idx, heatmap
            )
            if recovered_pid is not None and recovered_pid in assigned_pids: recovered_pid = None
            h_score = 0.0
            if heatmap is not None:
                cx, cy = centroid(detections[d_idx])
                h_score = heatmap.get_score(cx, cy)

            if recovered_pid is not None:
                if source_type == 'deleted':
                    for idx, entry in enumerate(self.deleted_tracks_data):
                        if entry[-1] == recovered_pid: 
                            del self.deleted_tracks_data[idx]; break
                            
                    tr = Track(recovered_pid, detections[d_idx], frame_idx, feat=feats[d_idx], hist=det_hists[d_idx], conf=confs[d_idx])
                    tr.is_confirmed = True; tr.locked = True
                    tr.heatmap_score = h_score
                    self.tracks[recovered_pid] = tr
                else:
                    self.tracks[recovered_pid].update(detections[d_idx], frame_idx, feat=feats[d_idx], hist=det_hists[d_idx], conf=confs[d_idx], heatmap_score=h_score)
                assignments.append((recovered_pid, detections[d_idx])); assigned_pids.add(recovered_pid)
            else:
                pid = self.next_pid; self.next_pid += 1
                tr = Track(pid, detections[d_idx], frame_idx, feat=feats[d_idx], hist=det_hists[d_idx], conf=confs[d_idx])
                tr.heatmap_score = h_score
                self.tracks[pid] = tr
                assignments.append((pid, detections[d_idx])); assigned_pids.add(pid); self.total_unique_fishes += 1

        # CLEANUP
        for pid in list(self.tracks.keys()):
            if pid not in assigned_pids:
                track = self.tracks[pid]; track.increment_miss()
                threshold = MAX_MISS_FRAMES
                if track.heatmap_score > 20.0: threshold = int(MAX_MISS_FRAMES * 1.5)
                if track.occluded and track.occlusion_timer < 120: continue
                if track.miss_count > threshold:
                    mf = track.get_reid_feature(); hist = track.hist
                    vel = track.kf.get_velocity()
                    self.deleted_tracks_data.append((mf, track.bbox.copy(), frame_idx, hist, vel, pid))
                    if len(self.deleted_tracks_data) > 1500:
                        self.deleted_tracks_data.pop(0)
                    del self.tracks[pid]
        return assignments

    def get_active_count(self):
        return sum(1 for t in self.tracks.values() if t.miss_count < INACTIVE_AFTER_MISSES)

# ---------------- MAIN RUN ----------------
def run(video_path, output_path, model_path, stats_output):
    print(f"Starting QUALITY RESTORED DeepSORT v4 (GPU + No Skip + Full Res)...")
    yolo = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    W, H = int(cap.get(3)), int(cap.get(4))
    
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps < 1 or source_fps > 120: source_fps = 24.0
    output_fps = source_fps * PLAYBACK_SPEED
    print(f"Input FPS: {source_fps}. Speed Factor: {PLAYBACK_SPEED}. Output FPS: {output_fps}")

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (W,H))
    out_heatmap = cv2.VideoWriter(HEATMAP_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (W,H))
    
    tracker = EnhancedDeepSORTTracker(DEVICE)
    tracker.set_frame_size(W,H)
    heatmap = MotionHeatmap(W, H)
    
    frame_idx = 0
    if os.path.exists(stats_output): os.remove(stats_output)
    start = time.time()
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        results = yolo(frame, conf=CONF_THRESHOLD, iou=NMS_IOU, verbose=False)[0]
        dets, feats, confs = [], [], []
        if results.boxes:
            boxes = results.boxes.xyxy.cpu().numpy()
            c_scores = results.boxes.conf.cpu().numpy()
            valid_indices=[]
            for i,(box,conf) in enumerate(zip(boxes, c_scores)):
                w = box[2] - box[0]; h = box[3] - box[1]
                if conf > CONF_THRESHOLD and (w*h) > 60: valid_indices.append(i)
            if valid_indices:
                bboxes = boxes[valid_indices].tolist()
                valid_confs = c_scores[valid_indices].tolist()
                raw_feats = tracker.reid.extract_batch(frame, bboxes)
                for bb, f, c in zip(bboxes, raw_feats, valid_confs):
                    dets.append(np.array(bb)); feats.append(f); confs.append(c)
        
        assigns = tracker.match(dets, feats, frame_idx, confs, (W,H), frame_img=frame, heatmap=heatmap)
        
        heatmap.update(tracker.tracks.values())
        
        # Write pure heatmap video
        heatmap_only_frame = heatmap.get_heatmap_image()
        out_heatmap.write(heatmap_only_frame)
        
        # frame = heatmap.apply_overlay(frame)
        
        # DRAW CLEAN VISUALIZATION
        for pid, box in assigns:
            x1,y1,x2,y2 = map(int, box)
            color = (0,255,0) if tracker.tracks[pid].is_confirmed else (0,165,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"ID {pid}"
            cv2.putText(frame, label, (x1, max(12,y1-5)), 0, 0.6, (255,255,255), 2)
            # cv2.putText(frame, label, (x1, y1 - 5), 0, 0.6, (0,0,0), 3)
            # cv2.putText(frame, label, (x1, y1 - 5), 0, 0.6, (255,255,255), 1)
        
        count = tracker.get_active_count()
        total = tracker.total_unique_fishes
        
        text = f"Active: {count} | Total Unique: {total}"
        cv2.putText(frame, text, (13, 30), 0, 0.8, (0,0,0), 4)
        cv2.putText(frame, text, (13, 30), 0, 0.8, (0, 255, 255), 2)

        proc_fps = frame_idx / (time.time() - start + 1e-9)
        out.write(frame)
        
        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx} | Active: {count} | ProcFPS: {proc_fps:.1f}")
            with open(stats_output, "a") as f:
                f.write(f"{frame_idx},{count},{total}\n")
    cap.release(); out.release(); out_heatmap.release()
    print(f"Done. Heatmap saved to {HEATMAP_VIDEO_PATH}")

if __name__ == "__main__":
    run(VIDEO_PATH, OUTPUT_PATH, MODEL_PATH, STATS_OUTPUT)
