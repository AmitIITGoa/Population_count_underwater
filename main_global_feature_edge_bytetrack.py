#!/usr/bin/env python3
"""
main_global_feature_edge_bytetrack_cpu_improved_v7_FIXED.py

Updates:
 1. ALGORITHM SWAP: DeepSORT -> ByteTrack (High/Low Conf split).
 2. OCCLUSION RECOVERY: Specific logic for tracks lost "Inside" the frame.
 3. PROXIMITY BOOST: If a new detection is close to where a track disappeared, boost score.
 4. DUPLICATE ID FIX: Strictly prevents same ID from being assigned to multiple detections.
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
OUTPUT_PATH = "output_video2_bytetrack_fixed_v3.mp4"
STATS_OUTPUT = "population_statistics_v7.txt"
MODEL_PATH = "best.pt"

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REID_DEVICE = "cpu"

# --- TUNING ---
FORCED_FPS = 20.0 
CONF_THRESHOLD = 0.45 # High confidence threshold for ByteTrack
LOW_CONF_THRESHOLD = 0.1 # Low confidence threshold for ByteTrack
NMS_IOU = 0.45

# Sticky Tracking
IOU_STICKY_THRESHOLD = 0.5 
REID_VETO_THRESHOLD = 0.2

# Standard Matching
BASE_COSINE_THRESH = 0.40
IOU_THRESHOLD = 0.30

# Edge vs Inside Logic
EDGE_MARGIN = 40 
EDGE_SIMILARITY_THRESHOLD = 0.65 
INSIDE_SIMILARITY_THRESHOLD = 0.40 # Baseline for inside

# OCCLUSION SETTINGS
PROXIMITY_RADIUS = 150.0 
PROXIMITY_SCORE_BOOST = 0.15 

# Timeouts
MAX_MISS_FRAMES = 150   # Remember fish for ~7 seconds if lost
MAX_GAP_FRAMES = 150
DEDUP_THRESHOLD = 0.15
KF_DT = 1.0
MAX_PIXEL_SPEED = 30.0 

MEMORY_BANK_SIZE = 120
LONG_TERM_SIZE = 120
INACTIVE_AFTER_MISSES = 45
EMA_ALPHA = 0.8
MULTICROP = True
CROP_SCALES = [1.0, 0.9]
HIST_BINS = 32
HIST_COMP_METHOD = cv2.HISTCMP_CORREL
N_CONFIRM = 3
DENSITY_LOW = 3
DENSITY_HIGH = 10
RESURRECT_AGE_FRAMES = 150 

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

def motion_distance_gate(det_centroid, track_centroid, track_velocity, max_distance=140.0):
    predicted_pos = np.array(track_centroid) + track_velocity * 2.0
    dist = euclidean_distance(det_centroid, predicted_pos)
    if dist > max_distance: return 1.0
    return min(dist / max_distance, 1.0)

# ---------------- POSITION HELPERS ----------------
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

# ---------------- KALMAN FILTER ----------------
class KalmanFilterCV:
    def __init__(self, cx: float, cy: float, dt: float = KF_DT):
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
    def predict(self): self.kf.predict()
    def update(self, bbox, frame_idx, feat=None, hist=None):
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

# ---------------- TRACKER (BYTETrack Version) ----------------
class EnhancedByteTracker:
    def __init__(self, device="cuda"):
        self.tracks: Dict[int, Track] = {}
        self.next_pid = 0
        self.reid = ReIDModel(REID_DEVICE)
        self.total_unique_fishes = 0
        self.deleted_tracks = deque(maxlen=1500)
        self.frame_w = None; self.frame_h = None

    def set_frame_size(self, w:int, h:int):
        self.frame_w = w; self.frame_h = h

    def _rbm_fuse(self, det_feat, track: Track):
        ema = track.smoothed_feat
        mem = track.memory.mean_long_term()
        parts = []; weights = []
        if ema is not None: parts.append(ema); weights.append(0.6)
        if mem is not None: parts.append(mem); weights.append(0.4)
        if not parts: return det_feat
        wsum = sum(weights)
        agg = sum(w * p for w,p in zip(weights, parts)) / (wsum + 1e-12)
        return L2_normalize(agg)

    def _recover_id_with_position_logic(self, feat, bbox, hist, current_frame_idx):
        """
        Enhanced Recovery with Occlusion Proximity Boost.
        """
        if feat is None: return None, None
        det_cent = centroid(bbox)
        new_side = get_position_label(bbox, self.frame_w, self.frame_h, EDGE_MARGIN)
        is_edge = (new_side != 'Inside')
        
        # Default thresholds
        required_similarity = EDGE_SIMILARITY_THRESHOLD if is_edge else INSIDE_SIMILARITY_THRESHOLD
        
        best_pid = None; best_sim = -1.0; source_type = None 

        def check_candidate(candidate_feat, candidate_bbox, candidate_last_frame):
            old_side = get_position_label(candidate_bbox, self.frame_w, self.frame_h, EDGE_MARGIN)
            
            # 1. Strict Teleport Check for Edges
            if are_opposite_sides(old_side, new_side): return -1.0
            
            # 2. Proximity / Speed Check
            cand_cent = centroid(candidate_bbox)
            dist_px = euclidean_distance(det_cent, cand_cent)
            time_gap = current_frame_idx - candidate_last_frame
            
            # Calculate allowed distance
            max_allowed_dist = max(150.0, float(time_gap) * MAX_PIXEL_SPEED)
            
            # --- OCCLUSION LOGIC START ---
            is_proximity_boost = False
            if old_side == 'Inside' and new_side == 'Inside':
                if dist_px < PROXIMITY_RADIUS:
                    is_proximity_boost = True
            
            if not is_proximity_boost and dist_px > max_allowed_dist:
                return -1.0 
            # --- OCCLUSION LOGIC END ---

            # 3. Similarity Calculation
            d = cosine_distance(feat, candidate_feat)
            sim = 1.0 - (2.0 * d)

            # Apply Boosts
            if is_proximity_boost:
                sim += PROXIMITY_SCORE_BOOST 
            
            return sim

        # Check Active (missed by sticky)
        for pid, tr in self.tracks.items():
            if (current_frame_idx - tr.last_seen) > INACTIVE_AFTER_MISSES: continue
            if tr.get_reid_feature() is None: continue
            sim = check_candidate(tr.get_reid_feature(), tr.bbox, tr.last_seen)
            if sim > best_sim:
                best_sim = sim; best_pid = pid; source_type = 'active'

        # Check Deleted (Resurrection)
        if self.deleted_tracks:
            for del_feat, del_bbox, del_frame, del_hist, _, del_pid in reversed(self.deleted_tracks):
                age = current_frame_idx - del_frame
                if age > RESURRECT_AGE_FRAMES: continue
                sim = check_candidate(del_feat, del_bbox, del_frame)
                
                if sim > 0:
                    hs = hist_similarity(hist, del_hist) if (hist is not None and del_hist is not None) else 0.0
                    if hs > 0.8: sim += 0.05
                
                if sim > best_sim:
                    best_sim = sim; best_pid = del_pid; source_type = 'deleted'

        if best_sim > required_similarity:
            return best_pid, source_type
        return None, None

    def match(self, detections, feats, confs, frame_idx, frame_wh: Tuple[int,int]=None, frame_img=None):
        if frame_wh is not None: self.set_frame_size(*frame_wh)
        det_hists=[compute_histogram_rgb(frame_img, d) if frame_img is not None else None for d in detections]
        
        for t in self.tracks.values(): t.predict()
        
        # --- BYTETrack Classification ---
        # Split detections into High and Low confidence
        det_high_indices = []
        det_low_indices = []
        
        for i, conf in enumerate(confs):
            if conf >= CONF_THRESHOLD:
                det_high_indices.append(i)
            else:
                det_low_indices.append(i)

        # Split tracks
        track_ids = list(self.tracks.keys())
        confirmed_tracks = [pid for pid in track_ids if self.tracks[pid].is_confirmed]
        unconfirmed_tracks = [pid for pid in track_ids if not self.tracks[pid].is_confirmed]

        assignments=[]
        assigned_pids=set()
        matched_high_indices = set()
        matched_low_indices = set()

        # --- STEP 1: Match High-Conf Dets with Confirmed Tracks (Using User's ReID+KF Logic) ---
        if confirmed_tracks and det_high_indices:
            cost_matrix = np.ones((len(det_high_indices), len(confirmed_tracks)), dtype=float)
            
            for r, d_idx in enumerate(det_high_indices):
                for c, pid in enumerate(confirmed_tracks):
                    track = self.tracks[pid]
                    fused_feat = self._rbm_fuse(feats[d_idx], track)
                    dist_feat = cosine_distance(feats[d_idx], fused_feat)
                    sim_feat = 1.0 - (2.0 * dist_feat)
                    val_iou = iou(detections[d_idx], track.bbox)

                    # Sticky Logic
                    if val_iou > IOU_STICKY_THRESHOLD:
                        if sim_feat > REID_VETO_THRESHOLD:
                            cost_matrix[r, c] = 0.001 
                            continue
                    
                    tr_cent = centroid(track.bbox); det_cent = centroid(detections[d_idx])
                    dist_move = motion_distance_gate(det_cent, tr_cent, track.kf.get_velocity())
                    cost_matrix[r, c] = 0.5*dist_feat + 0.3*dist_move + 0.2*(1.0-val_iou)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < BASE_COSINE_THRESH or cost_matrix[r,c] < 0.01:
                    d_idx = det_high_indices[r]
                    pid = confirmed_tracks[c]
                    self.tracks[pid].update(detections[d_idx], frame_idx, feat=feats[d_idx], hist=det_hists[d_idx])
                    assignments.append((pid, detections[d_idx]))
                    assigned_pids.add(pid)
                    matched_high_indices.add(d_idx)

        # --- STEP 2: Match Remaining Tracks with Low-Conf Dets (IOU Only) ---
        # Identify leftovers from Step 1
        leftover_tracks = [pid for pid in confirmed_tracks if pid not in assigned_pids]
        
        if leftover_tracks and det_low_indices:
            cost_matrix_low = np.zeros((len(det_low_indices), len(leftover_tracks)), dtype=float)
            
            for r, d_idx in enumerate(det_low_indices):
                for c, pid in enumerate(leftover_tracks):
                    track = self.tracks[pid]
                    # IOU only for low confidence (standard ByteTrack)
                    val_iou = iou(detections[d_idx], track.bbox)
                    cost_matrix_low[r, c] = 1.0 - val_iou

            row_ind, col_ind = linear_sum_assignment(cost_matrix_low)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix_low[r, c] < (1.0 - 0.5): # IOU threshold 0.5 for low conf
                    d_idx = det_low_indices[r]
                    pid = leftover_tracks[c]
                    # Update without feature/hist for low conf to avoid drift
                    self.tracks[pid].update(detections[d_idx], frame_idx, feat=None, hist=None)
                    assignments.append((pid, detections[d_idx]))
                    assigned_pids.add(pid)
                    matched_low_indices.add(d_idx)

        # --- STEP 3: Match Unconfirmed/Lost with Remaining High-Conf Dets (User Recovery) ---
        # remaining high dets
        remaining_high_indices = [i for i in det_high_indices if i not in matched_high_indices]
        
        for d_idx in remaining_high_indices:
            # Try to recover using your custom logic
            recovered_pid, source_type = self._recover_id_with_position_logic(
                feats[d_idx], detections[d_idx], det_hists[d_idx], frame_idx
            )

            # --- FIX: PREVENT DUPLICATE ID ASSIGNMENT ---
            if recovered_pid is not None and recovered_pid in assigned_pids:
                recovered_pid = None 
            # ---------------------------------------------

            if recovered_pid is not None:
                if source_type == 'deleted':
                    for idx, entry in enumerate(self.deleted_tracks):
                        if entry[-1] == recovered_pid:
                            del self.deleted_tracks[idx]; break
                    tr = Track(recovered_pid, detections[d_idx], frame_idx, feat=feats[d_idx], hist=det_hists[d_idx])
                    tr.is_confirmed = True; tr.locked = True
                    self.tracks[recovered_pid] = tr
                else:
                    self.tracks[recovered_pid].update(detections[d_idx], frame_idx, feat=feats[d_idx], hist=det_hists[d_idx])
                assignments.append((recovered_pid, detections[d_idx])); assigned_pids.add(recovered_pid)
            else:
                # NEW
                pid = self.next_pid; self.next_pid += 1
                tr = Track(pid, detections[d_idx], frame_idx, feat=feats[d_idx], hist=det_hists[d_idx])
                self.tracks[pid] = tr
                assignments.append((pid, detections[d_idx])); assigned_pids.add(pid); self.total_unique_fishes += 1

        # --- STEP 5: CLEANUP ---
        for pid in list(self.tracks.keys()):
            if pid not in assigned_pids:
                track = self.tracks[pid]; track.increment_miss()
                # occlusion survival
                if track.occluded and track.occlusion_timer < 120: continue
                if track.miss_count > MAX_MISS_FRAMES:
                    mf = track.get_reid_feature(); hist = track.hist
                    vel = track.kf.get_velocity()
                    self.deleted_tracks.append((mf, track.bbox.copy(), frame_idx, hist, vel, pid))
                    del self.tracks[pid]
        return assignments

    def get_active_count(self):
        return sum(1 for t in self.tracks.values() if t.miss_count < INACTIVE_AFTER_MISSES)

# ---------------- MAIN RUN ----------------
def run(video_path, output_path, model_path, stats_output):
    print(f"Starting Sticky-IOU ByteTrack v7 (Occlusion Fixed)...")
    yolo = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    W, H = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), FORCED_FPS, (W,H))
    
    tracker = EnhancedByteTracker(DEVICE)
    tracker.set_frame_size(W,H)
    frame_idx = 0
    if os.path.exists(stats_output): os.remove(stats_output)
    start = time.time()
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        # Lowered YOLO conf threshold to 0.1 to allow ByteTrack low-conf matching
        results = yolo(frame, conf=LOW_CONF_THRESHOLD, iou=NMS_IOU, verbose=False)[0]
        dets, feats, confs = [], [], []
        if results.boxes:
            boxes = results.boxes.xyxy.cpu().numpy()
            cfs = results.boxes.conf.cpu().numpy()
            valid_indices=[]
            for i,(box,conf) in enumerate(zip(boxes, cfs)):
                w = box[2] - box[0]; h = box[3] - box[1]
                if conf > LOW_CONF_THRESHOLD and (w*h) > 60: valid_indices.append(i)
            if valid_indices:
                bboxes = boxes[valid_indices].tolist()
                final_confs = cfs[valid_indices].tolist()
                
                # Only extract ReID for high confidence detections to save speed/noise
                # (Strictly speaking ByteTrack doesn't need ReID for low conf)
                # But your architecture computes it per batch. We will just compute for all 
                # valid indices to keep code simple "same to same".
                raw_feats = tracker.reid.extract_batch(frame, bboxes)
                
                for bb, f, c in zip(bboxes, raw_feats, final_confs):
                    dets.append(np.array(bb)); feats.append(f); confs.append(c)
                    
        assigns = tracker.match(dets, feats, confs, frame_idx, (W,H), frame_img=frame)
        
        for pid, box in assigns:
            x1,y1,x2,y2 = map(int, box)
            color = (0,255,0) if tracker.tracks[pid].is_confirmed else (0,165,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID {pid}", (x1, max(12,y1-5)), 0, 0.6, (255,255,255), 2)
        
        count = tracker.get_active_count()
        proc_fps = frame_idx / (time.time() - start + 1e-9)
        cv2.putText(frame, f"Count: {count} | Total: {tracker.total_unique_fishes}", (10,30), 0, 0.6, (0,0,255), 2)
        out.write(frame)
        
        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx} | Active: {count} | Total: {tracker.total_unique_fishes} | ProcFPS: {proc_fps:.1f}")
            with open(stats_output, "a") as f:
                f.write(f"{frame_idx},{count},{tracker.total_unique_fishes}\n")
    cap.release(); out.release()
    print("Done.")

if __name__ == "__main__":
    run(VIDEO_PATH, OUTPUT_PATH, MODEL_PATH, STATS_OUTPUT)
    
    
    
    