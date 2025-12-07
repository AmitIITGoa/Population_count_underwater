import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from segment_anything import sam_model_registry

# ======== CONFIG ========
DATA_DIR = "custom_dataset/images"
ANNOTATION_FILE = "custom_dataset/annotations/annotations_coco.json"
MODEL_TYPE = "vit_l"              # vit_b, vit_l, vit_h
PRETRAINED_PT = "sam2_l.pt"       # your pretrained SAM weights
OUTPUT_DIR = "output_sam"
EPOCHS = 50
LR = 1e-5
BATCH_SIZE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")
# =========================


# ---- Dataset ----
class CocoSegDataset(Dataset):
    def __init__(self, images_dir, annotation_path, transform=None):
        with open(annotation_path, 'r') as f:
            self.coco = json.load(f)
        self.images_dir = images_dir
        self.transform = transform
        self.images = {img["id"]: img for img in self.coco["images"]}
        self.annotations = self.coco["annotations"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_info = self.images[ann["image_id"]]
        img_path = os.path.join(self.images_dir, image_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        mask = torch.zeros((image.shape[1], image.shape[2]))
        if "segmentation" in ann and len(ann["segmentation"]) > 0:
            for seg in ann["segmentation"]:
                pts = torch.tensor(seg).view(-1, 2)
                x = pts[:, 0].long().clamp(0, mask.shape[1]-1)
                y = pts[:, 1].long().clamp(0, mask.shape[0]-1)
                mask[y, x] = 1.0

        return image, mask.unsqueeze(0)


# ---- Transforms ----
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])


# ---- Dataloader ----
dataset = CocoSegDataset(DATA_DIR, ANNOTATION_FILE, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ---- Load SAM model ----
print("🔹 Loading SAM model...")
sam_model = sam_model_registry[MODEL_TYPE](checkpoint=None)

if os.path.exists(PRETRAINED_PT):
    checkpoint = torch.load(PRETRAINED_PT, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sam_model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("✅ Loaded state_dict from pretrained .pt file")
    elif isinstance(checkpoint, dict):
        sam_model.load_state_dict(checkpoint, strict=False)
        print("✅ Loaded plain state_dict from pretrained .pt file")
    else:
        sam_model = checkpoint
        print("✅ Loaded full model object")
else:
    raise FileNotFoundError(f"❌ Pretrained .pt file not found at {PRETRAINED_PT}")

sam_model.to(DEVICE)
sam_model.train()


# ---- Optimizer and Loss ----
optimizer = torch.optim.Adam(sam_model.image_encoder.parameters(), lr=LR)
loss_fn = torch.nn.BCEWithLogitsLoss()


# ---- Training Loop ----
# ---- Training Loop ----
os.makedirs(OUTPUT_DIR, exist_ok=True)

for epoch in range(EPOCHS):
    total_loss = 0.0

    for imgs, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        # Forward through image encoder
        feats = sam_model.image_encoder(imgs)

        # Dummy positional encoding
        image_pe = torch.zeros_like(feats, device=DEVICE)

        # Dummy sparse + dense prompts
        dummy_sparse = torch.zeros((1, 256, sam_model.mask_decoder.transformer_dim), device=DEVICE)
        dummy_dense = torch.zeros_like(feats, device=DEVICE)

        # Forward through decoder
        preds = sam_model.mask_decoder(
            image_embeddings=feats,
            image_pe=image_pe,
            sparse_prompt_embeddings=dummy_sparse,
            dense_prompt_embeddings=dummy_dense,
            multimask_output=False
        )

        # ✅ Unpack tuple
        pred_masks, iou_pred = preds

        # Resize masks to ground truth size
        pred_masks = torch.nn.functional.interpolate(
            pred_masks, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )

        # Compute loss
        loss = loss_fn(pred_masks, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"📘 Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

    # Save checkpoint
    save_path = os.path.join(OUTPUT_DIR, f"sam_finetuned_epoch{epoch+1}.pt")
    torch.save(sam_model.state_dict(), save_path)
    print(f"💾 Saved model: {save_path}")

print("🎉 Training completed! All weights saved in:", OUTPUT_DIR)

