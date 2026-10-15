# -*- coding: utf-8 -*-
"""
Cross-Dataset Deepfake Detection with Domain Generalization
=============================================================
Train  : FaceForensics++ (labeled, source domain)
Adapt  : Celeb-DF, 50% split, UNLABELED (target domain, adaptation pool)
Test   : Celeb-DF, remaining 50% split, held out, NEVER seen in training

CHANGES IN THIS VERSION
-----------------------
The sparse Top-3 gate has been REPLACED by AttentionFusion, a dense
attention-based fusion over all four experts:

  1. The four 256-d expert embeddings are treated as a length-4 token
     sequence. A learned "expert identity" embedding is added to each token
     (analogous to positional embeddings) so the fusion knows WHICH expert
     produced each vector.
  2. A multi-head self-attention block (+ residual FFN + LayerNorm) lets the
     experts exchange information BEFORE fusion -- e.g. the DCT expert's
     spectral evidence can contextually recalibrate the RGB experts.
  3. A learned query token pools the refined expert tokens via attention.
     The pooling attention weights form a DENSE softmax over all 4 experts:
     every expert contributes to every prediction (no expert is dropped),
     and the weights remain interpretable per-sample.

Everything else (DCT expert, DANN domain-adversarial branch, SupCon loss,
schedules, evaluation protocol) is unchanged from the previous version.

Total loss = CE + w_supcon * SupCon + lambda_p * DomainAdversarialCE

METHODOLOGICAL CAVEAT (unchanged): this is a TRANSDUCTIVE setup -- the model
sees unlabeled target-domain images at train time. Final metrics come from a
disjoint held-out split. For a strictly inductive claim, set
UDA_ENABLED = False (DCT + SupCon only, no target images touched).
"""

import os
import gc
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Toggle: True = full UDA (domain-adversarial + DCT + SupCon)
#         False = inductive-safe mode (DCT + SupCon only, no target images used)
UDA_ENABLED = True


# --------------------------------------------------------------------------- #
# Data paths
# --------------------------------------------------------------------------- #
TRAIN_DIR = "/kaggle/input/datasets/saptarshipani/faceforensics/FaceForensic/train"
VAL_DIR   = "/kaggle/input/datasets/saptarshipani/faceforensics/FaceForensic/validation"
TEST_DIR  = "/kaggle/input/datasets/saptarshipani/celeb-df/Celeb Image Dataset/testImage"


def load_paths_labels(folder_path):
    """Paths and labels are always kept in sync (fixed desync bug)."""
    paths, labels = [], []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lname = file_name.lower()
        if lname.startswith("fake"):
            paths.append(file_path); labels.append(1)
        elif lname.startswith("real"):
            paths.append(file_path); labels.append(0)
        # else: silently skip (ambiguous filename)
    assert len(paths) == len(labels)
    return paths, labels


def list_image_paths(folder_path):
    """For the unlabeled target pool -- just paths, no label parsing."""
    paths = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(folder_path, file_name))
    return paths


train_paths, train_labels = load_paths_labels(TRAIN_DIR)
val_paths, val_labels     = load_paths_labels(VAL_DIR)

# --- Celeb-DF split: adaptation pool (unlabeled, train-time) vs true test pool ---
all_celeb_paths = list_image_paths(TEST_DIR)
rng = random.Random(42)
rng.shuffle(all_celeb_paths)
split_idx = len(all_celeb_paths) // 2
target_adapt_paths = all_celeb_paths[:split_idx]   # unlabeled, used only for domain loss
true_test_paths    = all_celeb_paths[split_idx:]   # held out, final evaluation only


def labels_from_paths(paths):
    labels = []
    for p in paths:
        fname = os.path.basename(p).lower()
        if fname.startswith("fake"):
            labels.append(1)
        elif fname.startswith("real"):
            labels.append(0)
        else:
            labels.append(-1)  # unknown, will be filtered
    return labels

true_test_labels = labels_from_paths(true_test_paths)
filtered = [(p, l) for p, l in zip(true_test_paths, true_test_labels) if l != -1]
true_test_paths, true_test_labels = zip(*filtered)
true_test_paths, true_test_labels = list(true_test_paths), list(true_test_labels)

print(len(train_paths), len(val_paths), len(target_adapt_paths), len(true_test_paths))


def show_dataset_stats(name, labels):
    total = len(labels)
    print(f"\n{name} Dataset\n{'-'*30}\nTotal Images : {total}\n"
          f"Real Images  : {labels.count(0)}\nFake Images  : {labels.count(1)}")

show_dataset_stats("Train", train_labels)
show_dataset_stats("Validation", val_labels)
show_dataset_stats("Celeb-DF Test (held out)", true_test_labels)


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
IMG_SIZE = 256

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Target-domain transform for adaptation: deliberately MILD -- we want the
# domain classifier aligning the TRUE dataset distribution.
target_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


class DeepfakeDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class UnlabeledImageDataset(Dataset):
    """For the target adaptation pool -- no labels are read or used."""
    def __init__(self, paths, transform=None):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


BATCH_SIZE = 16

train_dataset  = DeepfakeDataset(train_paths, train_labels, train_transform)
val_dataset    = DeepfakeDataset(val_paths, val_labels, val_transform)
test_dataset   = DeepfakeDataset(true_test_paths, true_test_labels, val_transform)
target_dataset = UnlabeledImageDataset(target_adapt_paths, target_transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
val_loader    = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)


# --------------------------------------------------------------------------- #
# Backbone experts (unchanged)
# --------------------------------------------------------------------------- #
class MultiScaleCNN(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv_d1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, dilation=1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_d2 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=2, dilation=2), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_d4 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=4, dilation=4), nn.BatchNorm2d(64), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        fused = self.conv_d1(x) + self.conv_d2(x) + self.conv_d4(x)
        fused = self.pool(fused)
        fused = torch.flatten(fused, 1)
        return self.fc(fused)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class DirectConvBlock(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.res1 = ResidualBlock(64)
        self.downsample = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.res2 = ResidualBlock(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.stem(x); x = self.res1(x); x = self.downsample(x); x = self.res2(x)
        x = self.pool(x); x = torch.flatten(x, 1)
        return self.fc(x)


# --------------------------------------------------------------------------- #
# Differentiable block-DCT frequency expert (unchanged)
# --------------------------------------------------------------------------- #
def _build_dct_basis(block_size: int = 8) -> torch.Tensor:
    """Exact 2D DCT-II basis as a (block_size^2, 1, block_size, block_size)
    conv weight tensor. Stride = block_size computes the true block-DCT
    coefficient for every non-overlapping block, differentiably."""
    n = block_size
    basis_1d = np.zeros((n, n))
    for k in range(n):
        ck = np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)
        for i in range(n):
            basis_1d[k, i] = ck * np.cos((np.pi * (2 * i + 1) * k) / (2 * n))

    filters = np.zeros((n * n, 1, n, n))
    for u in range(n):
        for v in range(n):
            filters[u * n + v, 0] = np.outer(basis_1d[u], basis_1d[v])
    return torch.tensor(filters, dtype=torch.float32)


class FreqDCTBranch(nn.Module):
    """4th expert: block-DCT spectral features, robust to dataset-specific
    compression/resize pipelines relative to raw RGB texture cues."""
    def __init__(self, out_dim: int = 256, block_size: int = 8):
        super().__init__()
        self.block_size = block_size
        dct_weights = _build_dct_basis(block_size)  # (64, 1, 8, 8)
        self.register_buffer("dct_weights", dct_weights)  # fixed, non-trainable

        n_freqs = block_size * block_size
        self.freq_conv = nn.Sequential(
            nn.Conv2d(n_freqs, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, out_dim))

    def forward(self, x):
        gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3])
        coeffs = F.conv2d(gray, self.dct_weights, stride=self.block_size)  # (B, 64, H/8, W/8)
        coeffs = torch.log1p(coeffs.abs())
        feat = self.freq_conv(coeffs)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)
        return self.proj(feat)


# --------------------------------------------------------------------------- #
# Gradient Reversal Layer + Domain Classifier (DANN) (unchanged)
# --------------------------------------------------------------------------- #
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


class DomainClassifier(nn.Module):
    """Binary classifier: source (FF++) = 0, target (Celeb-DF) = 1.
    Fed the fused feature THROUGH the GRL during training."""
    def __init__(self, in_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Supervised Contrastive Loss (Khosla et al. 2020) (unchanged)
# --------------------------------------------------------------------------- #
class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device_ = features.device
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device_)

        sim = torch.div(torch.matmul(features, features.T), self.temperature)
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device_)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)

        pos_count = mask.sum(1)
        pos_count = torch.clamp(pos_count, min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count

        loss = -mean_log_prob_pos.mean()
        return loss


# --------------------------------------------------------------------------- #
# NEW: Dense attention-based fusion over ALL experts (replaces TopKGate)
# --------------------------------------------------------------------------- #
class AttentionFusion(nn.Module):
    """Fuses the E expert embeddings as a token sequence.

    Stage 1 -- expert interaction:
        tokens = experts + learned expert-identity embeddings
        tokens = LayerNorm(tokens + MHSA(tokens))          (experts exchange evidence)
        tokens = LayerNorm(tokens + FFN(tokens))           (per-token refinement)

    Stage 2 -- dense pooling:
        A single learned query attends over the refined tokens.
        The pooling attention weights are a DENSE softmax over all E experts
        (every expert always contributes), and are returned for
        interpretability / logging.
    """
    def __init__(self, dim: int = 256, num_experts: int = 4,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.expert_id = nn.Parameter(torch.zeros(1, num_experts, dim))
        nn.init.trunc_normal_(self.expert_id, std=0.02)

        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

        self.pool_query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.pool_query, std=0.02)
        self.pool_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, expert_feats):
        # expert_feats: (B, E, D)
        x = expert_feats + self.expert_id
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ffn(x))

        q = self.pool_query.expand(x.size(0), -1, -1)          # (B, 1, D)
        fused, pool_w = self.pool_attn(q, x, x)                # (B, 1, D), (B, 1, E)
        return fused.squeeze(1), pool_w.squeeze(1)             # (B, D), (B, E)


EXPERT_DIM  = 256
SWIN_DIM    = 1024
NUM_EXPERTS = 4  # CNN, Swin, DirectConv, DCT-frequency


class HybridDeepfakeModelAttnFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0)
        self.swin_proj = nn.Sequential(nn.Linear(SWIN_DIM, EXPERT_DIM), nn.ReLU())

        self.cnn_branch = MultiScaleCNN(out_dim=EXPERT_DIM)
        self.direct_branch = DirectConvBlock(out_dim=EXPERT_DIM)
        self.freq_branch = FreqDCTBranch(out_dim=EXPERT_DIM)

        # Dense attention fusion over all 4 experts (replaces TopKGate)
        self.fusion = AttentionFusion(dim=EXPERT_DIM, num_experts=NUM_EXPERTS,
                                      num_heads=4, dropout=0.1)

        self.classifier = nn.Sequential(
            nn.Linear(EXPERT_DIM, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)
        )

        self.domain_classifier = DomainClassifier(in_dim=EXPERT_DIM)

    def forward(self, x, lambda_p: float = 0.0, return_features: bool = False):
        e1 = self.cnn_branch(x)
        e2 = self.swin_proj(self.swin(x))
        e3 = self.direct_branch(x)
        e4 = self.freq_branch(x)

        experts = torch.stack([e1, e2, e3, e4], dim=1)         # (B, 4, 256)
        fused, expert_weights = self.fusion(experts)           # (B, 256), (B, 4)
        self.last_expert_weights = expert_weights.detach()     # for logging/analysis

        logits = self.classifier(fused)

        domain_logits = None
        if lambda_p != 0.0 or return_features:
            reversed_feat = grad_reverse(fused, lambda_p)
            domain_logits = self.domain_classifier(reversed_feat)

        if return_features:
            return logits, fused, domain_logits
        return logits


model = HybridDeepfakeModelAttnFusion().to(device)
print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")


# --------------------------------------------------------------------------- #
# Losses, optimizer, schedule
# --------------------------------------------------------------------------- #
criterion = nn.CrossEntropyLoss()
domain_criterion = nn.CrossEntropyLoss()
supcon_criterion = SupConLoss(temperature=0.1)

W_SUPCON = 0.3   # weight for supervised contrastive term

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)

EPOCHS = 25
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS - 5, 1))


class EarlyStopping:
    def __init__(self, patience: int = 7):
        self.patience, self.best_loss, self.counter, self.stop = patience, np.inf, 0, False

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss:
            self.best_loss, self.counter = val_loss, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

early_stopping = EarlyStopping(patience=7)


def dann_lambda(progress: float) -> float:
    """Standard DANN warmup: 0 -> 1 across total training progress."""
    return 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0


# --------------------------------------------------------------------------- #
# Training loop (source classification + SupCon + domain-adversarial)
# --------------------------------------------------------------------------- #
def train_one_epoch(model, source_loader, target_loader, epoch, total_epochs):
    model.train()
    running_cls_loss, running_supcon_loss, running_domain_loss = 0.0, 0.0, 0.0
    preds_all, labels_all = [], []

    total_batches = len(source_loader)
    target_iter = itertools.cycle(target_loader) if UDA_ENABLED else None
    print_every = max(1, total_batches // 10)

    for batch_idx, (images, labels) in enumerate(source_loader):
        images, labels = images.to(device), labels.to(device)

        global_progress = (epoch + batch_idx / total_batches) / total_epochs
        lambda_p = dann_lambda(global_progress) if UDA_ENABLED else 0.0

        optimizer.zero_grad()

        logits, fused_src, domain_logits_src = model(images, lambda_p=lambda_p, return_features=True)
        cls_loss = criterion(logits, labels)
        supcon_loss = supcon_criterion(fused_src, labels)

        loss = cls_loss + W_SUPCON * supcon_loss

        domain_loss_val = torch.tensor(0.0, device=device)
        if UDA_ENABLED:
            target_images = next(target_iter).to(device)
            _, fused_tgt, domain_logits_tgt = model(target_images, lambda_p=lambda_p, return_features=True)

            domain_labels_src = torch.zeros(images.size(0), dtype=torch.long, device=device)
            domain_labels_tgt = torch.ones(target_images.size(0), dtype=torch.long, device=device)

            domain_loss_src = domain_criterion(domain_logits_src, domain_labels_src)
            domain_loss_tgt = domain_criterion(domain_logits_tgt, domain_labels_tgt)
            domain_loss_val = domain_loss_src + domain_loss_tgt

            loss = loss + domain_loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_cls_loss += cls_loss.item()
        running_supcon_loss += supcon_loss.item()
        running_domain_loss += domain_loss_val.item() if UDA_ENABLED else 0.0

        preds = torch.argmax(logits, dim=1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

        if (batch_idx + 1) % print_every == 0:
            batch_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            mean_w = model.last_expert_weights.mean(dim=0).cpu().numpy()
            print(f"Train [{batch_idx+1}/{total_batches}] "
                  f"CLS: {cls_loss.item():.4f}  SupCon: {supcon_loss.item():.4f}  "
                  f"Domain: {domain_loss_val.item():.4f}  lambda_p: {lambda_p:.3f}  "
                  f"Acc: {batch_acc:.4f}  "
                  f"ExpertW [cnn,swin,direct,dct]: {np.round(mean_w, 3)}")

    return (running_cls_loss / total_batches,
            running_supcon_loss / total_batches,
            running_domain_loss / total_batches,
            accuracy_score(labels_all, preds_all))


def evaluate(model, loader):
    """Pure classification eval -- lambda_p=0 disables the GRL path."""
    model.eval()
    running_loss = 0.0
    preds_all, labels_all, probs_all = [], [], []
    total_batches = len(loader)
    print_every = max(1, total_batches // 10)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, lambda_p=0.0, return_features=False)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())

            if (batch_idx + 1) % print_every == 0:
                batch_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
                print(f"  Val [{batch_idx+1}/{total_batches}] Loss: {loss.item():.4f}  Acc: {batch_acc:.4f}")

    return (running_loss / total_batches, accuracy_score(labels_all, preds_all),
            labels_all, preds_all, probs_all)


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
train_cls_losses, train_supcon_losses, train_domain_losses = [], [], []
val_losses, train_accs, val_accs, val_aucs = [], [], [], []
best_accs = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    cls_loss, supcon_loss, domain_loss, train_acc = train_one_epoch(
        model, train_loader, target_loader, epoch, EPOCHS
    )
    val_loss, val_acc, y_true, y_pred, probs = evaluate(model, val_loader)
    val_auc = roc_auc_score(y_true, probs)

    if epoch < 5:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()

    train_cls_losses.append(cls_loss); train_supcon_losses.append(supcon_loss)
    train_domain_losses.append(domain_loss); val_losses.append(val_loss)
    train_accs.append(train_acc); val_accs.append(val_acc); val_aucs.append(val_auc)

    print(f"Train CLS: {cls_loss:.4f}  SupCon: {supcon_loss:.4f}  Domain: {domain_loss:.4f}  Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  Val AUC: {val_auc:.4f}")

    if val_acc > best_accs:
        best_accs = val_acc
        torch.save(model.state_dict(), "best_model_attnfusion_uda.pth")
        print("Best model saved")

    early_stopping(val_loss)
    if early_stopping.stop:
        print("Early stopping triggered.")
        break

    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
epochs_range = range(1, len(val_losses) + 1)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(epochs_range, train_cls_losses, label="Train CLS Loss")
plt.plot(epochs_range, val_losses, label="Val Loss")
plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.title("Classification Loss")

plt.subplot(1, 3, 2)
plt.plot(epochs_range, train_accs, label="Train Accuracy")
plt.plot(epochs_range, val_accs, label="Val Accuracy")
plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy Curves")

plt.subplot(1, 3, 3)
plt.plot(epochs_range, train_domain_losses, label="Domain Adv Loss")
plt.plot(epochs_range, train_supcon_losses, label="SupCon Loss")
plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.title("Auxiliary Losses")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()


# --------------------------------------------------------------------------- #
# Final evaluation on TRUE held-out Celeb-DF test split
# --------------------------------------------------------------------------- #
model.load_state_dict(torch.load("best_model_attnfusion_uda.pth"))
test_loss, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc_score = roc_auc_score(y_true, y_prob)

print(f"\nTest Results (held-out Celeb-DF split, never seen in training)")
print("-" * 60)
print(f"Accuracy  : {test_acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC Score : {auc_score:.4f}")
print("\nPer-Class Metrics:\n")
print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"]).plot(cmap="Blues")
plt.title("Confusion Matrix (Held-out Celeb-DF)")
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Held-out Celeb-DF)"); plt.legend()
plt.savefig("roc_curve.png", dpi=150)
plt.show()
