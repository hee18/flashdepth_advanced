"""
Measure CLS token cosine distance on TartanAir samples.

Three experiments:
1. Adjacent frames (stride=1) within each sequence
2. Far-apart frames within the same sequence (stride=50, 100, 200)
3. Cross-sequence pairs (different scenes entirely)
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import itertools

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flashdepth.dinov2 import DINOv2


def load_frames(scene_path, num_frames=50, stride=1, offset=0):
    """Load frames from a TartanAir scene."""
    img_dir = os.path.join(scene_path, 'image_left')
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    files = files[offset:]
    files = files[:num_frames * stride:stride][:num_frames]

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frames = []
    for f in files:
        img = Image.open(os.path.join(img_dir, f)).convert('RGB')
        frames.append(transform(img))

    if len(frames) == 0:
        return None
    return torch.stack(frames)  # [T, 3, 518, 518]


def get_cls_tokens(encoder, frames, device, batch_size=8):
    """Extract CLS tokens from frames."""
    T = frames.shape[0]
    cls_tokens = []

    for i in range(0, T, batch_size):
        batch = frames[i:i+batch_size].to(device)
        with torch.no_grad():
            features = encoder.forward_features(batch)
            cls = features['x_norm_clstoken']
            cls_tokens.append(cls.cpu())

    return torch.cat(cls_tokens, dim=0)  # [T, embed_dim]


def cosine_distance(a, b):
    """Compute cosine distance between two vectors."""
    return (1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)).item()


def print_stats(distances, label):
    """Print statistics for a list of distances."""
    arr = np.array(distances)
    print(f"  {label}:")
    print(f"    Count:  {len(arr)}")
    print(f"    Mean:   {arr.mean():.6f}")
    print(f"    Std:    {arr.std():.6f}")
    print(f"    Min:    {arr.min():.6f}")
    print(f"    Max:    {arr.max():.6f}")
    print(f"    Median: {np.median(arr):.6f}")
    print(f"    P5:     {np.percentile(arr, 5):.6f}")
    print(f"    P25:    {np.percentile(arr, 25):.6f}")
    print(f"    P75:    {np.percentile(arr, 75):.6f}")
    print(f"    P95:    {np.percentile(arr, 95):.6f}")
    print(f"    P99:    {np.percentile(arr, 99):.6f}")


def main():
    device = "cuda:0"
    data_root = "/data/datasets/tartanair"

    print("Loading DINOv2 ViT-L encoder...")
    encoder = DINOv2(model_name='vitl', patch_size=14)
    encoder = encoder.to(device)
    encoder.eval()

    scenes = [
        ("abandonedfactory", "Easy", "P001"),
        ("hospital", "Easy", "P001"),
        ("ocean", "Easy", "P001"),
        ("office", "Easy", "P001"),
        ("seasidetown", "Easy", "P001"),
        ("japanesealley", "Easy", "P001"),
    ]

    # Collect CLS tokens per scene (sample more frames for far-apart analysis)
    scene_cls = {}  # scene_name -> cls_tokens [T, embed_dim]
    scene_names_valid = []

    for scene_name, difficulty, p_dir in scenes:
        scene_path = os.path.join(data_root, scene_name, difficulty, p_dir)
        if not os.path.exists(scene_path):
            print(f"  Skipping {scene_name} (not found)")
            continue

        # Load 300 frames (stride=1) to cover a long range
        frames = load_frames(scene_path, num_frames=300, stride=1)
        if frames is None:
            continue

        print(f"  {scene_name}: loaded {frames.shape[0]} frames")
        cls_tokens = get_cls_tokens(encoder, frames, device)
        scene_cls[scene_name] = cls_tokens
        scene_names_valid.append(scene_name)

    # ================================================================
    # Experiment 1: Adjacent frames (stride=1)
    # ================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: Adjacent frames (stride=1)")
    print(f"{'='*70}")

    adj_distances = []
    for name in scene_names_valid:
        tokens = scene_cls[name]
        for t in range(tokens.shape[0] - 1):
            d = cosine_distance(tokens[t], tokens[t+1])
            adj_distances.append(d)
    print_stats(adj_distances, "Adjacent (dt=1)")

    # ================================================================
    # Experiment 2: Far-apart frames within same sequence
    # ================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: Far-apart frames within SAME sequence")
    print(f"{'='*70}")

    for gap in [10, 25, 50, 100, 200]:
        gap_distances = []
        for name in scene_names_valid:
            tokens = scene_cls[name]
            T = tokens.shape[0]
            for t in range(0, T - gap, max(gap // 2, 1)):
                d = cosine_distance(tokens[t], tokens[t + gap])
                gap_distances.append(d)
        if gap_distances:
            print_stats(gap_distances, f"Same sequence, dt={gap}")

    # ================================================================
    # Experiment 3: Cross-sequence pairs (different scenes)
    # ================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 3: CROSS-SEQUENCE pairs (different scenes)")
    print(f"{'='*70}")

    cross_distances = []
    # Sample 20 random frames from each scene, compute all cross-scene pairs
    np.random.seed(42)
    sampled_cls = {}
    for name in scene_names_valid:
        tokens = scene_cls[name]
        T = tokens.shape[0]
        indices = np.random.choice(T, size=min(20, T), replace=False)
        sampled_cls[name] = tokens[indices]

    for (name_a, name_b) in itertools.combinations(scene_names_valid, 2):
        pair_dists = []
        for i in range(sampled_cls[name_a].shape[0]):
            for j in range(sampled_cls[name_b].shape[0]):
                d = cosine_distance(sampled_cls[name_a][i], sampled_cls[name_b][j])
                pair_dists.append(d)
        cross_distances.extend(pair_dists)
        arr = np.array(pair_dists)
        print(f"  {name_a} vs {name_b}: mean={arr.mean():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")

    print()
    print_stats(cross_distances, "All cross-sequence pairs")

    # ================================================================
    # Summary comparison
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    categories = [
        ("Adjacent (dt=1)", adj_distances),
    ]
    for gap in [10, 25, 50, 100, 200]:
        gap_dists = []
        for name in scene_names_valid:
            tokens = scene_cls[name]
            T = tokens.shape[0]
            for t in range(0, T - gap, max(gap // 2, 1)):
                gap_dists.append(cosine_distance(tokens[t], tokens[t + gap]))
        if gap_dists:
            categories.append((f"Same seq dt={gap}", gap_dists))
    categories.append(("Cross-sequence", cross_distances))

    print(f"  {'Category':<22} {'Mean':>10} {'Median':>10} {'P95':>10} {'Max':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for label, dists in categories:
        arr = np.array(dists)
        print(f"  {label:<22} {arr.mean():>10.6f} {np.median(arr):>10.6f} {np.percentile(arr, 95):>10.6f} {arr.max():>10.6f}")

    print()
    print("--- Threshold Recommendation ---")
    cross_arr = np.array(cross_distances)
    same_far_dists = []
    for name in scene_names_valid:
        tokens = scene_cls[name]
        T = tokens.shape[0]
        for t in range(0, T - 200, 100):
            same_far_dists.append(cosine_distance(tokens[t], tokens[t + 200]))
    same_far_arr = np.array(same_far_dists) if same_far_dists else np.array([0])

    print(f"  Same seq dt=200 max:     {same_far_arr.max():.4f}")
    print(f"  Cross-sequence min:      {cross_arr.min():.4f}")
    print(f"  Cross-sequence P5:       {np.percentile(cross_arr, 5):.4f}")
    gap = cross_arr.min() - same_far_arr.max()
    midpoint = (cross_arr.min() + same_far_arr.max()) / 2
    print(f"  Gap between them:        {gap:.4f}")
    print(f"  Suggested tau (midpoint): {midpoint:.4f}")


if __name__ == "__main__":
    main()
