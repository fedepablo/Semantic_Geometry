#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETUP AND QUICK TEST SCRIPT
===========================
This script ensures all dependencies are installed and runs a quick test
to verify the semantic geometry hypothesis is worth pursuing.

Run this first before the full pipeline!
"""

# Step 1: Install all requirements
print("🚀 Setting up environment for Semantic Geometry Analysis...")
print("-" * 50)

import subprocess
import sys

def install_requirements():
    """Install all required packages"""
    requirements = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'sentence-transformers',
        'torch',
        'transformers',
        'umap-learn',
        'scikit-learn',
        'plotly',
        'networkx',
        'tqdm',
        'scipy'
    ]
    
    print("📦 Installing required packages...")
    for package in requirements:
        print(f"  - {package}")
    
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + requirements)
    print("✅ All packages installed successfully!\n")

# Run installation
install_requirements()

# Step 2: Quick feasibility test
print("🔬 Running quick feasibility test...")
print("-" * 50)

import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
import warnings
warnings.filterwarnings('ignore')

# Quick test with 10 concepts
test_concepts = {
    "water": ("water", "水"),
    "fire": ("fire", "火"),
    "mountain": ("mountain", "山"),
    "wisdom": ("wisdom", "智慧"),
    "creation": ("creation", "创造"),
    "time": ("time", "时间"),
    "space": ("space", "空间"),
    "life": ("life", "生命"),
    "mind": ("mind", "心智"),
    "harmony": ("harmony", "和谐")
}

print("Loading model...")
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# Extract embeddings
en_texts = [v[0] for v in test_concepts.values()]
zh_texts = [v[1] for v in test_concepts.values()]

print("Extracting embeddings...")
en_embeddings = model.encode(en_texts)
zh_embeddings = model.encode(zh_texts)

# Compute density metrics
def compute_density(embeddings, k=3):
    """Simple density metric: average distance to k nearest neighbors"""
    distances = cosine_distances(embeddings)
    np.fill_diagonal(distances, np.inf)
    k_nearest = np.sort(distances, axis=1)[:, :k]
    return 1.0 / k_nearest.mean()

density_en = compute_density(en_embeddings)
density_zh = compute_density(zh_embeddings)

print(f"\n📊 Quick Results:")
print(f"  English density score: {density_en:.3f}")
print(f"  Chinese density score: {density_zh:.3f}")
print(f"  Difference: {((density_zh - density_en) / density_en * 100):.1f}%")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Distance matrices
im1 = ax1.imshow(cosine_distances(en_embeddings), cmap='viridis')
ax1.set_title('English Semantic Distances')
ax1.set_xticks(range(len(test_concepts)))
ax1.set_xticklabels([v[0] for v in test_concepts.values()], rotation=45, ha='right')
ax1.set_yticks(range(len(test_concepts)))
ax1.set_yticklabels([v[0] for v in test_concepts.values()])
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(cosine_distances(zh_embeddings), cmap='viridis')
ax2.set_title('Chinese Semantic Distances')
ax2.set_xticks(range(len(test_concepts)))
ax2.set_xticklabels([v[1] for v in test_concepts.values()], rotation=45, ha='right')
ax2.set_yticks(range(len(test_concepts)))
ax2.set_yticklabels([v[1] for v in test_concepts.values()])
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig('quick_test_distances.png', dpi=150, bbox_inches='tight')
plt.show()

# Test specific hypothesis: water-life connection
water_idx = list(test_concepts.keys()).index("water")
life_idx = list(test_concepts.keys()).index("life")

dist_en = cosine_distances(en_embeddings)[water_idx, life_idx]
dist_zh = cosine_distances(zh_embeddings)[water_idx, life_idx]

print(f"\n🔍 Specific test - 'water' to 'life' distance:")
print(f"  English: {dist_en:.3f}")
print(f"  Chinese: {dist_zh:.3f}")
print(f"  Chinese is {((dist_en - dist_zh) / dist_en * 100):.1f}% closer")

# Decision
print("\n" + "="*50)
if density_zh > density_en:
    print("✅ PROMISING: Chinese shows higher semantic density!")
    print("   This suggests the hypothesis has merit.")
    print("   Proceed with full analysis pipeline.")
else:
    print("🤔 UNEXPECTED: Chinese shows lower semantic density.")
    print("   This doesn't support the initial hypothesis.")
    print("   Consider: ")
    print("   - Testing with different models")
    print("   - Examining specific concept categories")
    print("   - Adjusting the theoretical framework")

print("\n📁 Quick test complete! Distance matrix saved as 'quick_test_distances.png'")
print("Ready to run full pipeline? Use the main notebook!")