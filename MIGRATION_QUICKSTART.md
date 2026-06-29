# Quick Start: Cosmos 2.5 → Cosmos 3 Migration

This guide will help you evaluate whether to migrate from your custom Cosmos 2.5 implementation to Cosmos 3.

## Current Status

You've implemented ego/agent conditioning on Cosmos 2.5, but training produces blurry results. This directory contains:

1. **Analysis document**: `COSMOS_UPGRADE_ANALYSIS.md` - Detailed comparison and recommendation
2. **Extraction script**: `extract_ego_poses_for_cosmos3.py` - Convert your data to Cosmos 3 format
3. **Test script**: `test_ego_extraction.py` - Validate your data before full extraction

## Recommendation: Migrate to Cosmos 3

**Why?** Cosmos 3 has native ego-conditioning built into its architecture. What you're trying to build is now a standard feature.

---

## Step 1: Read the Analysis (10 minutes)

```bash
cat COSMOS_UPGRADE_ANALYSIS.md
```

Key sections:
- **Current Implementation Analysis**: What you built and why it's having issues
- **Cosmos 3 Capabilities**: What's available natively
- **Cost-Benefit Analysis**: Migration effort vs debugging current code
- **Migration Path**: Step-by-step guide

---

## Step 2: Test Your Data Compatibility (5 minutes)

Test if your kinematic data format works with the extraction script:

```bash
python test_ego_extraction.py
```

**Expected output:**
```
Testing with: /scratch/user/u.pt152369/WM/data/datasetWM/sample.h5
======================================================================
H5 file keys: ['kinematics']
Loaded data from key: 'kinematics'
  Shape: (29, 32, 18)
  Dtype: float32

Data dimensions:
  T (frames): 29
  N (agents): 32
  D (features): 18

Agent analysis:
  Valid frames per agent:
    Agent 0: 29/29 frames (100.0%)
    ...

Ego vehicle trajectory:
  Start position: (0.00, 0.00, 1.50)
  End position:   (45.32, -12.15, 1.50)
  Total displacement: 47.93 meters
  Average speed: 1.73 m/frame

✓ Test successful! Data format looks compatible.
```

If this works, proceed to Step 3.

---

## Step 3: Extract Ego-Poses (30 minutes)

Extract ego-pose sequences from your full dataset:

```bash
python extract_ego_poses_for_cosmos3.py \
    --input_dir /scratch/user/u.pt152369/WM/data/datasetWM \
    --output_dir ./cosmos3_ego_poses \
    --num_samples 20
```

**What this does:**
- Converts your kinematic data `[T, N, 18]` to Cosmos 3 format `[T, 9]`
- Each frame: `[tx, ty, tz, r1, r2, r3, r4, r5, r6]` (position + 6D rotation)
- Creates JSON files compatible with Cosmos 3 inference API
- Generates summary with statistics

**Output:**
```
cosmos3_ego_poses/
├── sample_001_ego_pose.json
├── sample_002_ego_pose.json
├── ...
└── extraction_summary.json
```

---

## Step 4: Install Cosmos 3 (1-2 hours)

Follow the official Cosmos 3 setup guide:

```bash
# Option 1: Using Cosmos Framework (recommended)
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
pip install -e .

# Option 2: Using Diffusers (lighter weight)
pip install transformers diffusers accelerate
pip install cosmos-framework  # For utilities
```

Download models:
```bash
# Cosmos 3 Nano (fastest, good for testing)
huggingface-cli download nvidia/cosmos3-nano

# Cosmos 3 Base (balanced)
huggingface-cli download nvidia/cosmos3-base
```

---

## Step 5: Run Inference Test (30 minutes)

Test forward dynamics with one of your extracted samples:

```python
import json
from cosmos_framework import Cosmos3Client

# Load extracted ego-pose
with open('./cosmos3_ego_poses/sample_001_ego_pose.json') as f:
    data = json.load(f)

# Initialize Cosmos 3 client
client = Cosmos3Client(model="nvidia/cosmos3-nano")

# Run forward dynamics: action → future video
result = client.generate(
    video=data['video'],              # Your nuScenes video
    action=data['ego_pose'],          # 9D ego-pose sequence
    embodiment='autonomous_vehicle',
    max_output_frames=60,
    fps=10
)

# Save output
result.save_video('output_cosmos3.mp4')

print(f"Generated {result.num_frames} frames")
print(f"Video quality: {result.metrics.get('fvd', 'N/A')}")
```

**Compare with your current Cosmos 2.5 results:**
- Is the video sharper (less blurry)?
- Does ego-motion look consistent?
- Are physics plausible?

---

## Step 6: Decision Point

### If Cosmos 3 baseline is better than your Cosmos 2.5:
→ **Proceed with migration** (see Step 7)

### If Cosmos 3 baseline is worse:
→ Consider debugging Cosmos 2.5 (see `COSMOS_UPGRADE_ANALYSIS.md` Appendix)

**Expected result:** Cosmos 3 should be noticeably better for ego-conditioned generation.

---

## Step 7: Post-Training on NuScenes (1-2 weeks)

If baseline Cosmos 3 is good but you want better quality on your specific domain:

### 7.1 Prepare Full Dataset

```bash
# Extract all samples
python extract_ego_poses_for_cosmos3.py \
    --input_dir /scratch/user/u.pt152369/WM/data/datasetWM \
    --output_dir ./cosmos3_ego_poses_full \
    --num_samples -1  # Process all
```

### 7.2 Format for Training

Create a dataset manifest:

```python
# create_cosmos3_manifest.py
import json
from pathlib import Path

ego_dir = Path('./cosmos3_ego_poses_full')
manifest = []

for ego_file in sorted(ego_dir.glob('*_ego_pose.json')):
    with open(ego_file) as f:
        data = json.load(f)
    
    manifest.append({
        'video': data['video'],
        'action': data['ego_pose'],
        'num_frames': data['num_frames'],
        'text': 'Autonomous driving scenario',  # Add captions if available
    })

with open('nuscenes_cosmos3_manifest.json', 'w') as f:
    json.dump(manifest, f)

print(f"Created manifest with {len(manifest)} samples")
```

### 7.3 Run Post-Training

```bash
# Using Cosmos Framework
cosmos-train \
    --config configs/action_post_training.yaml \
    --data nuscenes_cosmos3_manifest.json \
    --embodiment autonomous_vehicle \
    --model nvidia/cosmos3-base \
    --output_dir ./checkpoints/cosmos3_nuscenes \
    --max_iter 5000 \
    --batch_size 4 \
    --lr 1e-5
```

**Training config** (configs/action_post_training.yaml):
```yaml
model:
  name: cosmos3-base
  action_conditioning: true
  embodiment: autonomous_vehicle

data:
  manifest: nuscenes_cosmos3_manifest.json
  video_size: [720, 1280]
  num_frames: 29
  fps: 10

training:
  max_iter: 5000
  batch_size: 4
  learning_rate: 1e-5
  warmup_steps: 500
  gradient_accumulation_steps: 4
  
  loss:
    video_weight: 1.0
    action_consistency_weight: 0.1  # Much lower than your 0.1 kinematic loss

optimizer:
  type: adamw
  weight_decay: 0.01

checkpointing:
  save_every: 500
  keep_last_n: 5
```

---

## Expected Timeline

| Phase | Duration | What You Get |
|-------|----------|--------------|
| **Testing (Steps 1-6)** | 2-3 days | Know if Cosmos 3 works for you |
| **Post-Training (Step 7)** | 1-2 weeks | Production-ready model on your data |
| **Total (baseline)** | 2-3 days | Working ego-conditioned generation |
| **Total (fine-tuned)** | 2-3 weeks | Domain-optimized model |

**Compare with debugging Cosmos 2.5:** 2-4 weeks with uncertain outcome.

---

## FAQ

### Q: Will I lose my agent conditioning work?

A: Not entirely. If you still need multi-agent trajectories (not just ego), you can:
1. Use Cosmos 3's ego-conditioning as primary
2. Add a lightweight agent conditioning layer (much simpler than your current 130M param head)
3. See `COSMOS_UPGRADE_ANALYSIS.md` for "Simplified agent conditioning" example

### Q: What if Cosmos 3 doesn't work for my use case?

A: Then you have valuable data to debug Cosmos 2.5. See the Appendix in `COSMOS_UPGRADE_ANALYSIS.md` for targeted fixes:
- Initialize `kin_scale = 0.1` instead of 0
- Reduce `kinematic_loss_weight` from 0.1 to 0.01
- Add gradient clipping
- Freeze kinematic head for first 1000 iterations

### Q: How much compute do I need?

- **Testing (Cosmos 3 inference):** 1x A100 (40GB) - same as you have now
- **Post-training:** 4x A100 (40GB) for ~1 week - similar to your LoRA training

### Q: Can I use my existing LoRA setup?

A: Not directly, but Cosmos 3 supports post-training with similar concepts. The architecture is different (Mixture-of-Transformers vs DiT), so you'll need to adapt.

---

## Support

- **Cosmos 3 Documentation:** https://github.com/NVIDIA/cosmos
- **Technical Report:** https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf
- **Action Cookbook:** https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3/generator/action
- **Hugging Face:** https://huggingface.co/collections/nvidia/cosmos3

---

## Decision Matrix

|  | Stick with Cosmos 2.5 | Migrate to Cosmos 3 |
|--|----------------------|---------------------|
| **Pros** | - Code already written<br>- Understands your implementation | - Native ego-conditioning<br>- Better architecture<br>- Proven results<br>- Less custom code<br>- Production support |
| **Cons** | - Blurry results<br>- 2-4 weeks debugging<br>- No guarantee of success<br>- High maintenance | - Need to learn new API<br>- Agent conditioning not native (but can add) |
| **Risk** | High (may not converge) | Low (proven architecture) |
| **Effort** | 2-4 weeks (debug) | 2-3 days (test) + 1-2 weeks (post-train) |
| **Outcome** | Uncertain | High confidence |

**Recommendation:** Spend 2-3 days on Steps 1-6 to evaluate. If Cosmos 3 baseline is better (likely), commit to migration.
