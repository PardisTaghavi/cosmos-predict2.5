# Cosmos 2.5 vs Cosmos 3 Migration Analysis

## Executive Summary

**Recommendation: Migrate to Cosmos 3**

Your current Cosmos 2.5 implementation with custom ego/agent conditioning is experiencing blurry training results. After analyzing your codebase and comparing with Cosmos 3 capabilities, I strongly recommend migrating to Cosmos 3 for the following reasons:

1. **Native ego-conditioning support** - What you're trying to build is now a core feature
2. **Architecture designed for action conditioning** - Mixture-of-Transformers with dedicated towers
3. **Proven implementation** - Already handles forward/inverse dynamics
4. **Simpler integration** - Less custom code to maintain
5. **Better training dynamics** - Designed from the ground up for this use case

---

## Current Implementation Analysis (Cosmos 2.5)

### What You Built

You've added ~8,500 lines of custom code across 25 files to implement:

1. **Kinematic Conditioning System:**
   - `KinematicConditioner`: Encodes 3D kinematics and splats to spatial grid using Gaussian splatting
   - Tracks 32 agents with 18D features: (x,y,z, vx,vy,vz, ax,ay,az, l,w,h, yaw, tracking_id, 4 class labels)
   - Projects 3D world positions to 2D image coordinates
   - Temporal windowing to match VAE compression (4:1 ratio)

2. **DETR-based Kinematic Prediction:**
   - `DETRKinematicHead`: 130M+ parameter head for trajectory prediction
   - Hungarian matching for agent assignments
   - Predicts future agent trajectories from DiT features

3. **LoRA Fine-tuning:**
   - Rank-32 LoRA on attention layers (q,k,v,output) and MLP layers
   - Training on nuScenes dataset

### Identified Issues

Based on your code review, here are the likely causes of blurry training:

#### 1. **Conditioning Integration Problems**

```python
# cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py:1810
x_B_T_H_W_D = x_B_T_H_W_D + self.kin_scale * tau_kin
```

- `kin_scale` starts at 0.0 (initialized as `torch.zeros(1)`)
- Model must learn to incorporate kinematic conditioning from scratch
- During early training, kinematics are effectively ignored
- This can cause instability and poor conditioning integration

#### 2. **Loss Weighting Imbalance**

```python
# cosmos_predict2/experiments/base/nuscenes.py:117
kinematic_loss_weight=0.1,  # 10% of video loss
```

- Kinematic loss (130M params) weighted at 10% vs video reconstruction loss (2B model)
- Multi-task learning can cause conflicting gradients
- Kinematic head may dominate early training, degrading video quality

#### 3. **Complex Temporal Alignment**

```python
# cosmos_predict2/_src/predict2/networks/kinematic_conditioner.py:209
T_latent_expected = 1 + (T_pixel - 1) // temporal_window
```

- Manual temporal windowing to match VAE compression
- Fragile implementation that must exactly match VAE behavior
- Mismatch can cause spatial-temporal misalignment → blur

#### 4. **Gaussian Splatting Artifacts**

```python
# cosmos_predict2/_src/predict2/networks/kinematic_conditioner.py:74
sigma=0.1,  # Gaussian kernel bandwidth
```

- Fixed bandwidth for all agents/scenarios
- May cause over-smoothing (blur) or under-representation
- Learned projection from 3D→2D may not converge properly

#### 5. **Training Dynamics**

- 134.5M new parameters (kinematic_conditioner 4.2M + kinematic_head 130.3M)
- Training with lr=2^(-14.5) ≈ 3e-5 for 5000 iterations
- LoRA rank-32 may not have enough capacity for both tasks
- EMA updates may not properly track kinematic modules

---

## Cosmos 3 Capabilities

### Native Ego-Conditioning Architecture

Cosmos 3 was designed from the ground up for action-conditioned generation:

**Architecture:**
- **Mixture-of-Transformers** with two towers:
  - **Reasoner Tower**: Processes vision + text → understanding
  - **Generator Tower**: Diffusion-based generation conditioned on reasoner output
- **9D Ego-Pose Representation**: 3D translation + 6D continuous rotation
- **Action-Aware Input/Output Layers**: Modality-specific projections

**Supported Embodiments:**

| Embodiment | Action Space | Dimensionality | Your Use Case |
|------------|--------------|----------------|---------------|
| **Autonomous Vehicle** | Ego pose (9D) | 3D position + 6D rotation | ✅ **PERFECT FIT** |
| Egocentric Motion | Full body pose | 57D | ❌ Too complex |
| DROID Robot | End-effector + gripper | 10D | ❌ Different domain |
| Dual-Arm Robot | Dual end-effector | 20D | ❌ Different domain |
| Humanoid Robot | Full body | 29D | ❌ Different domain |

### What You Get Out-of-the-Box

1. **Forward Dynamics**: Generate future video conditioned on action sequences
2. **Inverse Dynamics**: Predict ego-motion trajectories from observed videos
3. **Policy Generation**: Predict action sequences from observations + task prompts
4. **Proven Training**: Already post-trained on AV datasets with action conditioning

### Example Usage (from Cosmos 3 docs)

```python
# Forward dynamics: ego-pose → future video
response = cosmos3_client.completions.create(
    model="nvidia/cosmos3-nano",
    modalities={
        "video": video_input,
        "action": ego_pose_sequence  # 9D per frame: [tx, ty, tz, r1, r2, r3, r4, r5, r6]
    },
    max_output_frames=60,
    fps=10
)

# Inverse dynamics: video → ego-motion trajectory
response = cosmos3_client.completions.create(
    model="nvidia/cosmos3-nano",
    modalities={"video": video_input},
    return_action=True  # Recover ego-motion
)
```

**Action Format for AV:**
- **Input**: 9D ego-pose per frame [tx, ty, tz, r1, r2, r3, r4, r5, r6]
- **Units**: Meters (position), continuous 6D rotation representation
- **Generation**: 60 frames @ 10 FPS for AV scenarios

---

## Migration Path to Cosmos 3

### Phase 1: Setup & Evaluation (Quick Win)

1. **Install Cosmos 3**
   ```bash
   # Add Cosmos Framework to your environment
   pip install cosmos-framework
   # Or use Diffusers/Transformers
   pip install transformers diffusers
   ```

2. **Convert Your NuScenes Data**
   - You already have video data in `/scratch/user/u.pt152369/WM/data/datasetWM`
   - Create ego-pose sequences from your existing kinematic data:
     ```python
     # Your data: [B, T, N, 18] with agent trajectories
     # Extract ego vehicle (track_id matching ego or closest agent)
     ego_positions = kinematics[:, :, ego_idx, 0:3]  # [B, T, 3]
     ego_yaw = kinematics[:, :, ego_idx, 12]  # [B, T]
     
     # Convert to 9D ego-pose: [tx, ty, tz, r1-r6]
     ego_pose_9d = convert_to_6d_rotation(ego_positions, ego_yaw)
     ```

3. **Run Initial Inference**
   ```python
   # Test forward dynamics with your data
   from cosmos_framework import Cosmos3Client
   
   client = Cosmos3Client()
   result = client.generate(
       video=nuscenes_video,
       action=ego_pose_sequence,
       embodiment="autonomous_vehicle"
   )
   ```

4. **Baseline Evaluation**
   - Compare video quality vs your current Cosmos 2.5 implementation
   - Evaluate ego-motion consistency
   - Measure inference speed

**Expected Result:** Better video quality immediately without any training

### Phase 2: Post-Training (If Needed)

1. **Prepare Action-Labeled Dataset**
   ```python
   # Format: Cosmos 3 expects action annotations per frame
   {
       "video": "path/to/video.mp4",
       "action": [[tx, ty, tz, r1, r2, r3, r4, r5, r6], ...],  # 9D per frame
       "text": "Driving through urban intersection...",
       "metadata": {...}
   }
   ```

2. **Choose Post-Training Recipe**
   - **Cosmos 3 Nano**: Fast inference (fractions of a second)
   - **Cosmos 3 Base**: Balanced quality/speed
   - **Cosmos 3 Super**: Highest physics accuracy (for your LoRA target)

3. **Run Action Post-Training**
   ```bash
   # Use Cosmos Framework's SFT recipe
   cosmos-train \
       --config configs/action_post_training.yaml \
       --data nuscenes_action_labeled \
       --embodiment autonomous_vehicle \
       --max_iter 5000
   ```

4. **Evaluation Metrics**
   - Video quality (FVD, SSIM, LPIPS)
   - Ego-motion accuracy (position error, rotation error)
   - Physics plausibility (collision detection, dynamics consistency)

### Phase 3: Agent Conditioning (Optional)

If you still need **agent trajectories** (not just ego), you have two options:

**Option A: Text-based agent descriptions**
```python
prompt = """The ego vehicle drives forward through an intersection.
A sedan passes on the left moving at 15 m/s.
A pedestrian crosses from the right at the crosswalk.
A traffic light transitions from yellow to red."""

result = cosmos3.generate(video=frame, text=prompt, embodiment="av")
```

**Option B: Custom agent conditioning (lighter than your current approach)**
- Keep Cosmos 3's ego-conditioning
- Add lightweight agent embedding (no DETR, simpler splatting)
- Use much smaller loss weight (0.01 vs 0.1)
- Let ego-conditioning handle camera motion, agents are secondary

---

## Cost-Benefit Analysis

### Sticking with Cosmos 2.5 Custom Implementation

**Effort Required:**
- ⏱️ **2-4 weeks**: Debug blurry training issue
  - Tune kin_scale initialization (start higher?)
  - Adjust kinematic_loss_weight (try 0.01-0.05)
  - Fix temporal windowing bugs
  - Tune Gaussian splatting bandwidth
  - Rebalance loss weights
  - Add gradient clipping for kinematic head
  - Possibly reduce kinematic head capacity (fewer layers)

**Risks:**
- May still not achieve good video quality (multi-task learning is hard)
- Custom code is brittle (breaks with Cosmos updates)
- 130M param kinematic head may always interfere with video quality
- No guarantee of convergence

**Maintenance:**
- Ongoing debugging as Cosmos 2.5 evolves
- 8,500+ lines of custom code to maintain
- Knowledge locked in your implementation

### Migrating to Cosmos 3

**Effort Required:**
- ⏱️ **2-5 days**: Phase 1 setup + evaluation
  - Extract ego-poses from your data
  - Run inference tests
  - Evaluate baseline quality
  
- ⏱️ **1-2 weeks**: Phase 2 post-training (if needed)
  - Format dataset with action labels
  - Run post-training
  - Evaluation

**Benefits:**
- ✅ Native ego-conditioning (proven to work)
- ✅ Better video quality (designed for this)
- ✅ Faster inference (optimized architecture)
- ✅ Community support & updates
- ✅ Extensible to other embodiments
- ✅ Production-ready (vLLM-Omni support)
- ✅ Less code to maintain (~8,500 lines → ~500 lines)

**Risks:**
- Learning new API/framework (minimal - well documented)
- Agent conditioning not native (but can be added if truly needed)

---

## Specific Recommendations

### If Your Primary Goal is Ego-Conditioning:

**→ Migrate to Cosmos 3 immediately**

Your custom implementation is reinventing what Cosmos 3 already provides. The blurry training is likely due to the fundamental difficulty of retrofitting action conditioning onto a model not designed for it.

**Why you'll succeed with Cosmos 3:**
1. Your data (nuScenes) is exactly the AV domain Cosmos 3 was designed for
2. Ego-pose extraction is straightforward from your kinematic data
3. 9D ego-pose is simpler than your current 32-agent × 18D approach
4. Forward/inverse dynamics match your likely use cases

### If You Absolutely Need Agent Conditioning:

**→ Start with Cosmos 3 ego, add lightweight agent layer**

Don't try to do everything at once. The proper architecture is:

```
Primary: Cosmos 3 ego-conditioning (handles camera motion, scene dynamics)
         ↓
Secondary: Lightweight agent embeddings (much simpler than your DETR head)
```

**Simplified agent conditioning approach:**
```python
class SimpleAgentConditioner(nn.Module):
    """Lightweight agent conditioning - no DETR, no complex splatting"""
    def __init__(self, model_channels: int):
        super().__init__()
        # Just encode agent positions + velocities → small embedding
        self.agent_encoder = nn.Linear(6, 256)  # pos + vel only
        self.agent_projector = nn.Linear(256, model_channels)
        
    def forward(self, agents_B_N_6):
        # agents_B_N_6: [B, max_agents, 6] = (x,y,z,vx,vy,vz) only
        emb = self.agent_encoder(agents_B_N_6)  # [B, N, 256]
        return self.agent_projector(emb)  # [B, N, model_channels]
```

- No DETR (130M params → ~0.5M params)
- No Gaussian splatting (just attend to agents)
- No kinematic prediction (not needed for generation)
- Loss weight: 0.01 (vs 0.1)

---

## Action Items

### Immediate (Today):

1. ⬜ Read Cosmos 3 documentation:
   - Technical report: https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf
   - Action cookbook: https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3/generator/action

2. ⬜ Extract ego-pose from 1 sample video:
   ```python
   # Quick script to verify data compatibility
   kinematics = load_h5_kinematics("sample.h5")
   ego_idx = find_ego_vehicle(kinematics)
   ego_pose_9d = convert_to_9d(kinematics[:, ego_idx, :])
   print(f"Ego pose shape: {ego_pose_9d.shape}")  # Should be [T, 9]
   ```

3. ⬜ Decide: Go/No-Go on Cosmos 3 migration

### Week 1 (If Go):

1. ⬜ Install Cosmos 3 framework
2. ⬜ Convert 10-20 sample videos with ego-poses
3. ⬜ Run inference tests
4. ⬜ Compare video quality vs current Cosmos 2.5
5. ⬜ Measure inference latency

### Week 2-3 (If Quality is Good):

1. ⬜ Convert full nuScenes dataset with ego-pose labels
2. ⬜ Run post-training on Cosmos 3 Nano or Base
3. ⬜ Evaluate on validation set
4. ⬜ Decide if agent conditioning is still needed

---

## Conclusion

**The blurry training in your Cosmos 2.5 implementation is a symptom of a deeper issue**: retrofitting action conditioning onto a model not designed for it is fundamentally difficult. You've built a sophisticated system with Gaussian splatting, DETR prediction, and multi-task training, but the architecture is fighting you.

**Cosmos 3 solves this by design**. It's not just "adding a feature" - the entire Mixture-of-Transformers architecture was built ground-up for action-conditioned physical AI. Your use case (autonomous vehicles with ego-conditioning) is literally the primary use case in their documentation.

**My recommendation**: Spend 2-3 days on Phase 1 to evaluate Cosmos 3. If the baseline inference quality is better than your current blurry results (which I expect it will be), commit to the migration. You'll likely finish the full migration (including post-training) faster than debugging your current implementation, with better results and less maintenance burden.

**The work you've done isn't wasted** - you've learned what's needed for action-conditioned generation. That knowledge will make you much more effective using Cosmos 3's capabilities. But trying to build it from scratch on Cosmos 2.5 is like building a car engine from bicycle parts - technically possible, but why when a purpose-built engine is available?

---

## Appendix: Debugging Cosmos 2.5 (If You Must Stay)

If you decide to continue with Cosmos 2.5 despite my recommendation, here are targeted fixes:

### Fix 1: kin_scale Initialization
```python
# cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py:319
# Change:
actual_model.kin_scale = nn.Parameter(torch.zeros(1, device=device))
# To:
actual_model.kin_scale = nn.Parameter(torch.ones(1, device=device) * 0.1)
```

### Fix 2: Reduce kinematic_loss_weight
```python
# cosmos_predict2/experiments/base/nuscenes.py:117
kinematic_loss_weight=0.01,  # Down from 0.1
```

### Fix 3: Add gradient clipping
```python
# In training loop, before optimizer step:
torch.nn.utils.clip_grad_norm_(
    list(model.net.kinematic_conditioner.parameters()) +
    list(model.net.kinematic_head.parameters()),
    max_norm=1.0
)
```

### Fix 4: Freeze kinematic head initially
```python
# First 1000 iterations: only train video generation + conditioner
for param in model.net.kinematic_head.parameters():
    param.requires_grad = False
    
# After 1000 iter: unfreeze kinematic head
```

### Fix 5: Adaptive Gaussian bandwidth
```python
# cosmos_predict2/_src/predict2/networks/kinematic_conditioner.py
# Make sigma learnable:
self.log_sigma = nn.Parameter(torch.log(torch.tensor(0.1)))
# In forward:
sigma = torch.exp(self.log_sigma)
```

**Expected improvement**: Might reduce blur, but no guarantee of convergence. Still recommend Cosmos 3.
