"""
Flow Matching Visualization
Visualizes each step of the flow matching mechanism to understand how it works.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os
try:
    from scipy.spatial import ConvexHull
except ImportError:
    ConvexHull = None


def ensure_rgb(image_array):
    """
    Ensure image array is in RGB format (H, W, 3) with channels in RGB order.
    
    Args:
        image_array: numpy array of shape (H, W) or (H, W, C)
    
    Returns:
        RGB image array of shape (H, W, 3)
    """
    if len(image_array.shape) == 2:
        # Grayscale -> RGB
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[-1] == 1:
        # Single channel -> RGB
        image_array = np.repeat(image_array, 3, axis=-1)
    elif image_array.shape[-1] == 4:
        # RGBA -> RGB (drop alpha)
        image_array = image_array[:, :, :3]
    elif image_array.shape[-1] > 4:
        # Multiple channels -> RGB (take first 3)
        image_array = image_array[:, :, :3]
    
    assert image_array.shape[-1] == 3, f"Expected 3 channels (RGB), got {image_array.shape[-1]}"
    return image_array


def visualize_step1_endpoints(x_data, epsilon, save_path="flow_matching_step1_endpoints.png"):
    """
    Step 1: Visualize the two endpoints - real data and noise
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(np.clip(x_data, 0, 1))
    axes[0].set_title("Endpoint A: Real Data (x)\nt = 0", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(epsilon, 0, 1))
    axes[1].set_title("Endpoint B: Pure Noise (ε)\nt = 1", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle("Step 1: Two Endpoints", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_step2_linear_path(x_data, epsilon, t_values, save_path="flow_matching_step2_path.png"):
    """
    Step 2: Visualize the linear interpolation path
    x_t = (1-t)x + t*ε
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    fig, axes = plt.subplots(1, len(t_values), figsize=(20, 4))
    
    for i, t in enumerate(t_values):
        x_t = (1 - t) * x_data + t * epsilon
        axes[i].imshow(np.clip(x_t, 0, 1))
        axes[i].set_title(f't = {t:.2f}\nx_t = (1-{t:.2f})x + {t:.2f}ε', 
                         fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle("Step 2: Linear Interpolation Path\nx_t = (1-t)x + t·ε", 
                 fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_step3_velocity(x_data, epsilon, t_values, save_path="flow_matching_step3_velocity.png"):
    """
    Step 3: Visualize the ground-truth velocity
    v_t = ε - x (constant for linear path)
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    fig, axes = plt.subplots(2, len(t_values), figsize=(20, 8))
    
    v_ground_truth = epsilon - x_data
    
    for i, t in enumerate(t_values):
        x_t = (1 - t) * x_data + t * epsilon
        
        # Top row: x_t at different times
        axes[0, i].imshow(np.clip(x_t, 0, 1))
        axes[0, i].set_title(f't = {t:.2f}\nCurrent state x_t', fontsize=11)
        axes[0, i].axis('off')
        
        # Bottom row: velocity field (constant!)
        # Show velocity as a visualization
        v_vis = np.clip((v_ground_truth + 1) / 2, 0, 1)  # Normalize for display
        axes[1, i].imshow(v_vis)
        axes[1, i].set_title(f't = {t:.2f}\nv_t = ε - x\n(constant!)', 
                            fontsize=11, fontweight='bold', color='green')
        axes[1, i].axis('off')
    
    plt.suptitle("Step 3: Ground-Truth Velocity\nv_t = ε - x (constant along path)", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_step4_training_objective(x_data, epsilon, t_samples, save_path="flow_matching_step4_training.png"):
    """
    Step 4: Visualize the training objective
    Model learns to predict velocity: u(x_t, t, c) ≈ v_t = ε - x
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    fig = plt.figure(figsize=(16, 10))
    
    # Sample a few random t values
    n_samples = min(6, len(t_samples))
    selected_indices = np.random.choice(len(t_samples), n_samples, replace=False)
    selected_t = t_samples[selected_indices]
    
    gs = fig.add_gridspec(3, n_samples, hspace=0.3, wspace=0.1)
    
    for i, t in enumerate(selected_t):
        x_t = (1 - t) * x_data + t * epsilon
        v_target = epsilon - x_data
        
        # Row 1: Input to model
        ax1 = fig.add_subplot(gs[0, i])
        ax1.imshow(np.clip(x_t, 0, 1))
        ax1.set_title(f'Input: x_t\n(t = {t:.3f})', fontsize=10)
        ax1.axis('off')
        
        # Row 2: Target velocity
        ax2 = fig.add_subplot(gs[1, i])
        v_vis = np.clip((v_target + 1) / 2, 0, 1)
        ax2.imshow(v_vis)
        ax2.set_title(f'Target: v_t = ε - x', fontsize=10, fontweight='bold', color='green')
        ax2.axis('off')
        
        # Row 3: Loss visualization
        ax3 = fig.add_subplot(gs[2, i])
        # Simulate a "prediction" (in real training, this comes from neural net)
        # For visualization, show what the model should learn
        v_pred_vis = np.clip((v_target + 1) / 2, 0, 1)  # Perfect prediction
        ax3.imshow(v_pred_vis)
        ax3.set_title(f'Model learns:\nu(x_t, t) → v_t', fontsize=10, fontweight='bold', color='blue')
        ax3.axis('off')
    
    plt.suptitle("Step 4: Training Objective\nL = E[||u(x_t, t, c) - v_t||²]\nwhere v_t = ε - x", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_step5_generation_process(x_data, epsilon, n_steps=10, save_path="flow_matching_step5_generation.png"):
    """
    Step 5: Visualize the generation process (ODE integration)
    Start from noise, integrate backward: dx/dt = u(x, t)
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    fig, axes = plt.subplots(2, n_steps, figsize=(20, 8))
    
    # Simulate generation: start from noise, move toward data
    # In real generation, we'd use the learned model u(x, t)
    # Here we use the ground truth velocity for visualization
    v_field = epsilon - x_data
    
    # Generate path from noise to data (backward integration)
    t_gen = np.linspace(1.0, 0.0, n_steps)  # From noise to data
    
    for i, t in enumerate(t_gen):
        # In real generation: x_{t+dt} = x_t + dt * u(x_t, t)
        # For visualization, use exact path
        x_t = (1 - t) * x_data + t * epsilon
        
        # Top row: Generated samples
        axes[0, i].imshow(np.clip(x_t, 0, 1))
        axes[0, i].set_title(f't = {t:.2f}', fontsize=10)
        axes[0, i].axis('off')
        
        # Bottom row: Velocity direction
        axes[1, i].imshow(np.clip((v_field + 1) / 2, 0, 1))
        axes[1, i].set_title(f'Velocity field\n(learned)', fontsize=10)
        axes[1, i].axis('off')
    
    axes[0, 0].set_title('Start: Noise\nt = 1.0', fontsize=10, fontweight='bold', color='red')
    axes[0, -1].set_title('End: Data\nt = 0.0', fontsize=10, fontweight='bold', color='green')
    
    plt.suptitle("Step 5: Generation Process\nODE Integration: dx/dt = u(x, t)\nFrom noise → data", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_step7_time_distributions(save_path="flow_matching_step7_time_dist.png"):
    """
    Step 7: Visualize time sampling distributions
    Uniform vs Logit-normal
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    n_samples = 10000
    
    # Uniform distribution
    t_uniform = np.random.rand(n_samples)
    
    # Logit-normal distribution
    t_logitnormal = 1 / (1 + np.exp(-np.random.randn(n_samples)))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Uniform: histogram
    axes[0, 0].hist(t_uniform, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Uniform density = 1')
    axes[0, 0].set_xlabel('t', fontsize=12)
    axes[0, 0].set_ylabel('Density', fontsize=12)
    axes[0, 0].set_title('Uniform Distribution\nP(t) = 1 for t ∈ [0,1]', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Uniform: CDF
    sorted_uniform = np.sort(t_uniform)
    axes[0, 1].plot(sorted_uniform, np.linspace(0, 1, len(sorted_uniform)), 
                    linewidth=2, color='blue')
    axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal uniform')
    axes[0, 1].set_xlabel('t', fontsize=12)
    axes[0, 1].set_ylabel('CDF', fontsize=12)
    axes[0, 1].set_title('Uniform CDF', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Logit-normal: histogram
    axes[1, 0].hist(t_logitnormal, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('t', fontsize=12)
    axes[1, 0].set_ylabel('Density', fontsize=12)
    axes[1, 0].set_title('Logit-Normal Distribution\nBiased toward extremes', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Logit-normal: CDF
    sorted_logit = np.sort(t_logitnormal)
    axes[1, 1].plot(sorted_logit, np.linspace(0, 1, len(sorted_logit)), 
                    linewidth=2, color='green')
    axes[1, 1].set_xlabel('t', fontsize=12)
    axes[1, 1].set_ylabel('CDF', fontsize=13)
    axes[1, 1].set_title('Logit-Normal CDF', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle("Step 7: Time Sampling Distributions", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_step8_beta_shift(save_path="flow_matching_step8_beta_shift.png"):
    """
    Step 8: Visualize the β shift effect
    t_s = βt / (1 + (β-1)t)
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    t = np.linspace(0, 1, 1000)
    beta_values = [1.0, 1.5, 2.0, 3.0, 5.0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Transformation curves
    for beta in beta_values:
        t_s = (beta * t) / (1 + (beta - 1) * t)
        label = f'β = {beta}' + (' (no shift)' if beta == 1 else '')
        axes[0].plot(t, t_s, linewidth=2, label=label)
    
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Identity (β=1)')
    axes[0].set_xlabel('Original t', fontsize=12)
    axes[0].set_ylabel('Shifted t_s', fontsize=12)
    axes[0].set_title('β Shift Transformation\nt_s = βt / (1 + (β-1)t)', 
                     fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # Plot 2: Effect on distribution (histogram)
    n_samples = 10000
    t_original = np.random.rand(n_samples)
    
    for beta in [1.0, 2.0, 5.0]:
        t_shifted = (beta * t_original) / (1 + (beta - 1) * t_original)
        axes[1].hist(t_shifted, bins=50, density=True, alpha=0.5, 
                    label=f'β = {beta}', histtype='step', linewidth=2)
    
    axes[1].set_xlabel('t_s', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Effect on Training Distribution\nHigher β → more high-noise samples', 
                     fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("Step 8: β Shift for High-Noise Training", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_2d_flow_field(save_path="flow_matching_2d_flow_field.png"):
    """
    Visualize flow matching in 2D space to show the "wind field" concept
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    # Create a 2D grid
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)
    
    # Define some "data points" (targets)
    data_points = np.array([[-2, -2], [2, 2], [-2, 2], [2, -2]])
    
    # For each point in grid, compute average velocity toward nearest data point
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(len(x)):
        for j in range(len(y)):
            point = np.array([X[j, i], Y[j, i]])
            # Find nearest data point
            distances = np.sum((data_points - point)**2, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_data = data_points[nearest_idx]
            
            # Velocity points toward data (from noise)
            # In flow matching: v = data - noise, so we want to move toward data
            velocity = nearest_data - point
            U[j, i] = velocity[0]
            V[j, i] = velocity[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Vector field
    axes[0].quiver(X, Y, U, V, scale=15, width=0.003, alpha=0.6, color='blue')
    axes[0].scatter(data_points[:, 0], data_points[:, 1], 
                   s=200, c='green', marker='*', label='Data points', zorder=5)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].set_title('Learned Velocity Field\n"Wind Field" Concept', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # Plot 2: Streamlines showing paths
    axes[1].streamplot(X, Y, U, V, density=1.5, color='blue')
    axes[1].scatter(data_points[:, 0], data_points[:, 1], 
                   s=200, c='green', marker='*', label='Data points', zorder=5)
    
    # Draw some example paths
    for data_point in data_points:
        # Start from random noise point
        noise_point = np.random.randn(2) * 2
        # Simulate path
        path = [noise_point]
        current = noise_point.copy()
        dt = 0.1
        for _ in range(30):
            # Find velocity at current point
            dists = np.sum((data_points - current)**2, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_data = data_points[nearest_idx]
            velocity = nearest_data - current
            current = current + dt * velocity
            path.append(current.copy())
            if np.linalg.norm(current - nearest_data) < 0.1:
                break
        path = np.array(path)
        axes[1].plot(path[:, 0], path[:, 1], 'r-', linewidth=2, alpha=0.7, label='Example path' if data_point is data_points[0] else '')
    
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].set_title('Flow Paths\nFollowing velocity field from noise → data', 
                      fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    plt.suptitle("Geometric Intuition: Flow Matching as Vector Field", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_multiple_paths(x_data_list, epsilon_list, save_path="flow_matching_multiple_paths.png"):
    """
    Visualize multiple paths from different data points
    Shows how the model learns from many examples
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    n_paths = min(4, len(x_data_list))
    t_values = np.linspace(0, 1, 5)
    
    fig, axes = plt.subplots(n_paths, len(t_values), figsize=(20, 4*n_paths))
    
    if n_paths == 1:
        axes = axes.reshape(1, -1)
    
    for path_idx in range(n_paths):
        x_data = x_data_list[path_idx]
        epsilon = epsilon_list[path_idx]
        
        for t_idx, t in enumerate(t_values):
            x_t = (1 - t) * x_data + t * epsilon
            axes[path_idx, t_idx].imshow(np.clip(x_t, 0, 1))
            if path_idx == 0:
                axes[path_idx, t_idx].set_title(f't = {t:.2f}', fontsize=11)
            if t_idx == 0:
                axes[path_idx, t_idx].set_ylabel(f'Path {path_idx+1}', fontsize=11, fontweight='bold')
            axes[path_idx, t_idx].axis('off')
    
    plt.suptitle("Multiple Training Paths\nEach path: one (x, ε) pair\nModel learns from many such examples", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """
    Main function to generate all visualizations
    """
    print("=" * 60)
    print("Flow Matching Visualization")
    print("=" * 60)
    
    # Load or create sample image
    img_path = "/Users/ptgh/Downloads/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"
    
    if os.path.exists(img_path):
        img = Image.open(img_path)
        # Convert to RGB explicitly (handles RGBA, L, P modes, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        x_data = np.array(img).astype(np.float32) / 255.0
        print(f"Loaded image: {img_path} (mode: RGB)")
    else:
        # Create synthetic data if image not found
        print("Image not found, creating synthetic data...")
        x_data = np.random.rand(256, 256, 3).astype(np.float32)
    
    # Ensure RGB format (not BGR) - shape should be (H, W, 3) with channels in RGB order
    x_data = ensure_rgb(x_data)
    
    # Sample noise
    epsilon = np.random.randn(*x_data.shape).astype(np.float32)
    
    # Time values for visualization
    t_values = np.linspace(0.0, 1.0, 5)
    t_samples = np.random.rand(100)  # For training visualization
    
    print("\nGenerating visualizations...")
    
    # Step 1: Endpoints
    visualize_step1_endpoints(x_data, epsilon)
    
    # Step 2: Linear path
    visualize_step2_linear_path(x_data, epsilon, t_values)
    
    # Step 3: Velocity
    visualize_step3_velocity(x_data, epsilon, t_values)
    
    # Step 4: Training objective
    visualize_step4_training_objective(x_data, epsilon, t_samples)
    
    # Step 5: Generation process
    visualize_step5_generation_process(x_data, epsilon, n_steps=10)
    
    # Step 7: Time distributions
    visualize_step7_time_distributions()
    
    # Step 8: Beta shift
    visualize_step8_beta_shift()
    
    # 2D flow field
    visualize_2d_flow_field()
    
    # Multiple paths (create multiple data-noise pairs)
    x_data_list = [x_data] * 4
    epsilon_list = [np.random.randn(*x_data.shape).astype(np.float32) for _ in range(4)]
    visualize_multiple_paths(x_data_list, epsilon_list)
    
    # Generation animation: noise → data
    print("\nGenerating noise-to-data animation...")
    visualize_noise_to_data_animation(x_data, epsilon)
    visualize_point_flow_animation(x_data, epsilon)
    
    # Latent space visualization
    print("\nGenerating latent space visualizations...")
    visualize_latent_space_flow_matching(x_data, epsilon)
    
    # Two moons dataset animation
    print("\nGenerating two moons dataset animation...")
    visualize_two_moons_flow_matching()
    
    print("\n" + "=" * 60)
    print("All visualizations complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - flow_matching_step1_endpoints.png")
    print("  - flow_matching_step2_path.png")
    print("  - flow_matching_step3_velocity.png")
    print("  - flow_matching_step4_training.png")
    print("  - flow_matching_step5_generation.png")
    print("  - flow_matching_step7_time_dist.png")
    print("  - flow_matching_step8_beta_shift.png")
    print("  - flow_matching_2d_flow_field.png")
    print("  - flow_matching_multiple_paths.png")
    print("  - flow_matching_noise_to_data.gif")
    print("  - flow_matching_point_flow.gif")
    print("  - flow_matching_latent_space.png")
    print("  - flow_matching_latent_optimization.png")
    print("  - flow_matching_latent_velocity_field.png")
    print("  - flow_matching_two_moons.gif")


def visualize_two_moons_flow_matching(save_path="flow_matching_two_moons.gif", n_steps=60):
    """
    Animate flow matching on the two-moons dataset.
    Shows how points from a Gaussian distribution are pushed by the velocity field
    towards the two-moons data manifold over time.
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    def generate_two_moons(n_samples=200, noise=0.05):
        """Generate two-moons dataset: convex up and convex down"""
        np.random.seed(42)
        n_samples_per_moon = n_samples // 2
        
        # First moon (convex up - upper crescent)
        angle1 = np.linspace(0, np.pi, n_samples_per_moon)
        x1 = np.cos(angle1)
        y1 = np.sin(angle1) + 0.5  # Shift up
        moon1 = np.column_stack([x1, y1]) + np.random.randn(n_samples_per_moon, 2) * noise
        
        # Second moon (convex down - lower crescent, flipped)
        angle2 = np.linspace(0, np.pi, n_samples_per_moon)
        x2 = np.cos(angle2)
        y2 = -np.sin(angle2) - 0.5  # Flipped and shifted down
        moon2 = np.column_stack([x2, y2]) + np.random.randn(n_samples_per_moon, 2) * noise
        
        return np.vstack([moon1, moon2])
    
    def find_nearest_point_on_manifold(point, manifold_points):
        """Find nearest point on the two-moons manifold"""
        distances = np.linalg.norm(manifold_points - point, axis=1)
        nearest_idx = np.argmin(distances)
        return manifold_points[nearest_idx]
    
    # Generate two-moons manifold (orange curve)
    manifold_points = generate_two_moons(n_samples=200, noise=0.05)
    
    # Generate initial points from Gaussian distribution (black crosses)
    np.random.seed(123)
    n_points = 100
    initial_points = np.random.randn(n_points, 2) * 1.5
    
    # Create grid for velocity field visualization
    x_range = np.linspace(-2.5, 2.5, 20)
    y_range = np.linspace(-2, 2, 16)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Time progression: from t=1 (initial Gaussian) to t=0 (converged to manifold)
    t_values = np.linspace(1.0, 0.0, n_steps)
    
    fig = plt.figure(figsize=(14, 10))
    
    def animate(frame):
        fig.clear()
        gs = fig.add_gridspec(2, 1, hspace=0.3, height_ratios=[3, 1])
        
        step_idx = min(int(frame * n_steps / 50), n_steps - 1)
        t_current = t_values[step_idx]
        
        # Interpolate points: start from Gaussian, move towards manifold
        current_points = []
        for point in initial_points:
            # Find target on manifold
            target = find_nearest_point_on_manifold(point, manifold_points)
            # Interpolate: x_t = t * initial + (1-t) * target
            # But for flow matching, we go from noise (initial) to data (target)
            # So: x_t = t * initial + (1-t) * target
            interpolated = t_current * point + (1 - t_current) * target
            current_points.append(interpolated)
        current_points = np.array(current_points)
        
        # Main plot: Show points and velocity field
        ax_main = fig.add_subplot(gs[0])
        
        # Compute velocity field on grid
        # Start with random arrows, gradually improve to point toward moons
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        # Initialize random velocity field (only once, reuse seed)
        np.random.seed(999)
        U_random = np.random.randn(*X.shape) * 0.3
        V_random = np.random.randn(*Y.shape) * 0.3
        
        # Compute correct velocity field (pointing toward nearest moon)
        U_correct = np.zeros_like(X)
        V_correct = np.zeros_like(Y)
        
        for i in range(len(x_range)):
            for j in range(len(y_range)):
                grid_point = np.array([X[j, i], Y[j, i]])
                # Find nearest point on manifold
                nearest_manifold = find_nearest_point_on_manifold(grid_point, manifold_points)
                # Velocity points towards manifold
                velocity = nearest_manifold - grid_point
                # Normalize and scale
                distance = np.linalg.norm(velocity)
                if distance > 0.01:
                    velocity = velocity / distance * min(distance, 0.5)
                U_correct[j, i] = velocity[0]
                V_correct[j, i] = velocity[1]
        
        # Interpolate between random and correct: start random (t=1), end correct (t=0)
        # As t decreases, arrows become more correct
        learning_progress = 1 - t_current  # 0 to 1 as we go from t=1 to t=0
        U = (1 - learning_progress) * U_random + learning_progress * U_correct
        V = (1 - learning_progress) * V_random + learning_progress * V_correct
        
        # Draw velocity field (blue arrows)
        ax_main.quiver(X, Y, U, V, scale=15, width=0.003, alpha=0.6, color='blue', zorder=1)
        
        # Draw two-moons manifold (orange curve)
        # Sort points for smooth curve - convex up and convex down
        moon1_points = manifold_points[:len(manifold_points)//2]
        moon2_points = manifold_points[len(manifold_points)//2:]
        
        # Sort by angle for smooth plotting
        def sort_by_angle(points):
            angles = np.arctan2(points[:, 1], points[:, 0])
            sorted_idx = np.argsort(angles)
            return points[sorted_idx]
        
        moon1_sorted = sort_by_angle(moon1_points)
        moon2_sorted = sort_by_angle(moon2_points)
        
        # Plot first moon (convex up)
        ax_main.plot(moon1_sorted[:, 0], moon1_sorted[:, 1], 
                    'o-', color='orange', linewidth=4, markersize=8, 
                    label='Data Manifold (Two Moons)', zorder=3, alpha=0.8)
        
        # Plot second moon (convex down)
        ax_main.plot(moon2_sorted[:, 0], moon2_sorted[:, 1], 
                    'o-', color='orange', linewidth=4, markersize=8, 
                    zorder=3, alpha=0.8)
        
        # Draw current points (black crosses)
        ax_main.scatter(current_points[:, 0], current_points[:, 1], 
                       c='black', s=150, marker='x', linewidths=2,
                       label='Points (flowing)', zorder=4, alpha=0.8)
        
        ax_main.set_xlabel('x', fontsize=14)
        ax_main.set_ylabel('y', fontsize=14)
        arrow_status = "Random" if learning_progress < 0.2 else "Learning" if learning_progress < 0.8 else "Correct"
        ax_main.set_title(f'Flow Matching on Two Moons Dataset\n'
                         f't = {t_current:.2f} | Arrows: {arrow_status} '
                         f'({"Initial (Gaussian)" if t_current > 0.9 else "Converged" if t_current < 0.1 else "Transition"})',
                         fontsize=15, fontweight='bold')
        ax_main.legend(loc='upper right', fontsize=12)
        ax_main.grid(True, alpha=0.3)
        ax_main.set_aspect('equal')
        ax_main.set_xlim([-2.5, 2.5])
        ax_main.set_ylim([-2, 2])
        
        # Progress bar
        ax_prog = fig.add_subplot(gs[1])
        progress = 1 - t_current
        ax_prog.barh([0], [progress], color='green', alpha=0.7, height=0.5)
        ax_prog.barh([0], [1-progress], left=[progress], color='gray', alpha=0.3, height=0.5)
        ax_prog.set_xlim([0, 1])
        ax_prog.set_ylim([-0.5, 0.5])
        ax_prog.set_xlabel('Progress: Gaussian → Two Moons', fontsize=12, fontweight='bold')
        ax_prog.set_title(f'{progress*100:.1f}% Complete', fontsize=13, fontweight='bold')
        ax_prog.set_yticks([])
        ax_prog.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Flow Matching: Vector Field Pushes Points Towards Data Manifold', 
                    fontsize=16, fontweight='bold', y=0.98)
    
    try:
        anim = FuncAnimation(fig, animate, frames=50, interval=100, repeat=True)
        anim.save(save_path, writer=PillowWriter(fps=10))
        plt.close()
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Warning: Could not create two moons animation: {e}")
        print("Skipping animation (requires PillowWriter)")
        plt.close()


def visualize_latent_space_flow_matching(x_data, epsilon, save_paths=None):
    """
    Simplified visualization of flow matching in latent space.
    Shows: noise distribution, data distribution, velocity vectors, and optimization goal.
    """
    if save_paths is None:
        save_paths = {
            'overview': 'flow_matching_latent_space.png',
            'optimization': 'flow_matching_latent_optimization.png',
            'velocity': 'flow_matching_latent_velocity_field.png'
        }
    
    # Check if all visualizations already exist
    all_exist = all(os.path.exists(path) for path in save_paths.values())
    if all_exist:
        print(f"Skipping latent space visualizations (all already exist)")
        return
    
    # Simulate latent space: project high-dim data to 2D for visualization
    # In reality, latents are high-dimensional (e.g., 16 channels × H × W)
    # Here we create a simplified 2D representation
    
    # Create multiple data points and noise points in "latent space"
    n_data_points = 20
    n_noise_points = 20
    
    # Simulate data manifold (clustered structure)
    np.random.seed(42)
    data_center = np.array([0.0, 0.0])
    data_points = np.random.randn(n_data_points, 2) * 0.5 + data_center
    
    # Simulate noise distribution (Gaussian, spread out)
    noise_points = np.random.randn(n_noise_points, 2) * 2.0
    
    # Create overview visualization
    visualize_latent_space_overview(data_points, noise_points, save_paths['overview'])
    
    # Create optimization visualization
    visualize_latent_optimization_process(data_points, noise_points, save_paths['optimization'])
    
    # Create velocity field visualization
    visualize_latent_velocity_field(data_points, noise_points, save_paths['velocity'])


def visualize_latent_space_overview(data_points, noise_points, save_path="flow_matching_latent_space.png"):
    """
    Overview of flow matching in latent space: distributions and flow paths
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Noise and Data Distributions
    ax1 = axes[0, 0]
    ax1.scatter(noise_points[:, 0], noise_points[:, 1], 
               c='red', alpha=0.6, s=100, label='Noise Distribution\nN(0, I)', marker='x')
    ax1.scatter(data_points[:, 0], data_points[:, 1], 
               c='green', alpha=0.6, s=100, label='Data Distribution\n(Manifold)', marker='o')
    
    # Draw flow paths from noise to nearest data point
    for noise_pt in noise_points[:5]:  # Show 5 example paths
        nearest_data_idx = np.argmin(np.linalg.norm(data_points - noise_pt, axis=1))
        nearest_data = data_points[nearest_data_idx]
        ax1.plot([noise_pt[0], nearest_data[0]], 
                [noise_pt[1], nearest_data[1]], 
                'b--', alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('Latent Dimension 1', fontsize=12)
    ax1.set_ylabel('Latent Dimension 2', fontsize=12)
    ax1.set_title('Latent Space: Noise → Data Flow\n'
                 'Goal: Learn velocity field to transport noise to data',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Plot 2: Velocity Vectors
    ax2 = axes[0, 1]
    # Show velocity vectors: v = data - noise
    sample_pairs = list(zip(noise_points[:8], data_points[:8]))
    
    for noise_pt, data_pt in sample_pairs:
        # Show interpolated point
        t = 0.5
        interp_pt = t * noise_pt + (1 - t) * data_pt
        velocity = data_pt - noise_pt
        
        ax2.scatter([interp_pt[0]], [interp_pt[1]], 
                   c='blue', s=50, alpha=0.7, zorder=3)
        ax2.arrow(interp_pt[0], interp_pt[1],
                 velocity[0] * 0.3, velocity[1] * 0.3,
                 head_width=0.1, head_length=0.08,
                 fc='blue', ec='blue', alpha=0.6, linewidth=2)
    
    ax2.scatter(noise_points[:, 0], noise_points[:, 1], 
               c='red', alpha=0.3, s=50, marker='x', label='Noise')
    ax2.scatter(data_points[:, 0], data_points[:, 1], 
               c='green', alpha=0.3, s=50, marker='o', label='Data')
    ax2.set_xlabel('Latent Dimension 1', fontsize=12)
    ax2.set_ylabel('Latent Dimension 2', fontsize=12)
    ax2.set_title('Velocity Vectors at Interpolated Points\n'
                 'v_t = ε - x (constant along path)',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Plot 3: What is Learned?
    ax3 = axes[1, 0]
    # Show learned velocity field as a continuous function
    x_range = np.linspace(-3, 3, 30)
    y_range = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Compute velocity field: for each point, find nearest data and compute v = data - point
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            point = np.array([X[j, i], Y[j, i]])
            # Find nearest data point
            distances = np.linalg.norm(data_points - point, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_data = data_points[nearest_idx]
            # Velocity points toward data
            velocity = nearest_data - point
            U[j, i] = velocity[0]
            V[j, i] = velocity[1]
    
    # Normalize for visualization
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = U / (magnitude + 1e-8)
    V_norm = V / (magnitude + 1e-8)
    
    ax3.quiver(X, Y, U_norm, V_norm, magnitude, 
              cmap='viridis', scale=20, width=0.003, alpha=0.7)
    ax3.scatter(data_points[:, 0], data_points[:, 1], 
               c='green', s=150, marker='*', label='Data Points', zorder=5)
    ax3.set_xlabel('Latent Dimension 1', fontsize=12)
    ax3.set_ylabel('Latent Dimension 2', fontsize=12)
    ax3.set_title('Learned Velocity Field u(x, t)\n'
                 'Model predicts velocity at any point in latent space',
                 fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Plot 4: Optimization Goal
    ax4 = axes[1, 1]
    # Show the loss landscape
    # Loss = ||u(x_t, t) - v_t||² where v_t = ε - x
    
    # Simulate loss at different points
    test_points = np.random.randn(15, 2) * 2
    losses = []
    
    for test_pt in test_points:
        # Find nearest data and noise pair
        noise_distances = np.linalg.norm(noise_points - test_pt, axis=1)
        data_distances = np.linalg.norm(data_points - test_pt, axis=1)
        
        nearest_noise_idx = np.argmin(noise_distances)
        nearest_data_idx = np.argmin(data_distances)
        
        noise_pt = noise_points[nearest_noise_idx]
        data_pt = data_points[nearest_data_idx]
        
        # True velocity
        v_true = data_pt - noise_pt
        
        # Simulate prediction error (loss)
        # Perfect model: error = 0, imperfect: error > 0
        prediction_error = np.random.rand() * 0.5  # Simulated
        loss = prediction_error ** 2
        losses.append(loss)
    
    scatter = ax4.scatter(test_points[:, 0], test_points[:, 1], 
                         c=losses, s=200, cmap='Reds', alpha=0.7,
                         edgecolors='black', linewidths=1)
    ax4.scatter(data_points[:, 0], data_points[:, 1], 
               c='green', s=100, marker='*', label='Data', zorder=5)
    ax4.set_xlabel('Latent Dimension 1', fontsize=12)
    ax4.set_ylabel('Latent Dimension 2', fontsize=12)
    ax4.set_title('Optimization Goal: Minimize Loss\n'
                 'L(θ) = E[||u(x_t, t; θ) - v_t||²]\n'
                 'Red = high loss, White = low loss',
                 fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    plt.colorbar(scatter, ax=ax4, label='Loss')
    
    plt.suptitle('Flow Matching in Latent Space: Overview', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_latent_optimization_process(data_points, noise_points, save_path="flow_matching_latent_optimization.png"):
    """
    Show how optimization improves the velocity field predictions
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Simulate optimization stages
    stages = ['Initial (Random)', 'Mid Training', 'Converged']
    prediction_errors = [0.8, 0.3, 0.05]  # Decreasing error
    
    for stage_idx, (stage_name, error) in enumerate(zip(stages, prediction_errors)):
        # Before optimization (row 0)
        ax_before = axes[0, stage_idx]
        
        # Show velocity predictions with error
        sample_noise = noise_points[0]
        sample_data = data_points[0]
        true_velocity = sample_data - sample_noise
        
        # Simulate prediction with error
        prediction = true_velocity + np.random.randn(2) * error
        prediction = prediction / (np.linalg.norm(prediction) + 1e-8) * np.linalg.norm(true_velocity)
        
        # Show interpolated point
        t = 0.5
        interp_pt = t * sample_noise + (1 - t) * sample_data
        
        ax_before.scatter([sample_noise[0]], [sample_noise[1]], 
                         c='red', s=200, marker='x', label='Noise', zorder=5)
        ax_before.scatter([sample_data[0]], [sample_data[1]], 
                         c='green', s=200, marker='o', label='Data', zorder=5)
        ax_before.scatter([interp_pt[0]], [interp_pt[1]], 
                         c='blue', s=150, marker='s', label='x_t', zorder=5)
        
        # True velocity (green)
        ax_before.arrow(interp_pt[0], interp_pt[1],
                       true_velocity[0] * 0.4, true_velocity[1] * 0.4,
                       head_width=0.15, head_length=0.12,
                       fc='green', ec='green', alpha=0.8, linewidth=3,
                       label='True v_t', zorder=4)
        
        # Predicted velocity (red, with error)
        ax_before.arrow(interp_pt[0], interp_pt[1],
                       prediction[0] * 0.4, prediction[1] * 0.4,
                       head_width=0.15, head_length=0.12,
                       fc='red', ec='red', alpha=0.6, linewidth=2,
                       linestyle='--', label='Predicted u', zorder=3)
        
        ax_before.set_xlabel('Latent Dim 1', fontsize=10)
        ax_before.set_ylabel('Latent Dim 2', fontsize=10)
        ax_before.set_title(f'{stage_name}\nError: {error:.2f}', 
                           fontsize=12, fontweight='bold')
        ax_before.legend(fontsize=9, loc='upper right')
        ax_before.grid(True, alpha=0.3)
        ax_before.set_aspect('equal')
        ax_before.set_xlim([-2, 2])
        ax_before.set_ylim([-2, 2])
        
        # After optimization (row 1): Show loss convergence
        ax_loss = axes[1, stage_idx]
        iterations = np.arange(0, 1000, 10)
        
        if stage_idx == 0:
            # Initial: high loss, slow decrease
            loss_curve = 1.0 * np.exp(-iterations / 500) + 0.3
        elif stage_idx == 1:
            # Mid: moderate loss, faster decrease
            loss_curve = 0.5 * np.exp(-iterations / 200) + 0.1
        else:
            # Converged: low loss, stable
            loss_curve = 0.1 * np.exp(-iterations / 100) + 0.01
        
        ax_loss.plot(iterations, loss_curve, 'b-', linewidth=2)
        ax_loss.fill_between(iterations, loss_curve, alpha=0.3, color='blue')
        ax_loss.set_xlabel('Training Iteration', fontsize=10)
        ax_loss.set_ylabel('Loss L(θ)', fontsize=10)
        ax_loss.set_title(f'Loss Convergence\nFinal: {loss_curve[-1]:.4f}', 
                         fontsize=12, fontweight='bold')
        ax_loss.set_yscale('log')
        ax_loss.grid(True, alpha=0.3)
    
    plt.suptitle('Optimization Process: How Model Learns Velocity Field', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_latent_velocity_field(data_points, noise_points, save_path="flow_matching_latent_velocity_field.png"):
    """
    Detailed visualization of velocity field in latent space
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create grid for velocity field
    x_range = np.linspace(-3, 3, 25)
    y_range = np.linspace(-3, 3, 25)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Compute velocity field
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    Magnitude = np.zeros_like(X)
    
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            point = np.array([X[j, i], Y[j, i]])
            # Find nearest data point
            distances = np.linalg.norm(data_points - point, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_data = data_points[nearest_idx]
            # Velocity = data - point (points toward data)
            velocity = nearest_data - point
            U[j, i] = velocity[0]
            V[j, i] = velocity[1]
            Magnitude[j, i] = np.linalg.norm(velocity)
    
    # Plot 1: Vector field
    ax1 = axes[0]
    ax1.quiver(X, Y, U, V, Magnitude, cmap='viridis', 
              scale=15, width=0.004, alpha=0.7)
    ax1.scatter(data_points[:, 0], data_points[:, 1], 
               c='green', s=200, marker='*', label='Data Points', zorder=5)
    ax1.scatter(noise_points[:, 0], noise_points[:, 1], 
               c='red', s=100, marker='x', alpha=0.5, label='Noise Points', zorder=4)
    ax1.set_xlabel('Latent Dimension 1', fontsize=12)
    ax1.set_ylabel('Latent Dimension 2', fontsize=12)
    ax1.set_title('Velocity Field: u(x, t)\n'
                 'Arrows show direction and magnitude',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Magnitude heatmap
    ax2 = axes[1]
    im = ax2.contourf(X, Y, Magnitude, levels=20, cmap='hot', alpha=0.8)
    ax2.scatter(data_points[:, 0], data_points[:, 1], 
               c='green', s=200, marker='*', label='Data', zorder=5, edgecolors='white')
    ax2.set_xlabel('Latent Dimension 1', fontsize=12)
    ax2.set_ylabel('Latent Dimension 2', fontsize=12)
    ax2.set_title('Velocity Magnitude\n||u(x, t)||',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    plt.colorbar(im, ax=ax2, label='Magnitude')
    
    # Plot 3: Streamlines (flow paths)
    ax3 = axes[2]
    ax3.streamplot(X, Y, U, V, density=1.5, color='blue')
    ax3.scatter(data_points[:, 0], data_points[:, 1], 
               c='green', s=200, marker='*', label='Data', zorder=5)
    ax3.scatter(noise_points[:, 0], noise_points[:, 1], 
               c='red', s=100, marker='x', alpha=0.5, label='Noise', zorder=4)
    
    # Draw example paths
    for noise_pt in noise_points[:3]:
        path = [noise_pt]
        current = noise_pt.copy()
        dt = 0.1
        for _ in range(30):
            # Find velocity at current point
            dists = np.linalg.norm(data_points - current, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_data = data_points[nearest_idx]
            velocity = nearest_data - current
            current = current + dt * velocity
            path.append(current.copy())
            if np.linalg.norm(current - nearest_data) < 0.1:
                break
        path = np.array(path)
        ax3.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, alpha=0.7, 
                label='Example Path' if noise_pt is noise_points[0] else '')
    
    ax3.set_xlabel('Latent Dimension 1', fontsize=12)
    ax3.set_ylabel('Latent Dimension 2', fontsize=12)
    ax3.set_title('Flow Paths: Following Velocity Field\n'
                 'Noise → Data along learned paths',
                 fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    plt.suptitle('Velocity Field in Latent Space: Detailed View', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_noise_to_data_animation(x_data, epsilon, save_path="flow_matching_noise_to_data.gif", n_steps=50):
    """
    Animate the generation process: pure noise gradually transforms into clean data
    by following the learned velocity field (ODE integration)
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    # Ensure RGB format
    x_data = ensure_rgb(x_data)
    epsilon = ensure_rgb(epsilon)
    
    # Simulate ODE integration from t=1 (noise) to t=0 (data)
    # In real generation: dx/dt = u(x, t), integrated backward from t=1 to t=0
    # For visualization, we use the exact linear path (what perfect model would do)
    t_values = np.linspace(1.0, 0.0, n_steps)  # From noise to data
    
    # Pre-compute all frames
    frames = []
    for t in t_values:
        x_t = t * epsilon + (1 - t) * x_data
        frames.append(np.clip(x_t, 0, 1))
    
    fig = plt.figure(figsize=(12, 8))
    
    def animate(frame):
        fig.clear()
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)
        
        # Current time step
        step_idx = min(int(frame * n_steps / 50), n_steps - 1)
        t_current = t_values[step_idx]
        current_image = frames[step_idx]
        
        # Main image display (large)
        ax_main = fig.add_subplot(gs[:, :2])
        ax_main.imshow(current_image)
        ax_main.set_title(f'Generation Process: Noise → Data\n'
                         f'Time: t = {t_current:.3f} ({"Noise" if t_current > 0.9 else "Data" if t_current < 0.1 else "Transition"})',
                         fontsize=14, fontweight='bold')
        ax_main.axis('off')
        
        # Progress bar
        ax_prog = fig.add_subplot(gs[0, 2])
        progress = 1 - t_current  # Progress from 0 to 1
        ax_prog.barh([0], [progress], color='green', alpha=0.7)
        ax_prog.barh([0], [1-progress], left=[progress], color='gray', alpha=0.3)
        ax_prog.set_xlim([0, 1])
        ax_prog.set_ylim([-0.5, 0.5])
        ax_prog.set_xlabel('Progress', fontsize=11)
        ax_prog.set_title(f'{progress*100:.1f}% Complete', fontsize=12, fontweight='bold')
        ax_prog.set_yticks([])
        ax_prog.grid(True, alpha=0.3, axis='x')
        
        # Show velocity field magnitude
        v_field = epsilon - x_data
        # Compute magnitude per pixel (L2 norm across RGB channels)
        v_magnitude = np.linalg.norm(v_field, axis=-1)
        
        # Use robust normalization: clip outliers and normalize
        # This handles the case where most values are similar but a few are extreme
        v_median = np.median(v_magnitude)
        v_std = np.std(v_magnitude)
        
        # Normalize using robust statistics
        v_normalized = (v_magnitude - v_median) / (3 * v_std + 1e-8)  # 3-sigma normalization
        v_normalized = np.clip((v_normalized + 1) / 2, 0, 1)  # Shift to [0, 1]
        
        # Alternative: use log scale if values span orders of magnitude
        if v_magnitude.max() / (v_magnitude.min() + 1e-8) > 10:
            v_log = np.log1p(v_magnitude - v_magnitude.min() + 1e-8)
            v_normalized = v_log / (v_log.max() + 1e-8)
        
        ax_vel = fig.add_subplot(gs[1, 2])
        im = ax_vel.imshow(v_normalized, cmap='viridis', vmin=0, vmax=1)
        ax_vel.set_title(f'Velocity Field Magnitude\n(||ε - x|| per pixel)', 
                        fontsize=10, fontweight='bold')
        ax_vel.axis('off')
        cbar = plt.colorbar(im, ax=ax_vel, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Magnitude', fontsize=9)
        
        # Add text info
        fig.text(0.5, 0.02, 
                f'Step {step_idx+1}/{n_steps} | '
                f'Following velocity field: dx/dt = u(x, t) | '
                f'ODE Integration: t=1.0 → t=0.0',
                ha='center', fontsize=10, style='italic')
        
        plt.suptitle('Flow Matching Generation: Noise → Data', 
                     fontsize=16, fontweight='bold', y=0.98)
    
    try:
        anim = FuncAnimation(fig, animate, frames=50, interval=100, repeat=True)
        anim.save(save_path, writer=PillowWriter(fps=10))
        plt.close()
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Warning: Could not create animation: {e}")
        print("Skipping animation (requires PillowWriter)")
        plt.close()


def visualize_point_flow_animation(x_data, epsilon, save_path="flow_matching_point_flow.gif", n_steps=40):
    """
    Visualize how individual points/pixels flow from noise positions to data positions
    Shows the actual movement/transformation happening during generation
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    # Ensure RGB format
    x_data = ensure_rgb(x_data)
    epsilon = ensure_rgb(epsilon)
    
    # Downsample for visualization (too many points otherwise)
    h, w = x_data.shape[:2]
    downsample = max(1, min(h, w) // 20)  # Sample ~20x20 points
    h_sampled = h // downsample
    w_sampled = w // downsample
    
    # Sample points
    y_coords, x_coords = np.meshgrid(
        np.arange(0, h, downsample),
        np.arange(0, w, downsample),
        indexing='ij'
    )
    
    # Get RGB values at sampled points
    x_data_sampled = x_data[::downsample, ::downsample]
    epsilon_sampled = epsilon[::downsample, ::downsample]
    
    # Compute velocity field (constant for linear interpolation)
    v_field = epsilon_sampled - x_data_sampled
    
    # Generate frames
    t_values = np.linspace(1.0, 0.0, n_steps)
    
    fig = plt.figure(figsize=(16, 8))
    
    def animate(frame):
        fig.clear()
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
        
        step_idx = min(int(frame * n_steps / 40), n_steps - 1)
        t_current = t_values[step_idx]
        
        # Current interpolated image
        current_image = t_current * epsilon + (1 - t_current) * x_data
        current_sampled = t_current * epsilon_sampled + (1 - t_current) * x_data_sampled
        
        # Plot 1: Full image with overlay
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(np.clip(current_image, 0, 1))
        
        # Overlay sampled points
        colors = np.clip(current_sampled, 0, 1)
        for i in range(h_sampled):
            for j in range(w_sampled):
                y, x = y_coords[i, j], x_coords[i, j]
                color = colors[i, j]
                ax1.scatter(x, y, c=[color], s=30, edgecolors='white', linewidths=0.5, alpha=0.8)
        
        ax1.set_title(f'Generation Process (t = {t_current:.3f})\n'
                     f'Points flowing from noise → data',
                     fontsize=13, fontweight='bold')
        ax1.axis('off')
        
        # Plot 2: Show velocity vectors (only at sampled points)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(np.clip(current_image, 0, 1), alpha=0.5)
        
        # Draw velocity vectors (scaled for visibility)
        scale = 5.0  # Scale factor for vector visualization
        for i in range(0, h_sampled, 2):  # Subsample further for clarity
            for j in range(0, w_sampled, 2):
                y, x = y_coords[i, j], x_coords[i, j]
                v = v_field[i, j]
                # Convert velocity to 2D (use magnitude and direction)
                v_mag = np.linalg.norm(v)
                if v_mag > 0.01:  # Only show significant velocities
                    # Use color to show direction
                    v_normalized = v / (v_mag + 1e-8)
                    color = (v_normalized + 1) / 2  # Normalize to [0,1]
                    ax2.arrow(x, y, 
                             v[0] * scale, v[1] * scale,
                             head_width=3, head_length=2,
                             fc=color, ec=color, alpha=0.6, linewidth=1)
        
        ax2.set_title('Velocity Field\nArrows show flow direction',
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Plot 3: Progress visualization
        ax3 = fig.add_subplot(gs[1, 1])
        progress = 1 - t_current
        
        # Show noise and data side by side
        noise_portion = current_sampled
        data_portion = x_data_sampled
        
        # Create comparison
        comparison = np.hstack([
            np.clip(noise_portion, 0, 1),
            np.clip(data_portion, 0, 1)
        ])
        
        ax3.imshow(comparison)
        ax3.axvline(x=w_sampled-0.5, color='red', linestyle='--', linewidth=2)
        ax3.text(w_sampled//2, -h_sampled*0.1, 'Current', ha='center', fontsize=11, fontweight='bold')
        ax3.text(w_sampled + w_sampled//2, -h_sampled*0.1, 'Target', ha='center', fontsize=11, fontweight='bold')
        ax3.set_title(f'Progress: {progress*100:.1f}%\nCurrent → Target',
                     fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        plt.suptitle(f'Point Flow Visualization: Step {step_idx+1}/{n_steps}',
                    fontsize=16, fontweight='bold')
    
    try:
        anim = FuncAnimation(fig, animate, frames=40, interval=120, repeat=True)
        anim.save(save_path, writer=PillowWriter(fps=8))
        plt.close()
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Warning: Could not create point flow animation: {e}")
        print("Skipping animation (requires PillowWriter)")
        plt.close()


def visualize_loss_function(save_path="flow_matching_loss_function.png"):
    """
    Visualize what the model optimizes for:
    L(θ) = E[||u(x_t, t, c; θ) - v_t||²]
    where v_t = ε - x (ground truth velocity)
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Loss as function of prediction error
    error = np.linspace(0, 2, 1000)
    mse_loss = error ** 2
    
    axes[0, 0].plot(error, mse_loss, 'b-', linewidth=3, label='MSE Loss')
    axes[0, 0].axvline(x=0, color='g', linestyle='--', linewidth=2, label='Optimal (error=0)')
    axes[0, 0].set_xlabel('Prediction Error: ||u(x_t, t) - v_t||', fontsize=12)
    axes[0, 0].set_ylabel('Loss: ||u - v_t||²', fontsize=12)
    axes[0, 0].set_title('Loss Function: L = ||u(x_t, t, c; θ) - v_t||²', 
                        fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, 2])
    
    # Plot 2: Loss surface in 2D (predicted vs true velocity)
    v_true = np.linspace(-2, 2, 100)
    v_pred = np.linspace(-2, 2, 100)
    V_true, V_pred = np.meshgrid(v_true, v_pred)
    Loss = (V_pred - V_true) ** 2
    
    im = axes[0, 1].contourf(V_true, V_pred, Loss, levels=20, cmap='viridis')
    axes[0, 1].plot([-2, 2], [-2, 2], 'r--', linewidth=3, label='Optimal: v_pred = v_true')
    axes[0, 1].set_xlabel('True Velocity v_t', fontsize=12)
    axes[0, 1].set_ylabel('Predicted Velocity u(x_t, t)', fontsize=12)
    axes[0, 1].set_title('Loss Landscape\nLower loss (dark) = better', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im, ax=axes[0, 1], label='Loss')
    
    # Plot 3: Expected loss over training distribution
    # Simulate: model starts with random predictions, converges to correct
    iterations = np.arange(0, 1000, 10)
    # Simulate convergence: loss decreases exponentially
    initial_error = 1.5
    convergence_rate = 0.995
    expected_loss = (initial_error * (convergence_rate ** iterations)) ** 2
    
    axes[1, 0].plot(iterations, expected_loss, 'b-', linewidth=3, label='Expected Loss')
    axes[1, 0].set_xlabel('Training Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Expected Loss E[||u - v_t||²]', fontsize=12)
    axes[1, 0].set_title('Optimization Objective\nMinimize: E_{x,ε,t}[||u(x_t, t) - v_t||²]', 
                        fontsize=13, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Loss components visualization
    # Show that loss is computed per sample, then averaged
    n_samples = 10
    sample_losses = np.random.exponential(0.5, n_samples)  # Simulated per-sample losses
    mean_loss = np.mean(sample_losses)
    
    x_pos = np.arange(n_samples)
    axes[1, 1].bar(x_pos, sample_losses, alpha=0.7, color='blue', label='Per-sample loss')
    axes[1, 1].axhline(y=mean_loss, color='red', linestyle='--', linewidth=2, 
                     label=f'Mean loss = {mean_loss:.3f}')
    axes[1, 1].set_xlabel('Training Sample', fontsize=12)
    axes[1, 1].set_ylabel('Loss: ||u - v_t||²', fontsize=12)
    axes[1, 1].set_title('Loss Computation\nL = (1/B) Σᵢ ||u(x_t⁽ⁱ⁾, t⁽ⁱ⁾) - v_t⁽ⁱ⁾||²', 
                        fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("What the Model Optimizes For\nL(θ) = E_{x,ε,t}[||u(x_t, t, c; θ) - v_t||²]", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_loss_landscape(save_path="flow_matching_loss_landscape.png"):
    """
    Visualize the loss landscape/manifold in parameter space
    Shows how loss changes as predictions improve
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    fig = plt.figure(figsize=(16, 6))
    
    # Create a 2D parameter space (simplified: just prediction quality)
    # In reality, this is high-dimensional neural network parameter space
    
    # Simulate: prediction quality vs loss
    prediction_quality = np.linspace(0, 1, 100)  # 0 = random, 1 = perfect
    loss = (1 - prediction_quality) ** 2  # Loss decreases as quality increases
    
    # Plot 1: 1D loss landscape
    ax1 = fig.add_subplot(131)
    ax1.plot(prediction_quality, loss, 'b-', linewidth=3)
    ax1.fill_between(prediction_quality, loss, alpha=0.3, color='blue')
    ax1.scatter([0], [1], s=200, c='red', marker='o', zorder=5, label='Initial (random)')
    ax1.scatter([1], [0], s=200, c='green', marker='*', zorder=5, label='Optimal (perfect)')
    ax1.set_xlabel('Prediction Quality', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Landscape (1D projection)\nLower loss = better predictions', 
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 2D loss manifold
    ax2 = fig.add_subplot(132, projection='3d')
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    # Loss = distance from optimal point (0, 0)
    Z = X**2 + Y**2
    
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
    ax2.scatter([0], [0], [0], s=200, c='green', marker='*', label='Global minimum')
    ax2.set_xlabel('Parameter 1', fontsize=10)
    ax2.set_ylabel('Parameter 2', fontsize=10)
    ax2.set_zlabel('Loss', fontsize=10)
    ax2.set_title('Loss Manifold (2D projection)\nHigh-dimensional optimization', 
                  fontsize=13, fontweight='bold')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)
    
    # Plot 3: Optimization trajectory
    ax3 = fig.add_subplot(133)
    # Simulate optimization path
    n_steps = 50
    theta1_path = np.linspace(-1.5, 0, n_steps) + 0.1 * np.random.randn(n_steps)
    theta2_path = np.linspace(-1.5, 0, n_steps) + 0.1 * np.random.randn(n_steps)
    loss_path = theta1_path**2 + theta2_path**2
    
    # Contour plot
    x = np.linspace(-2, 0.5, 50)
    y = np.linspace(-2, 0.5, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    ax3.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    
    # Optimization path
    ax3.plot(theta1_path, theta2_path, 'r-', linewidth=2, alpha=0.7, label='Optimization path')
    ax3.scatter(theta1_path[0], theta2_path[0], s=200, c='red', marker='o', 
               zorder=5, label='Start')
    ax3.scatter(theta1_path[-1], theta2_path[-1], s=200, c='green', marker='*', 
               zorder=5, label='Converged')
    ax3.scatter([0], [0], s=200, c='blue', marker='x', linewidths=3, zorder=5, label='Optimum')
    ax3.set_xlabel('Parameter 1', fontsize=12)
    ax3.set_ylabel('Parameter 2', fontsize=12)
    ax3.set_title('Optimization Trajectory\nGradient descent path', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    plt.suptitle("Loss Landscape & Optimization Manifold", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_loss_vs_noise_level(x_data, epsilon, save_path="flow_matching_loss_vs_noise.png"):
    """
    Visualize how loss varies with noise level (t)
    Shows which noise levels are harder to predict
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    t_values = np.linspace(0, 1, 100)
    
    # Simulate: loss is typically higher at intermediate noise levels
    # At t=0: easy (pure data, clear structure)
    # At t=1: easy (pure noise, model learns to predict constant velocity)
    # At t~0.5: hardest (mixed, ambiguous)
    loss_curve = 0.5 * np.sin(np.pi * t_values) + 0.3
    loss_curve = np.maximum(loss_curve, 0.1)  # Ensure positive
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.2)
    
    # Plot 1: Loss vs noise level (spans first 3 columns)
    ax1 = fig.add_subplot(gs[0, :3])
    ax1.plot(t_values, loss_curve, 'b-', linewidth=3, alpha=0.7)
    ax1.fill_between(t_values, loss_curve, alpha=0.3, color='blue')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
                label='Hardest (t=0.5)')
    ax1.set_xlabel('Noise Level t', fontsize=12)
    ax1.set_ylabel('Expected Loss E[||u - v_t||²]', fontsize=12)
    ax1.set_title('Loss vs Noise Level\nDifferent t values have different difficulty', 
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training distribution effect (spans last 2 columns)
    n_samples = 10000
    t_uniform = np.random.rand(n_samples)
    t_logit = 1 / (1 + np.exp(-np.random.randn(n_samples)))
    
    loss_uniform = 0.5 * np.sin(np.pi * t_uniform) + 0.3
    loss_logit = 0.5 * np.sin(np.pi * t_logit) + 0.3
    
    ax2 = fig.add_subplot(gs[0, 3:])
    ax2.hist(t_uniform, bins=30, weights=loss_uniform, density=True, 
            alpha=0.6, color='blue', label='Uniform')
    ax2.hist(t_logit, bins=30, weights=loss_logit, density=True, 
            alpha=0.6, color='green', label='Logit-normal')
    ax2.set_xlabel('Noise Level t', fontsize=10)
    ax2.set_ylabel('Weighted Density', fontsize=10)
    ax2.set_title('Training Distribution\nEffect', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3-7: Sample images at different t values (5 images in row 1)
    t_samples = [0.0, 0.25, 0.5, 0.75, 1.0]
    sample_images = []
    for t in t_samples:
        x_t = (1 - t) * x_data + t * epsilon
        sample_images.append(x_t)
    
    for i, (t, img) in enumerate(zip(t_samples, sample_images)):
        ax = fig.add_subplot(gs[1, i])
        loss_val = 0.5 * np.sin(np.pi * t) + 0.3
        ax.imshow(np.clip(img, 0, 1))
        color = 'red' if t == 0.5 else 'black'
        ax.set_title(f't={t:.2f}\nLoss≈{loss_val:.2f}', fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
    
    # Plot 8: Why β-shift helps (spans all columns in row 2)
    ax3 = fig.add_subplot(gs[2, :])
    beta = 2.0
    t_original = np.linspace(0, 1, 100)
    t_shifted = (beta * t_original) / (1 + (beta - 1) * t_original)
    loss_original = 0.5 * np.sin(np.pi * t_original) + 0.3
    loss_shifted = 0.5 * np.sin(np.pi * t_shifted) + 0.3
    
    ax3.plot(t_original, loss_original, 'b-', linewidth=2, label='Original distribution', alpha=0.7)
    ax3.plot(t_shifted, loss_shifted, 'g-', linewidth=2, label='After β-shift (β=2)', alpha=0.7)
    ax3.fill_between(t_original, loss_original, alpha=0.2, color='blue')
    ax3.fill_between(t_shifted, loss_shifted, alpha=0.2, color='green')
    ax3.set_xlabel('Noise Level t', fontsize=12)
    ax3.set_ylabel('Expected Loss', fontsize=12)
    ax3.set_title('Why β-Shift Helps: Focus Training on Harder Samples', 
                 fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle("Loss Variation with Noise Level\nWhy β-shift helps: focus on harder samples", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_convergence_animation(x_data, epsilon, save_path="flow_matching_convergence.gif"):
    """
    Create animation showing how predictions converge during training
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    # Pre-compute data
    v_true = epsilon - x_data
    n_iterations = 50
    iterations = np.arange(n_iterations)
    initial_error = 1.0
    convergence_rate = 0.95
    errors = initial_error * (convergence_rate ** iterations)
    
    fig = plt.figure(figsize=(20, 8))
    
    def animate(frame):
        fig.clear()
        gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.2)
        
        iter_idx = min(int(frame * n_iterations / 50), n_iterations - 1)
        current_error = errors[iter_idx]
        progress = 1 - (current_error / initial_error)
        
        # Simulate prediction
        np.random.seed(42)  # For reproducibility
        v_pred = v_true * progress + np.random.randn(*v_true.shape) * current_error * 0.3
        
        # Show 5 time steps of generation
        t_gen = np.linspace(1.0, 0.0, 5)
        for i, t in enumerate(t_gen):
            ax = fig.add_subplot(gs[0, i])
            x_t_gen = (1 - t) * x_data + t * epsilon
            prediction_error = current_error * (1 - t)
            np.random.seed(42 + i)
            x_t_gen = x_t_gen + np.random.randn(*x_t_gen.shape) * prediction_error * 0.1
            ax.imshow(np.clip(x_t_gen, 0, 1))
            ax.set_title(f't={t:.2f}', fontsize=10)
            ax.axis('off')
        
        # Loss curve
        ax_loss = fig.add_subplot(gs[1, 0])
        ax_loss.plot(iterations[:iter_idx+1], errors[:iter_idx+1]**2, 'b-', linewidth=2)
        ax_loss.scatter([iter_idx], [errors[iter_idx]**2], s=100, c='red', zorder=5)
        ax_loss.set_xlabel('Iteration', fontsize=10)
        ax_loss.set_ylabel('Loss', fontsize=10)
        ax_loss.set_title(f'Loss: {errors[iter_idx]**2:.4f}', fontsize=10)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_yscale('log')
        
        # Error bar
        ax_err = fig.add_subplot(gs[1, 1])
        ax_err.barh([0], [current_error], color='blue', alpha=0.7)
        ax_err.set_xlim([0, initial_error])
        ax_err.set_xlabel('Error', fontsize=10)
        ax_err.set_title(f'Error: {current_error:.4f}', fontsize=10)
        ax_err.set_yticks([])
        
        # Velocity comparison
        v_vis_true = np.clip((v_true + 1) / 2, 0, 1)
        v_vis_pred = np.clip((v_pred + 1) / 2, 0, 1)
        
        ax_vtrue = fig.add_subplot(gs[1, 2])
        ax_vtrue.imshow(v_vis_true)
        ax_vtrue.set_title('True Velocity', fontsize=10)
        ax_vtrue.axis('off')
        
        ax_vpred = fig.add_subplot(gs[1, 3])
        ax_vpred.imshow(v_vis_pred)
        ax_vpred.set_title('Predicted Velocity', fontsize=10)
        ax_vpred.axis('off')
        
        ax_prog = fig.add_subplot(gs[1, 4])
        ax_prog.text(0.5, 0.5, f'Progress:\n{progress*100:.1f}%', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    transform=ax_prog.transAxes)
        ax_prog.axis('off')
        
        plt.suptitle(f'Training Convergence (Iteration {iter_idx})', 
                    fontsize=16, fontweight='bold')
    
    try:
        anim = FuncAnimation(fig, animate, frames=50, interval=100, repeat=True)
        anim.save(save_path, writer=PillowWriter(fps=10))
        plt.close()
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Warning: Could not create animation: {e}")
        print("Skipping animation (requires PillowWriter)")
        plt.close()


def visualize_optimization_trajectory(save_path="flow_matching_optimization_trajectory.png"):
    """
    Visualize the optimization trajectory in high-dimensional space
    Shows how gradient descent navigates the loss landscape
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Simulate optimization trajectory
    n_steps = 100
    steps = np.arange(n_steps)
    
    # Simulate loss decrease (exponential decay with noise)
    true_loss = 2.0 * np.exp(-steps / 30) + 0.1
    noisy_loss = true_loss + 0.1 * np.random.randn(n_steps)
    noisy_loss = np.maximum(noisy_loss, 0.05)  # Ensure positive
    
    # Plot 1: Loss over iterations
    axes[0].plot(steps, noisy_loss, 'b-', alpha=0.5, linewidth=1, label='Noisy loss')
    axes[0].plot(steps, true_loss, 'r-', linewidth=2, label='Smoothed loss')
    axes[0].set_xlabel('Training Iteration', fontsize=12)
    axes[0].set_ylabel('Loss L(θ)', fontsize=12)
    axes[0].set_title('Loss Convergence\nL(θ) decreases over time', fontsize=13, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Gradient magnitude (learning signal)
    # Gradients are large initially, decrease as we converge
    gradient_magnitude = 0.5 * np.exp(-steps / 25) + 0.05
    axes[1].plot(steps, gradient_magnitude, 'g-', linewidth=2)
    axes[1].fill_between(steps, gradient_magnitude, alpha=0.3, color='green')
    axes[1].set_xlabel('Training Iteration', fontsize=12)
    axes[1].set_ylabel('||∇L(θ)||', fontsize=12)
    axes[1].set_title('Gradient Magnitude\nLarge gradients → fast learning', 
                     fontsize=13, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Parameter space trajectory (2D projection)
    # Simulate parameter updates following gradient descent
    theta1 = np.cumsum(-0.1 * np.exp(-steps / 30) * np.random.randn(n_steps))
    theta2 = np.cumsum(-0.1 * np.exp(-steps / 30) * np.random.randn(n_steps))
    
    # Create loss contours
    x = np.linspace(theta1.min() - 0.5, theta1.max() + 0.5, 50)
    y = np.linspace(theta2.min() - 0.5, theta2.max() + 0.5, 50)
    X, Y = np.meshgrid(x, y)
    # Loss decreases toward center
    center_x, center_y = theta1[-1], theta2[-1]
    Z = (X - center_x)**2 + (Y - center_y)**2
    
    contour = axes[2].contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
    axes[2].plot(theta1, theta2, 'r-', linewidth=2, alpha=0.7, label='Optimization path')
    axes[2].scatter(theta1[0], theta2[0], s=200, c='red', marker='o', 
                   zorder=5, label='Start')
    axes[2].scatter(theta1[-1], theta2[-1], s=200, c='green', marker='*', 
                   zorder=5, label='Current')
    axes[2].set_xlabel('Parameter Dimension 1', fontsize=12)
    axes[2].set_ylabel('Parameter Dimension 2', fontsize=12)
    axes[2].set_title('Parameter Space Trajectory\n(2D projection of high-dim space)', 
                     fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal')
    
    plt.suptitle("Optimization Trajectory: How the Model Learns", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()

