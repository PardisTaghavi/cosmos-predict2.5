"""
3D Flow Matching Visualization
Shows flow matching in 3D latent space with a folded manifold as clean data.
Black dots are pushed by evolving velocity vector fields toward the manifold.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import os


def generate_folded_manifold(n_samples=500):
    """
    Generate a 3D folded manifold (like a curved sheet) representing clean data.
    """
    np.random.seed(42)
    
    # Create a folded surface: z = f(x, y) with folds
    u = np.linspace(-2, 2, int(np.sqrt(n_samples)))
    v = np.linspace(-2, 2, int(np.sqrt(n_samples)))
    U, V = np.meshgrid(u, v)
    
    # Create a folded surface with multiple folds
    X = U
    Y = V
    Z = 0.5 * np.sin(2 * np.pi * U / 2) * np.cos(2 * np.pi * V / 2) + 0.3 * np.sin(np.pi * U)
    
    # Flatten to points
    points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    # Add small noise for realism
    points += np.random.randn(*points.shape) * 0.05
    
    return points


def find_nearest_point_on_manifold(point, manifold_points):
    """Find nearest point on the manifold"""
    distances = np.linalg.norm(manifold_points - point, axis=1)
    nearest_idx = np.argmin(distances)
    return manifold_points[nearest_idx]


def visualize_3d_flow_matching(save_path="flow_matching_3d_fold.gif", n_steps=60):
    """
    Animate flow matching in 3D latent space with a folded manifold.
    Shows black dots being pushed by evolving velocity fields toward the clean data manifold.
    Now includes kinematic conditioning (red points) that guide the flow.
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    # Generate folded manifold (clean data) - FIXED, doesn't move
    manifold_points = generate_folded_manifold(n_samples=500)
    
    # Sample fixed subset for visualization (same every frame)
    np.random.seed(42)  # Fixed seed for consistent visualization
    vis_indices = np.random.choice(len(manifold_points), 200, replace=False)
    vis_manifold = manifold_points[vis_indices]  # Fixed manifold points
    
    # Generate kinematic constraint points (red) - these act as structural anchors
    # These represent 3D kinematic constraints that guide the flow
    np.random.seed(456)  # Different seed for kinematic points
    n_kinematic = 15
    # Place kinematic constraints strategically on/near the manifold
    kinematic_points = []
    for _ in range(n_kinematic):
        # Sample a point on the manifold
        idx = np.random.randint(0, len(manifold_points))
        base_point = manifold_points[idx]
        # Add small offset to show they're constraints, not exact manifold points
        offset = np.random.randn(3) * 0.2
        kinematic_points.append(base_point + offset)
    kinematic_points = np.array(kinematic_points)
    
    # Generate initial points (black dots) - scattered around the manifold
    np.random.seed(123)
    n_points = 80
    # Create points in a sphere around the manifold
    initial_points = np.random.randn(n_points, 3) * 2.5
    
    # Time progression: from t=1 (initial scattered) to t=0 (converged to manifold)
    t_values = np.linspace(1.0, 0.0, n_steps)
    
    # Create figure with 3D subplot
    fig = plt.figure(figsize=(16, 10))
    
    def animate(frame):
        fig.clear()
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2, 
                             height_ratios=[3, 1], width_ratios=[2, 1])
        
        step_idx = min(int(frame * n_steps / 50), n_steps - 1)
        t_current = t_values[step_idx]
        learning_progress = 1 - t_current  # 0 to 1 as we go from t=1 to t=0
        
        # Interpolate points: start from scattered, move towards manifold
        # With kinematic conditioning: points are pulled toward BOTH manifold AND kinematic constraints
        current_points = []
        kinematic_influence = 0.3  # How much kinematic constraints influence the flow
        
        for point in initial_points:
            # Find target on manifold
            target_manifold = find_nearest_point_on_manifold(point, manifold_points)
            
            # Find nearest kinematic constraint
            kinematic_distances = np.linalg.norm(kinematic_points - point, axis=1)
            nearest_kin_idx = np.argmin(kinematic_distances)
            target_kinematic = kinematic_points[nearest_kin_idx]
            
            # Blend: flow toward manifold, but kinematic constraints guide the path
            # As learning progresses, kinematic influence increases
            kinematic_weight = kinematic_influence * learning_progress
            target = (1 - kinematic_weight) * target_manifold + kinematic_weight * target_kinematic
            
            # Interpolate: x_t = t * initial + (1-t) * target
            interpolated = t_current * point + (1 - t_current) * target
            current_points.append(interpolated)
        current_points = np.array(current_points)
        
        # Compute kinematic conditioning effect on latents
        # Points closer to kinematic constraints get "colored" by kinematic conditioning
        kinematic_conditioning_strength = np.zeros(len(current_points))
        for i, point in enumerate(current_points):
            # Distance to nearest kinematic constraint
            dists_to_kin = np.linalg.norm(kinematic_points - point, axis=1)
            min_dist_kin = np.min(dists_to_kin)
            # Distance to manifold
            dist_to_manifold = np.linalg.norm(point - find_nearest_point_on_manifold(point, manifold_points))
            
            # Kinematic conditioning is stronger when:
            # 1. Close to kinematic constraints
            # 2. Learning has progressed (kinematic constraints become more influential)
            kinematic_strength = learning_progress * np.exp(-min_dist_kin / 0.5)  # Gaussian falloff
            kinematic_conditioning_strength[i] = kinematic_strength
        
        # Main 3D plot
        ax_main = fig.add_subplot(gs[0, :], projection='3d')
        
        # Draw folded manifold (clean data) - FIXED, doesn't move
        # vis_manifold is computed once outside the animation loop
        ax_main.scatter(vis_manifold[:, 0], vis_manifold[:, 1], vis_manifold[:, 2],
                       c='orange', s=50, alpha=0.6, label='Clean Data (Folded Manifold - Fixed)', 
                       edgecolors='darkorange', linewidths=0.5)
        
        # Draw kinematic constraint points (red) - FIXED structural anchors
        ax_main.scatter(kinematic_points[:, 0], kinematic_points[:, 1], kinematic_points[:, 2],
                       c='red', s=150, marker='*', alpha=0.8, label='Kinematic Constraints (3D Structure)',
                       edgecolors='darkred', linewidths=1.5, zorder=5)
        
        # Draw current points - color by kinematic conditioning strength
        # Points with strong kinematic conditioning appear redder
        colors = []
        for strength in kinematic_conditioning_strength:
            # Blend from black (no conditioning) to red (strong conditioning)
            if strength > 0.1:
                # Red tint based on kinematic conditioning strength
                colors.append([strength, 0, 0])  # Red component
            else:
                colors.append([0, 0, 0])  # Black
        
        colors = np.array(colors)
        # Normalize to [0, 1] range for RGB
        if colors.max() > 0:
            colors = colors / colors.max()
        
        # Create RGB colors: black → dark red → red based on kinematic conditioning
        point_colors = np.zeros((len(current_points), 3))
        for i, strength in enumerate(kinematic_conditioning_strength):
            if strength > 0.1:
                # Red tint: stronger kinematic conditioning = more red
                point_colors[i] = [min(strength * 2, 1.0), 0, 0]  # Red channel
            else:
                point_colors[i] = [0, 0, 0]  # Black
        
        ax_main.scatter(current_points[:, 0], current_points[:, 1], current_points[:, 2],
                       c=point_colors, s=80, marker='o', alpha=0.8, 
                       label='Latents (red = kinematic conditioning)',
                       edgecolors='gray', linewidths=0.5)
        
        # Draw lines from points to nearest kinematic constraints (when conditioning is active)
        if learning_progress > 0.2:  # Only show when kinematic conditioning is active
            for i, point in enumerate(current_points[:20]):  # Show subset for clarity
                if kinematic_conditioning_strength[i] > 0.2:
                    kin_distances = np.linalg.norm(kinematic_points - point, axis=1)
                    nearest_kin_idx = np.argmin(kin_distances)
                    nearest_kin = kinematic_points[nearest_kin_idx]
                    # Draw line with alpha based on conditioning strength
                    ax_main.plot([point[0], nearest_kin[0]], 
                               [point[1], nearest_kin[1]], 
                               [point[2], nearest_kin[2]],
                               'r--', alpha=kinematic_conditioning_strength[i] * 0.5, 
                               linewidth=1, zorder=1)
        
        ax_main.set_xlabel('Latent Dim 1', fontsize=11)
        ax_main.set_ylabel('Latent Dim 2', fontsize=11)
        ax_main.set_zlabel('Latent Dim 3', fontsize=11)
        avg_kinematic_strength = np.mean(kinematic_conditioning_strength)
        ax_main.set_title(f'Flow Matching with Kinematic Conditioning\n'
                         f't = {t_current:.2f} | '
                         f'Kinematic Influence: {avg_kinematic_strength:.2f} '
                         f'({"Initial" if t_current > 0.9 else "Converged" if t_current < 0.1 else "Transition"})',
                         fontsize=13, fontweight='bold', pad=20)
        ax_main.legend(loc='upper left', fontsize=10)
        ax_main.set_xlim([-3, 3])
        ax_main.set_ylim([-3, 3])
        ax_main.set_zlim([-2, 2])
        
        # Progress visualization (top right)
        ax_prog = fig.add_subplot(gs[1, 0])
        progress = learning_progress
        ax_prog.barh([0], [progress], color='green', alpha=0.7, height=0.5)
        ax_prog.barh([0], [1-progress], left=[progress], color='gray', alpha=0.3, height=0.5)
        ax_prog.set_xlim([0, 1])
        ax_prog.set_ylim([-0.5, 0.5])
        ax_prog.set_xlabel('Optimization Progress', fontsize=11, fontweight='bold')
        ax_prog.set_title(f'{progress*100:.1f}% Complete', fontsize=12, fontweight='bold')
        ax_prog.set_yticks([])
        ax_prog.grid(True, alpha=0.3, axis='x')
        
        # Kinematic conditioning strength visualization (bottom right)
        ax_quality = fig.add_subplot(gs[1, 1])
        
        avg_kinematic_strength = np.mean(kinematic_conditioning_strength)
        
        # Show kinematic conditioning strength as a bar
        ax_quality.barh([0], [avg_kinematic_strength], color='red', alpha=0.7, height=0.5)
        ax_quality.set_xlim([0, 1])
        ax_quality.set_ylim([-0.5, 0.5])
        ax_quality.set_xlabel('Kinematic Conditioning Strength', fontsize=11, fontweight='bold')
        ax_quality.set_title(f'Avg Strength: {avg_kinematic_strength:.2f}', fontsize=12, fontweight='bold')
        ax_quality.set_yticks([])
        ax_quality.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('3D Flow Matching with Kinematic Conditioning\n'
                     'Red = Kinematic Constraints Guide Flow Toward Structurally Consistent Solutions', 
                    fontsize=16, fontweight='bold', y=0.98)
    
    try:
        anim = FuncAnimation(fig, animate, frames=50, interval=120, repeat=True)
        anim.save(save_path, writer=PillowWriter(fps=8))
        plt.close()
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Warning: Could not create 3D animation: {e}")
        print("Skipping animation (requires PillowWriter)")
        plt.close()


def visualize_3d_static_view(save_path="flow_matching_3d_fold_static.png"):
    """
    Create a static 3D visualization showing the folded manifold and points.
    """
    if os.path.exists(save_path):
        print(f"Skipping (already exists): {save_path}")
        return
    
    # Generate folded manifold - FIXED
    manifold_points = generate_folded_manifold(n_samples=500)
    
    # Sample fixed subset for visualization
    np.random.seed(42)  # Same seed as animation for consistency
    vis_indices = np.random.choice(len(manifold_points), 200, replace=False)
    vis_manifold = manifold_points[vis_indices]  # Fixed manifold points
    
    # Generate kinematic constraint points (red) - same as animation
    np.random.seed(456)
    n_kinematic = 15
    kinematic_points = []
    for _ in range(n_kinematic):
        idx = np.random.randint(0, len(manifold_points))
        base_point = manifold_points[idx]
        offset = np.random.randn(3) * 0.2
        kinematic_points.append(base_point + offset)
    kinematic_points = np.array(kinematic_points)
    
    # Generate points at different stages
    np.random.seed(123)
    n_points = 80
    initial_points = np.random.randn(n_points, 3) * 2.5
    kinematic_influence = 0.3
    
    fig = plt.figure(figsize=(18, 6))
    
    stages = [
        ("Initial (t=1.0)", 1.0),
        ("Mid Training (t=0.5)", 0.5),
        ("Converged (t=0.0)", 0.0)
    ]
    
    for idx, (title, t_val) in enumerate(stages):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        # Interpolate points with kinematic conditioning
        learning_progress = 1 - t_val
        current_points = []
        kinematic_conditioning_strength = []
        
        for point in initial_points:
            target_manifold = find_nearest_point_on_manifold(point, manifold_points)
            kinematic_distances = np.linalg.norm(kinematic_points - point, axis=1)
            nearest_kin_idx = np.argmin(kinematic_distances)
            target_kinematic = kinematic_points[nearest_kin_idx]
            
            kinematic_weight = kinematic_influence * learning_progress
            target = (1 - kinematic_weight) * target_manifold + kinematic_weight * target_kinematic
            interpolated = t_val * point + (1 - t_val) * target
            current_points.append(interpolated)
            
            # Compute kinematic conditioning strength
            min_dist_kin = np.min(kinematic_distances)
            strength = learning_progress * np.exp(-min_dist_kin / 0.5)
            kinematic_conditioning_strength.append(strength)
        
        current_points = np.array(current_points)
        kinematic_conditioning_strength = np.array(kinematic_conditioning_strength)
        
        # Draw manifold - FIXED (same points every time)
        ax.scatter(vis_manifold[:, 0], vis_manifold[:, 1], vis_manifold[:, 2],
                  c='orange', s=50, alpha=0.6, edgecolors='darkorange', linewidths=0.5,
                  label='Clean Data (Fixed)' if idx == 0 else '')
        
        # Draw kinematic constraints
        ax.scatter(kinematic_points[:, 0], kinematic_points[:, 1], kinematic_points[:, 2],
                  c='red', s=150, marker='*', alpha=0.8, edgecolors='darkred', linewidths=1.5,
                  label='Kinematic Constraints' if idx == 0 else '', zorder=5)
        
        # Draw points colored by kinematic conditioning
        point_colors = np.zeros((len(current_points), 3))
        for i, strength in enumerate(kinematic_conditioning_strength):
            if strength > 0.1:
                point_colors[i] = [min(strength * 2, 1.0), 0, 0]  # Red tint
            else:
                point_colors[i] = [0, 0, 0]  # Black
        
        ax.scatter(current_points[:, 0], current_points[:, 1], current_points[:, 2],
                  c=point_colors, s=80, marker='o', alpha=0.8, edgecolors='gray', linewidths=0.5,
                  label='Latents (red = kinematic)' if idx == 0 else '')
        
        ax.set_xlabel('Dim 1', fontsize=10)
        ax.set_ylabel('Dim 2', fontsize=10)
        ax.set_zlabel('Dim 3', fontsize=10)
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-2, 2])
    
    plt.suptitle('3D Flow Matching with Kinematic Conditioning\n'
                 'Red = Structural Constraints Guide Flow', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("3D Flow Matching Visualization")
    print("=" * 60)
    
    print("\nGenerating 3D flow matching visualizations...")
    visualize_3d_flow_matching()
    visualize_3d_static_view()
    
    print("\n" + "=" * 60)
    print("All visualizations complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - flow_matching_3d_fold.gif")
    print("  - flow_matching_3d_fold_static.png")

