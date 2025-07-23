# === General Plotting ===

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle

# PLOTTING THE BEZIER CURVE ON A FOOTBALL PITCH
# Set up the figure with publication quality
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=300)

# Define pitch colors
pitch_color = '#4a7c59'
line_color = 'white'
run_color = '#dc2677'
control_color = '#4169E1'
mid_control_color = '#FFA500'

# Draw pitch
pitch = FancyBboxPatch((0, 0), 12, 8, boxstyle="round,pad=0.1", 
                        facecolor=pitch_color, edgecolor=line_color, linewidth=2)
ax.add_patch(pitch)

# Draw pitch markings
ax.plot([6, 6], [0, 8], color=line_color, linewidth=1.5)
circle = Circle((6, 4), 1.2, fill=False, edgecolor=line_color, linewidth=1.5)
ax.add_patch(circle)

# Penalty areas
left_box = patches.Rectangle((0, 2), 2, 4, fill=False, edgecolor=line_color, linewidth=1.5)
right_box = patches.Rectangle((10, 2), 2, 4, fill=False, edgecolor=line_color, linewidth=1.5)
ax.add_patch(left_box)
ax.add_patch(right_box)

# Define control points
P0 = np.array([2.5, 5.5])
P1 = np.array([4, 2.5])
P2 = np.array([8, 1.5])
P3 = np.array([10, 3])

# Function to calculate Bezier curve
def bezier_curve(P0, P1, P2, P3, num_points=100):
    t = np.linspace(0, 1, num_points)
    curve = np.array([(1-ti)**3 * P0 + 3*(1-ti)**2*ti * P1 + 
                    3*(1-ti)*ti**2 * P2 + ti**3 * P3 for ti in t])
    return curve

# Draw control polygon
ax.plot([P0[0], P1[0], P2[0], P3[0]], [P0[1], P1[1], P2[1], P3[1]], 
        'w--', alpha=0.5, linewidth=1)

# Draw Bezier curve
curve = bezier_curve(P0, P1, P2, P3)
ax.plot(curve[:, 0], curve[:, 1], color=run_color, linewidth=3)

# Add arrow at the end
ax.annotate('', xy=(P3[0], P3[1]), xytext=(curve[-10, 0], curve[-10, 1]),
            arrowprops=dict(arrowstyle='->', color=run_color, lw=3))

# Draw control points
ax.scatter(*P0, color=control_color, s=100, zorder=5, edgecolor='white', linewidth=2)
ax.scatter(*P1, color=mid_control_color, s=100, zorder=5, edgecolor='white', linewidth=2)
ax.scatter(*P2, color=mid_control_color, s=100, zorder=5, edgecolor='white', linewidth=2)
ax.scatter(*P3, color=control_color, s=100, zorder=5, edgecolor='white', linewidth=2)

# Add labels
ax.text(P0[0]-0.3, P0[1]+0.3, r'$P_0$', color='white', fontsize=12, weight='bold')
ax.text(P1[0], P1[1]-0.4, r'$P_1$', color='white', fontsize=12, weight='bold')
ax.text(P2[0], P2[1]-0.4, r'$P_2$', color='white', fontsize=12, weight='bold')
ax.text(P3[0]+0.2, P3[1]+0.3, r'$P_3$', color='white', fontsize=12, weight='bold')

# Add parameter values along curve
for t_val in [0.25, 0.5, 0.75]:
    point = (1-t_val)**3 * P0 + 3*(1-t_val)**2*t_val * P1 + \
            3*(1-t_val)*t_val**2 * P2 + t_val**3 * P3
    ax.scatter(*point, color='white', s=40, zorder=5)
    ax.text(point[0], point[1]+0.3, f'$t={t_val}$', color='white', 
            fontsize=9, ha='center')

# Add start/end labels
ax.text(P0[0], P0[1]+0.7, r'$t=0$', color=run_color, fontsize=10, weight='bold')
ax.text(P3[0], P3[1]-0.5, r'$t=1$', color=run_color, fontsize=10, weight='bold')

# Add direction indicators
ax.text(10.5, 7, 'Attack →', color='white', fontsize=11, weight='bold')
ax.text(0.5, 7, '← Defense', color='white', fontsize=11, weight='bold')

# Set axis properties
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(-0.5, 8.5)
ax.set_aspect('equal')
ax.axis('off')

# Add mathematical equation as text box
equation = r'$\mathbf{B}(t) = (1-t)^3\mathbf{P}_0 + 3(1-t)^2t\mathbf{P}_1 + 3(1-t)t^2\mathbf{P}_2 + t^3\mathbf{P}_3$'
ax.text(6, -0.3, equation, ha='center', va='top', fontsize=11, 
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('bezier_curve_football_run.pdf', dpi=300, bbox_inches='tight')
plt.savefig('bezier_curve_football_run.png', dpi=300, bbox_inches='tight')
plt.show()


# Set up figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)

# Define Bezier control points for cluster curve
P0 = np.array([1, 1])
P1 = np.array([3, 2])
P2 = np.array([6, 4])
P3 = np.array([9, 3.5])

# Generate Bezier curve
def bezier_curve(t, P0, P1, P2, P3):
    return (1-t)**3 * P0 + 3*(1-t)**2*t * P1 + 3*(1-t)*t**2 * P2 + t**3 * P3

# Generate trajectory (slightly perturbed from Bezier)
N = 9
t_vals = np.linspace(0, 1, N)
bezier_points = np.array([bezier_curve(t, P0, P1, P2, P3) for t in t_vals])
trajectory_points = bezier_points + np.random.normal(0, 0.3, (N, 2))
trajectory_points[0] = bezier_points[0] + [0, 0.2]  # Ensure visible difference
trajectory_points[-1] = bezier_points[-1] + [0, 0.2]

# Draw background
ax.set_facecolor('#f0f0f0')
ax.grid(True, alpha=0.3, linestyle='--')

# Draw Bezier curve
t_smooth = np.linspace(0, 1, 100)
bezier_smooth = np.array([bezier_curve(t, P0, P1, P2, P3) for t in t_smooth])
ax.plot(bezier_smooth[:, 0], bezier_smooth[:, 1], 'b-', linewidth=3, 
        label='Cluster Bezier curve')

# Draw trajectory points
ax.scatter(trajectory_points[:, 0], trajectory_points[:, 1], 
           color='#dc2677', s=100, zorder=5, label='Resampled trajectory')

# Draw Bezier evaluation points
ax.scatter(bezier_points[:, 0], bezier_points[:, 1], 
           color='blue', s=100, marker='s', zorder=5, label='Bezier samples')

# Draw L1 distances for selected points
selected_indices = [1, 3, 5, 7]
for i in selected_indices:
    traj_pt = trajectory_points[i]
    bez_pt = bezier_points[i]
    
    # Draw Manhattan path
    ax.plot([traj_pt[0], bez_pt[0]], [traj_pt[1], traj_pt[1]], 
            'orange', linestyle='--', linewidth=2, alpha=0.7)
    ax.plot([bez_pt[0], bez_pt[0]], [traj_pt[1], bez_pt[1]], 
            'orange', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add distance annotations
    dx = abs(traj_pt[0] - bez_pt[0])
    dy = abs(traj_pt[1] - bez_pt[1])
    
    if dx > 0.1:
        ax.text((traj_pt[0] + bez_pt[0])/2, traj_pt[1] + 0.1, 
                f'Δx={dx:.2f}', fontsize=8, ha='center')
    if dy > 0.1:
        ax.text(bez_pt[0] + 0.1, (traj_pt[1] + bez_pt[1])/2, 
                f'Δy={dy:.2f}', fontsize=8, rotation=90, va='center')

# Calculate and display total L1 distance
l1_distances = np.sum(np.abs(trajectory_points - bezier_points), axis=1)
avg_l1 = np.mean(l1_distances)

# Add text box with calculation
textstr = f'L1 Distance Calculation:\n'
textstr += f'N = {N} resampled points\n'
textstr += f'Average L1 distance = {avg_l1:.3f} units'
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Labels and formatting
ax.set_xlabel('x (meters)', fontsize=12)
ax.set_ylabel('y (meters)', fontsize=12)
ax.set_title('L1 Distance Between Trajectory and Cluster Bezier Curve', fontsize=14)
ax.legend(loc='lower right')
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)

plt.tight_layout()
plt.savefig('l1_distance_bezier.pdf', dpi=300, bbox_inches='tight')
plt.savefig('l1_distance_bezier.png', dpi=300, bbox_inches='tight')
plt.show()


#==== Autoencoder Architecture Diagram ====

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

# Define colors
input_color = '#e8f4f8'
encoder_colors = ['#cce5ff', '#99ccff', '#66b2ff']
latent_color = '#ffe6e6'
decoder_colors = ['#ccffcc', '#99ff99', '#66ff66']

# Define positions and sizes
box_width = 1.2
box_height = 0.8
y_center = 3

positions = {
    'input': (0.5, y_center),
    'flatten': (2, y_center),
    'enc1': (3.5, y_center),
    'enc2': (5, y_center),
    'latent': (6.5, y_center),
    'dec1': (8, y_center),
    'dec2': (9.5, y_center),
    'unflatten': (11, y_center),
    'output': (12.5, y_center)
}

# Draw boxes
def draw_box(ax, pos, width, height, text, color, fontsize=10):
    box = FancyBboxPatch((pos[0]-width/2, pos[1]-height/2), width, height,
                         boxstyle="round,pad=0.1", facecolor=color,
                         edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(pos[0], pos[1], text, ha='center', va='center', fontsize=fontsize, weight='bold')

# Draw trajectory boxes
draw_box(ax, positions['input'], box_width*0.8, box_height*1.5, 
         'Input\n$\\tilde{\\mathbf{T}}$\n25×2', input_color, 9)
draw_box(ax, positions['output'], box_width*0.8, box_height*1.5, 
         'Output\n$\\hat{\\mathbf{T}}$\n25×2', input_color, 9)

# Draw layer boxes
draw_box(ax, positions['flatten'], box_width, box_height, 'Flatten', encoder_colors[0])
draw_box(ax, positions['enc1'], box_width, box_height, 'Linear\nReLU', encoder_colors[1])
draw_box(ax, positions['enc2'], box_width, box_height, 'Linear', encoder_colors[2])
draw_box(ax, positions['latent'], box_width, box_height, 'Latent\n$\\mathbf{z}$', latent_color)
draw_box(ax, positions['dec1'], box_width, box_height, 'Linear\nReLU', decoder_colors[0])
draw_box(ax, positions['dec2'], box_width, box_height, 'Linear', decoder_colors[1])
draw_box(ax, positions['unflatten'], box_width, box_height, 'Reshape', decoder_colors[2])

# Draw arrows
arrow_style = "Simple, tail_width=0.5, head_width=6, head_length=8"
arrow_kwargs = dict(arrowstyle=arrow_style, color="black", lw=1.5)

for i, (start, end) in enumerate([
    ('input', 'flatten'), ('flatten', 'enc1'), ('enc1', 'enc2'),
    ('enc2', 'latent'), ('latent', 'dec1'), ('dec1', 'dec2'),
    ('dec2', 'unflatten'), ('unflatten', 'output')
]):
    start_pos = positions[start]
    end_pos = positions[end]
    arrow = FancyArrowPatch((start_pos[0] + box_width/2, start_pos[1]),
                           (end_pos[0] - box_width/2, end_pos[1]),
                           **arrow_kwargs)
    ax.add_patch(arrow)

# Add dimension labels
dims = {'flatten': '50', 'enc1': '64', 'enc2': '16', 
        'latent': '16', 'dec1': '64', 'dec2': '50', 'unflatten': '25×2'}
for key, dim in dims.items():
    ax.text(positions[key][0], positions[key][1] - box_height/2 - 0.3, 
            dim, ha='center', fontsize=9, color='gray', style='italic')

# Add encoder/decoder labels
ax.text(3.5, 4.5, 'Encoder $f_\\theta$', ha='center', fontsize=12, color='blue', weight='bold')
ax.text(9.5, 4.5, 'Decoder $g_\\phi$', ha='center', fontsize=12, color='green', weight='bold')

# Add dashed boxes for encoder/decoder
encoder_box = Rectangle((1.5, 2), 4, 2, fill=False, linestyle='--', 
                       edgecolor='blue', linewidth=1)
decoder_box = Rectangle((7.5, 2), 4, 2, fill=False, linestyle='--', 
                       edgecolor='green', linewidth=1)
ax.add_patch(encoder_box)
ax.add_patch(decoder_box)

# Add title
ax.text(6.5, 5.5, 'Trajectory Autoencoder Architecture', 
        ha='center', fontsize=14, weight='bold')

# Add clustering annotation
ax.text(6.5, 1, 'K-means clustering\napplied to latent space', 
        ha='center', fontsize=10, style='italic', color='red',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red'))

plt.tight_layout()
plt.savefig('autoencoder_architecture.pdf', dpi=300, bbox_inches='tight')
plt.savefig('autoencoder_architecture.png', dpi=300, bbox_inches='tight')
plt.show()