"""
Synthetic Training Data Generator

Generate 3D scenes with geometric shapes (cylinder, cone) and render them as 
black and white 2D projection images with realistic shadows.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from matplotlib.path import Path


def create_cylinder(radius=1, height=2, resolution=50):
    """Create a 3D cylinder mesh."""
    theta = np.linspace(0, 2*np.pi, resolution)
    z = np.linspace(0, height, resolution)
    Theta, Z = np.meshgrid(theta, z)
    X = radius * np.cos(Theta)
    Y = radius * np.sin(Theta)
    return X, Y, Z


def create_cone(radius=1, height=2, resolution=50):
    """Create a 3D cone mesh."""
    theta = np.linspace(0, 2*np.pi, resolution)
    z = np.linspace(0, height, resolution)
    Theta, Z = np.meshgrid(theta, z)
    R = radius * (1 - Z / height)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    return X, Y, Z


def spherical_to_cartesian(azimuth, elevation, radius):
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        azimuth: Angle in radians (0 to 2π)
        elevation: Angle in radians (from xy-plane, -π/2 to π/2)
        radius: Distance from origin
    
    Returns:
        3D point as numpy array [x, y, z]
    """
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    return np.array([x, y, z])


def random_point_on_sphere(radius=10, min_elevation=20, max_elevation=80):
    """
    Generate a random point on a sphere (e.g., for sun/light position).
    
    Args:
        radius: Distance from origin
        min_elevation: Minimum elevation angle in degrees
        max_elevation: Maximum elevation angle in degrees
    
    Returns:
        3D point as numpy array [x, y, z]
    """
    azimuth = np.random.uniform(0, 2 * np.pi)
    elevation = np.random.uniform(np.radians(min_elevation), np.radians(max_elevation))
    return spherical_to_cartesian(azimuth, elevation, radius)


def compute_shadow(object_x, object_y, object_z, light_pos):
    """Compute shadow projection on the ground plane (z=0)."""
    obj_x = object_x.ravel()
    obj_y = object_y.ravel()
    obj_z = object_z.ravel()
    
    dir_x = obj_x - light_pos[0]
    dir_y = obj_y - light_pos[1]
    dir_z = obj_z - light_pos[2]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        t = -light_pos[2] / dir_z
        valid = (t > 0) & (obj_z < light_pos[2]) & np.isfinite(t)
        
        shadow_x = np.where(valid, light_pos[0] + t * dir_x, np.nan)
        shadow_y = np.where(valid, light_pos[1] + t * dir_y, np.nan)
    
    valid_mask = ~np.isnan(shadow_x)
    return shadow_x[valid_mask], shadow_y[valid_mask]


def rasterize_convex_hull(img, points, u_range, v_range, img_width, img_height, color_value=0):
    """Rasterize the convex hull of projected points."""
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        img_u = ((hull_points[:, 0] - u_range[0]) / (u_range[1] - u_range[0]) * img_width).astype(int)
        # Flip v-axis: high v values should map to low row indices (top of image)
        img_v = ((v_range[1] - hull_points[:, 1]) / (v_range[1] - v_range[0]) * img_height).astype(int)
        
        y_grid, x_grid = np.mgrid[0:img_height, 0:img_width]
        points_grid = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        
        path = Path(np.column_stack([img_u, img_v]))
        mask = path.contains_points(points_grid).reshape(img_height, img_width)
        
        img[mask] = color_value
    except:
        img_u = ((points[:, 0] - u_range[0]) / (u_range[1] - u_range[0]) * img_width).astype(int)
        # Flip v-axis: high v values should map to low row indices (top of image)
        img_v = ((v_range[1] - points[:, 1]) / (v_range[1] - v_range[0]) * img_height).astype(int)
        valid = (img_u >= 0) & (img_u < img_width) & (img_v >= 0) & (img_v < img_height)
        img[img_v[valid], img_u[valid]] = color_value
    
    return img


def _generate_scene_data(random_projection=True, randomize_objects=True, add_shadows=True):
    """
    Internal helper to generate all scene data.
    Returns a dictionary with all intermediate data for rendering and visualization.
    """
    # Randomize object parameters
    if randomize_objects:
        cyl_radius = np.random.uniform(0.5, 1.5)
        cyl_height = np.random.uniform(2.0, 4.0)
        cyl_center = np.array([np.random.uniform(-2, 0), np.random.uniform(-1, 1), 0])
        
        cone_radius = np.random.uniform(0.5, 1.5)
        cone_height = np.random.uniform(1.5, 3.5)
        cone_center = np.array([np.random.uniform(2, 5), np.random.uniform(-1, 1), 0])
    else:
        cyl_radius, cyl_height = 1.0, 3.0
        cyl_center = np.array([0, 0, 0])
        cone_radius, cone_height = 1.2, 2.5
        cone_center = np.array([3, 0, 0])
    
    # Create 3D objects
    cyl_x, cyl_y, cyl_z = create_cylinder(cyl_radius, cyl_height)
    cyl_x += cyl_center[0]
    cyl_y += cyl_center[1]
    cyl_z += cyl_center[2]
    
    cone_x, cone_y, cone_z = create_cone(cone_radius, cone_height)
    cone_x += cone_center[0]
    cone_y += cone_center[1]
    cone_z += cone_center[2]
    
    # Generate shadows
    light_pos = None
    cyl_shadow_x = cyl_shadow_y = cone_shadow_x = cone_shadow_y = np.array([])
    if add_shadows:
        light_pos = random_point_on_sphere()
        cyl_shadow_x, cyl_shadow_y = compute_shadow(cyl_x, cyl_y, cyl_z, light_pos)
        cone_shadow_x, cone_shadow_y = compute_shadow(cone_x, cone_y, cone_z, light_pos)
    
    # Define projection plane
    camera_pos = None
    azimuth = elevation = None
    if random_projection:
        azimuth = np.random.uniform(0, 2 * np.pi)
        elevation = np.random.uniform(np.radians(10), np.radians(80))
    else:
        azimuth = 0
        elevation = np.radians(45)

    # Camera position using spherical coordinates
    camera_pos = spherical_to_cartesian(azimuth, elevation, radius=20)
    normal = camera_pos / np.linalg.norm(camera_pos)
    
    # u_vec is horizontal, perpendicular to normal's xy projection
    u_vec = np.array([-normal[1], normal[0], 0])
    u_vec = u_vec / np.linalg.norm(u_vec)
    
    # v_vec points upward in the image (positive z component)
    v_vec = np.cross(normal, u_vec)
    v_vec = v_vec / np.linalg.norm(v_vec)
    if v_vec[2] < 0:
        v_vec = -v_vec
    
    # Project to plane
    def project_to_plane(x, y, z):
        points_3d = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        u = np.dot(points_3d, u_vec)
        v = np.dot(points_3d, v_vec)
        return u, v
    
    # Project objects
    cyl_u, cyl_v = project_to_plane(cyl_x, cyl_y, cyl_z)
    cyl_points = np.column_stack([cyl_u, cyl_v])
    
    cone_u, cone_v = project_to_plane(cone_x, cone_y, cone_z)
    cone_points = np.column_stack([cone_u, cone_v])
    
    # Project shadows
    all_u = np.concatenate([cyl_u, cone_u])
    all_v = np.concatenate([cyl_v, cone_v])
    
    cyl_shadow_points = cone_shadow_points = np.array([]).reshape(0, 2)
    if add_shadows:
        cyl_shadow_u, cyl_shadow_v = project_to_plane(cyl_shadow_x, cyl_shadow_y, np.zeros_like(cyl_shadow_x))
        cyl_shadow_points = np.column_stack([cyl_shadow_u, cyl_shadow_v])
        
        cone_shadow_u, cone_shadow_v = project_to_plane(cone_shadow_x, cone_shadow_y, np.zeros_like(cone_shadow_x))
        cone_shadow_points = np.column_stack([cone_shadow_u, cone_shadow_v])
        
        if len(cyl_shadow_u) > 0:
            all_u = np.concatenate([all_u, cyl_shadow_u])
        if len(cone_shadow_u) > 0:
            all_u = np.concatenate([all_u, cone_shadow_u])
    
    u_range = (all_u.min() - 0.5, all_u.max() + 0.5)
    v_range = (all_v.min() - 0.5, all_v.max() + 0.5)
    
    return {
        'cyl_x': cyl_x, 'cyl_y': cyl_y, 'cyl_z': cyl_z,
        'cone_x': cone_x, 'cone_y': cone_y, 'cone_z': cone_z,
        'cyl_shadow_x': cyl_shadow_x, 'cyl_shadow_y': cyl_shadow_y,
        'cone_shadow_x': cone_shadow_x, 'cone_shadow_y': cone_shadow_y,
        'light_pos': light_pos,
        'camera_pos': camera_pos,
        'azimuth': azimuth, 'elevation': elevation,
        'cyl_points': cyl_points, 'cone_points': cone_points,
        'cyl_shadow_points': cyl_shadow_points, 'cone_shadow_points': cone_shadow_points,
        'u_range': u_range, 'v_range': v_range,
        'cyl_radius': cyl_radius, 'cyl_height': cyl_height, 'cyl_center': cyl_center,
        'cone_radius': cone_radius, 'cone_height': cone_height, 'cone_center': cone_center
    }


def generate_synthetic_scene(random_projection=True, randomize_objects=True, 
                              add_shadows=True, img_width=800, img_height=600, 
                              return_scene_data=False, return_separate_masks=False):
    """
    Generate a 3D scene with cylinder and cone, render as black & white 2D image.
    NO PLOTTING - just returns the binary image array.
    
    Args:
        random_projection: Whether to use random camera viewing angle
        randomize_objects: Whether to randomize object parameters
        add_shadows: Whether to add shadows
        img_width: Width of output image
        img_height: Height of output image
        return_scene_data: If True, returns (image, scene_data) tuple
        return_separate_masks: If True, returns (image_without_shadows, shadow_mask, image_with_shadows)
                              where shadow_mask is binary (1=shadow, 0=no shadow)
    
    Returns:
        Binary image array (1, img_width, img_height)
        OR tuple of (image, scene_data) if return_scene_data=True
        OR tuple of (image_without_shadows, shadow_mask, image_with_shadows, scene_data) 
           if return_separate_masks=True
    """
    # Generate all scene data
    data = _generate_scene_data(random_projection, randomize_objects, add_shadows)
    
    if return_separate_masks:
        # Create three separate images
        image_with_shadows = np.ones((img_height, img_width))
        image_without_shadows = np.ones((img_height, img_width))
        shadow_mask = np.zeros((img_height, img_width))
        
        # Rasterize shadows separately
        if add_shadows:
            if len(data['cyl_shadow_points']) > 3:
                shadow_mask = rasterize_convex_hull(shadow_mask, data['cyl_shadow_points'], 
                                                    data['u_range'], data['v_range'], 
                                                    img_width, img_height, 1.0)
                image_with_shadows = rasterize_convex_hull(image_with_shadows, data['cyl_shadow_points'], 
                                                           data['u_range'], data['v_range'], 
                                                           img_width, img_height, 0.5)
            if len(data['cone_shadow_points']) > 3:
                shadow_mask = rasterize_convex_hull(shadow_mask, data['cone_shadow_points'], 
                                                    data['u_range'], data['v_range'], 
                                                    img_width, img_height, 1.0)
                image_with_shadows = rasterize_convex_hull(image_with_shadows, data['cone_shadow_points'], 
                                                           data['u_range'], data['v_range'], 
                                                           img_width, img_height, 0.5)
        
        # Rasterize objects on both images (objects overwrite shadows)
        image_with_shadows = rasterize_convex_hull(image_with_shadows, data['cyl_points'], 
                                                   data['u_range'], data['v_range'], 
                                                   img_width, img_height, 0)
        image_with_shadows = rasterize_convex_hull(image_with_shadows, data['cone_points'], 
                                                   data['u_range'], data['v_range'], 
                                                   img_width, img_height, 0)
        image_without_shadows = rasterize_convex_hull(image_without_shadows, data['cyl_points'], 
                                                      data['u_range'], data['v_range'], 
                                                      img_width, img_height, 0)
        image_without_shadows = rasterize_convex_hull(image_without_shadows, data['cone_points'], 
                                                      data['u_range'], data['v_range'], 
                                                      img_width, img_height, 0)
        
        # Remove object bases from shadow mask (only observable shadow, not shape footprints)
        # Where objects are present (value 0 in image_without_shadows), set shadow_mask to 0
        shadow_mask[image_without_shadows < 0.5] = 0.0
        
        # Add channel dimension
        image_without_shadows = image_without_shadows[np.newaxis, :, :]
        shadow_mask = shadow_mask[np.newaxis, :, :]
        image_with_shadows = image_with_shadows[np.newaxis, :, :]
        
        return image_without_shadows, shadow_mask, image_with_shadows, data
    
    # Original behavior: create composite image
    image = np.ones((img_height, img_width))
    
    # Rasterize shadows first
    if add_shadows:
        if len(data['cyl_shadow_points']) > 3:
            image = rasterize_convex_hull(image, data['cyl_shadow_points'], data['u_range'], 
                                         data['v_range'], img_width, img_height, 0.5)
        if len(data['cone_shadow_points']) > 3:
            image = rasterize_convex_hull(image, data['cone_shadow_points'], data['u_range'], 
                                         data['v_range'], img_width, img_height, 0.5)
    
    # Rasterize objects
    image = rasterize_convex_hull(image, data['cyl_points'], data['u_range'], 
                                  data['v_range'], img_width, img_height, 0)
    image = rasterize_convex_hull(image, data['cone_points'], data['u_range'], 
                                  data['v_range'], img_width, img_height, 0)
    
    image_array = image[np.newaxis, :, :]
    
    if return_scene_data:
        return image_array, data
    return image_array


def visualize_synthetic_scene(scene_data=None, random_projection=True, randomize_objects=True, 
                               add_shadows=True, img_width=800, img_height=600, save_path=None):
    """
    Generate and VISUALIZE a 3D scene. Shows 3D view, projection, and binary image.
    
    Args:
        scene_data: Optional pre-generated scene data from generate_synthetic_scene().
                   If provided, this data is visualized instead of generating new data.
        random_projection: Whether to use random camera viewing angle (ignored if scene_data provided)
        randomize_objects: Whether to randomize object parameters (ignored if scene_data provided)
        add_shadows: Whether to add shadows (ignored if scene_data provided)
        img_width: Width of output image
        img_height: Height of output image
        save_path: Optional path to save the figure
    
    Returns:
        Binary image array (1, img_width, img_height)
    """
    # Generate all scene data (or use provided data)
    if scene_data is None:
        data = _generate_scene_data(random_projection, randomize_objects, add_shadows)
    else:
        data = scene_data
    
    # Print scene info
    if randomize_objects:
        print(f"Cylinder: r={data['cyl_radius']:.2f}, h={data['cyl_height']:.2f}, "
              f"c=({data['cyl_center'][0]:.2f},{data['cyl_center'][1]:.2f},{data['cyl_center'][2]:.2f})")
        print(f"Cone: r={data['cone_radius']:.2f}, h={data['cone_height']:.2f}, "
              f"c=({data['cone_center'][0]:.2f},{data['cone_center'][1]:.2f},{data['cone_center'][2]:.2f})")
    
    if add_shadows and data['light_pos'] is not None:
        print(f"Light: ({data['light_pos'][0]:.2f}, {data['light_pos'][1]:.2f}, {data['light_pos'][2]:.2f})")
    
    if random_projection and data['azimuth'] is not None:
        print(f"Camera: Az={np.degrees(data['azimuth']):.1f}°, El={np.degrees(data['elevation']):.1f}°")
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # 3D Visualization
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(data['cyl_x'], data['cyl_y'], data['cyl_z'], alpha=0.7, color='blue')
    ax1.plot_surface(data['cone_x'], data['cone_y'], data['cone_z'], alpha=0.7, color='red')
    
    if add_shadows:
        if len(data['cyl_shadow_x']) > 0:
            ax1.scatter(data['cyl_shadow_x'], data['cyl_shadow_y'], 
                       np.zeros_like(data['cyl_shadow_x']), c='gray', s=1, alpha=0.5)
        if len(data['cone_shadow_x']) > 0:
            ax1.scatter(data['cone_shadow_x'], data['cone_shadow_y'], 
                       np.zeros_like(data['cone_shadow_x']), c='gray', s=1, alpha=0.5)
        if data['light_pos'] is not None:
            ax1.scatter([data['light_pos'][0]], [data['light_pos'][1]], [data['light_pos'][2]], 
                       c='yellow', s=200, marker='*', edgecolors='orange', linewidths=2)
    
    if data['camera_pos'] is not None:
        ax1.scatter([data['camera_pos'][0]], [data['camera_pos'][1]], [data['camera_pos'][2]], 
                   c='cyan', s=200, marker='*', edgecolors='blue', linewidths=2)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Scene')
    ax1.set_box_aspect([1,1,1])
    
    # 2D Projection
    ax2 = fig.add_subplot(132)
    
    try:
        hull = ConvexHull(data['cyl_points'])
        ax2.fill(data['cyl_points'][hull.vertices, 0], data['cyl_points'][hull.vertices, 1], 
                color='blue', alpha=1.0, edgecolor='darkblue', linewidth=2)
    except:
        ax2.scatter(data['cyl_points'][:, 0], data['cyl_points'][:, 1], c='blue', s=1, alpha=0.5)
    
    try:
        hull = ConvexHull(data['cone_points'])
        ax2.fill(data['cone_points'][hull.vertices, 0], data['cone_points'][hull.vertices, 1], 
                color='red', alpha=1.0, edgecolor='darkred', linewidth=2)
    except:
        ax2.scatter(data['cone_points'][:, 0], data['cone_points'][:, 1], c='red', s=1, alpha=0.5)
    
    if add_shadows:
        try:
            if len(data['cyl_shadow_points']) > 3:
                hull = ConvexHull(data['cyl_shadow_points'])
                ax2.fill(data['cyl_shadow_points'][hull.vertices, 0], 
                        data['cyl_shadow_points'][hull.vertices, 1], 
                        color='gray', alpha=0.5, edgecolor='darkgray', linewidth=1)
        except:
            pass
        
        try:
            if len(data['cone_shadow_points']) > 3:
                hull = ConvexHull(data['cone_shadow_points'])
                ax2.fill(data['cone_shadow_points'][hull.vertices, 0], 
                        data['cone_shadow_points'][hull.vertices, 1], 
                        color='gray', alpha=0.5, edgecolor='darkgray', linewidth=1)
        except:
            pass
    
    ax2.set_xlim(data['u_range'])
    ax2.set_ylim(data['v_range'])
    ax2.set_xlabel('u')
    ax2.set_ylabel('v')
    ax2.set_title('Projection')
    ax2.set_aspect('equal')
    ax2.set_facecolor('white')
    ax2.grid(True, alpha=0.3)
    
    # Binary image
    ax3 = fig.add_subplot(133)
    image = np.ones((img_height, img_width))
    
    if add_shadows:
        if len(data['cyl_shadow_points']) > 3:
            image = rasterize_convex_hull(image, data['cyl_shadow_points'], data['u_range'], 
                                         data['v_range'], img_width, img_height, 0.5)
        if len(data['cone_shadow_points']) > 3:
            image = rasterize_convex_hull(image, data['cone_shadow_points'], data['u_range'], 
                                         data['v_range'], img_width, img_height, 0.5)
    
    image = rasterize_convex_hull(image, data['cyl_points'], data['u_range'], 
                                  data['v_range'], img_width, img_height, 0)
    image = rasterize_convex_hull(image, data['cone_points'], data['u_range'], 
                                  data['v_range'], img_width, img_height, 0)
    
    ax3.imshow(image, cmap='gray', interpolation='nearest', 
              extent=data['u_range'] + data['v_range'], origin='upper')
    ax3.set_xlabel('u')
    ax3.set_ylabel('v')
    ax3.set_title('Binary Image')
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return image[np.newaxis, :, :]


if __name__ == "__main__":
    # Example 1: Just generate data (no visualization)
    print("Generating training data...")
    data = generate_synthetic_scene(random_projection=True, randomize_objects=True, add_shadows=True)
    print(f"Data shape: {data.shape}")
    
    # Example 2: Generate and visualize
    print("\nGenerating with visualization...")
    img = visualize_synthetic_scene(random_projection=True, randomize_objects=True, add_shadows=True)
    print(f"Image shape: {img.shape}")
