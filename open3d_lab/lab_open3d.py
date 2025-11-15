import open3d as o3d
import numpy as np

# ---------- SETTINGS ----------
MODEL_PATH = "model.ply"   # change to your file name
VOXEL_SIZE = 0.05          # can be changed
PLANE_POINT = np.array([0, 0, 0])   # point on the plane
PLANE_NORMAL = np.array([0, 0, 1])  # plane normal (here: horizontal plane)


def print_mesh_info(title, mesh: o3d.geometry.TriangleMesh):
    print(f"\n=== {title} ===")
    print(f"Vertices: {np.asarray(mesh.vertices).shape[0]}")
    print(f"Triangles: {np.asarray(mesh.triangles).shape[0]}")
    print(f"Has vertex colors: {mesh.has_vertex_colors()}")
    print(f"Has vertex normals: {mesh.has_vertex_normals()}")


def print_pcd_info(title, pcd: o3d.geometry.PointCloud):
    print(f"\n=== {title} ===")
    print(f"Points: {np.asarray(pcd.points).shape[0]}")
    print(f"Has colors: {pcd.has_colors()}")
    print(f"Has normals: {pcd.has_normals()}")


# 1. LOADING AND VISUALIZATION ----------------------------------------------
print("Step 1: Loading mesh...")
mesh = o3d.io.read_triangle_mesh(MODEL_PATH)
mesh.compute_vertex_normals()

if not mesh.has_vertex_colors():
    # just give a simple color if the model has no colors
    mesh.paint_uniform_color([0.8, 0.8, 0.8])

print_mesh_info("Original Mesh", mesh)
o3d.visualization.draw_geometries([mesh], window_name="Original Mesh")

# 2. CONVERSION TO POINT CLOUD ---------------------------------------------
# Option A: try to read file directly as point cloud
print("\nStep 2: Converting to point cloud...")
pcd = o3d.io.read_point_cloud(MODEL_PATH)
if len(pcd.points) == 0:
    # Option B: sample points from mesh if file is not a point cloud
    print("File is not a point cloud, sampling from mesh...")
    pcd = mesh.sample_points_poisson_disk(number_of_points=50000)

print_pcd_info("Point Cloud (from model)", pcd)
o3d.visualization.draw_geometries([pcd], window_name="Sampled Point Cloud")

# 3. SURFACE RECONSTRUCTION (POISSON) --------------------------------------
print("\nStep 3: Poisson surface reconstruction...")
pcd.estimate_normals()
mesh_rec, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9
)

# Crop artifacts with bounding box (based on original pcd bounds)
bbox = pcd.get_axis_aligned_bounding_box()
mesh_rec = mesh_rec.crop(bbox)

print_mesh_info("Reconstructed Mesh (Poisson)", mesh_rec)
o3d.visualization.draw_geometries([mesh_rec], window_name="Reconstructed Mesh (Poisson)")

# 4. VOXELIZATION -----------------------------------------------------------
print("\nStep 4: Voxelization...")
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=VOXEL_SIZE)

voxels = voxel_grid.get_voxels()
print(f"\n=== Voxel Grid ===")
print(f"Number of voxels: {len(voxels)}")
print(f"Has colors: {voxel_grid.has_colors()}")

o3d.visualization.draw_geometries([voxel_grid], window_name="Voxelized Model")

# 5. ADDING A PLANE ---------------------------------------------------------
print("\nStep 5: Adding a plane...")

# Make a simple plane as a big thin box (rectangle)
plane = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.01)
plane.compute_vertex_normals()
plane.paint_uniform_color([0.3, 0.3, 0.3])

# Move and scale plane near the object (use bbox of original mesh)
bbox_mesh = mesh.get_axis_aligned_bounding_box()
center = bbox_mesh.get_center()

# Scale the plane to cover the model
scale_factor = max(bbox_mesh.get_extent()) * 1.2
plane.scale(scale_factor, center=(0, 0, 0))

# Position plane: here we place it under the model and tilt a bit
plane.translate(center - np.array([0, 0, bbox_mesh.get_extent()[2] * 0.5]))
R = plane.get_rotation_matrix_from_xyz((np.deg2rad(20), np.deg2rad(0), np.deg2rad(30)))
plane.rotate(R, center=center)

o3d.visualization.draw_geometries([mesh, plane], window_name="Plane + Original Mesh")

# For clipping we need plane equation: using PLANE_POINT and PLANE_NORMAL
PLANE_NORMAL = PLANE_NORMAL / np.linalg.norm(PLANE_NORMAL)


# 6. SURFACE CLIPPING -------------------------------------------------------
print("\nStep 6: Clipping points on one side of plane...")

# We will clip on the point cloud, then reconstruct mesh again
points = np.asarray(pcd.points)

# Example: redefine plane so it passes roughly through the center of model
PLANE_POINT = center  # point on the plane
PLANE_NORMAL = np.array([0.0, 0.0, 1.0])  # plane normal pointing upwards

PLANE_NORMAL = PLANE_NORMAL / np.linalg.norm(PLANE_NORMAL)

# Signed distance from each point to plane
dists = (points - PLANE_POINT) @ PLANE_NORMAL

# Keep only points with distance <= 0 (left/below side)
keep_idx = np.where(dists <= 0)[0]
pcd_clipped = pcd.select_by_index(keep_idx)

print_pcd_info("Point Cloud after clipping", pcd_clipped)

# Reconstruct mesh again from clipped cloud so we have triangles
pcd_clipped.estimate_normals()
mesh_clipped, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd_clipped, depth=9
)
mesh_clipped = mesh_clipped.crop(pcd_clipped.get_axis_aligned_bounding_box())

print_mesh_info("Mesh after clipping (from clipped pcd)", mesh_clipped)
o3d.visualization.draw_geometries([mesh_clipped], window_name="Clipped Mesh")

# 7. WORKING WITH COLOR AND EXTREMES ---------------------------------------
print("\nStep 7: Gradient color by Z and highlight extremes...")

# Choose axis: 2 -> Z, 0 -> X, 1 -> Y
axis = 2

pcd_grad = pcd_clipped  # work on clipped cloud
coords = np.asarray(pcd_grad.points)[:, axis]
z_min, z_max = coords.min(), coords.max()
print(f"Axis {axis} min: {z_min}, max: {z_max}")

# Normalize to [0, 1]
t = (coords - z_min) / (z_max - z_min + 1e-9)

# Simple gradient: red -> blue
colors = np.zeros((len(t), 3))
colors[:, 0] = 1.0 - t  # red decreases
colors[:, 2] = t        # blue increases
pcd_grad.colors = o3d.utility.Vector3dVector(colors)

# Find extreme points
idx_min = int(np.argmin(coords))
idx_max = int(np.argmax(coords))

point_min = coords[idx_min]
point_max = coords[idx_max]

extreme_min = np.asarray(pcd_grad.points)[idx_min]
extreme_max = np.asarray(pcd_grad.points)[idx_max]

print("\nExtreme coordinates along chosen axis:")
print(f"Min point: {extreme_min}")
print(f"Max point: {extreme_max}")

# Highlight extremes by small spheres
sphere_min = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
sphere_min.translate(extreme_min)
sphere_min.paint_uniform_color([0, 1, 0])  # green

sphere_max = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
sphere_max.translate(extreme_max)
sphere_max.paint_uniform_color([1, 0, 0])  # red

# Also draw coordinate frame so orientation is clear
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

o3d.visualization.draw_geometries(
    [pcd_grad, sphere_min, sphere_max, axes],
    window_name="3D Model with Z-Extremes and Axes"
)

print("\nDone.")
