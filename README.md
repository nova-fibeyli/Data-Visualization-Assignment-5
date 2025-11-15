# Data-Visualization-Assignment-5
# Open3D Assignment - README

This project contains the full pipeline for completing all 7 tasks required in the Open3D lab assignment.

## Files
- `lab_open3d.py` — main script containing all processing steps.
- `model.ply` — your unique 3D model (replace with your own file).
- `README.md` — this file.

## Requirements
Install Open3D before running:

```bash
pip install open3d

How to Run

Inside the project folder:

```bash
python lab_open3d.py

Close each visualization window to continue to the next step.

Steps Performed in Script

Load and visualize original mesh

Convert to point cloud

Poisson surface reconstruction

Voxelization of point cloud

Add plane to the scene

Clip surface by plane

Apply custom Z-gradient and mark extreme points

Each step prints:

number of vertices / triangles

whether colors and normals exist

number of voxels (if applicable)

coordinates of extrema
