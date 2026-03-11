import numpy as np
import open3d as o3d
import trimesh


def pointcloud_to_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    poisson_depth: int = 8,
    density_quantile: float = 0.05,
    estimate_normals_radius: float = 0.05,
    estimate_normals_max_nn: int = 30,
) -> trimesh.Trimesh:
    """Build a triangle mesh from a coloured point cloud.

    Pipeline:
      1. Open3D PointCloud with normals estimated from local neighbourhoods.
      2. Poisson surface reconstruction.
      3. Low-density vertex trimming (removes hull artefacts).
      4. Convert to Trimesh.

    Parameters
    ----------
    points : [N, 3]
    colors : [N, 3]  in [0, 1]
    poisson_depth : int
        Octree depth (higher = finer detail, slower).
    density_quantile : float
        Remove vertices below this density quantile.
    estimate_normals_radius : float
        Search radius for normal estimation.
    estimate_normals_max_nn : int
        Max neighbours for normal estimation.

    Returns
    -------
    trimesh.Trimesh with vertex colours.
    """
    # ── 1. Build Open3D point cloud ─────────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if colors.max() > 1.01:
        colors = colors / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    # ── 2. Estimate normals ─────────────────────────────────────────────
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=estimate_normals_radius, max_nn=estimate_normals_max_nn
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # ── 3. Poisson surface reconstruction ───────────────────────────────
    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, linear_fit=True
    )

    # ── 4. Remove low-density vertices (hull artefacts) ─────────────────
    if density_quantile > 0:
        densities = np.asarray(densities)
        thresh = np.quantile(densities, density_quantile)
        vertices_to_remove = densities < thresh
        mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

    mesh_o3d.compute_vertex_normals()

    # ── 5. Convert to Trimesh ───────────────────────────────────────────
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    vertex_normals = np.asarray(mesh_o3d.vertex_normals)

    if mesh_o3d.has_vertex_colors():
        vc = (np.asarray(mesh_o3d.vertex_colors) * 255).astype(np.uint8)
    else:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        _, idx = tree.query(vertices)
        vc = colors[idx]
        if vc.max() <= 1.01:
            vc = vc * 255
        vc = vc.astype(np.uint8)

    # RGBA for trimesh
    vertex_colors_rgba = np.column_stack(
        [vc, np.full((len(vc), 1), 255, dtype=np.uint8)]
    )

    tri = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        vertex_colors=vertex_colors_rgba,
        process=False,
    )
    return tri
