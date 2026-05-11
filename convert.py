import trimesh
import numpy as np
from PIL import Image
import os
import re


def compute_avg_color(mesh):
    """Compute the average albedo color for a mesh by sampling its texture at vertex UVs."""
    visual = mesh.visual

    # Default fallback
    default = np.array([255, 255, 255, 255], dtype=np.uint8)

    if not isinstance(visual, trimesh.visual.TextureVisuals):
        print(f"  Not TextureVisuals ({type(visual).__name__}), using white")
        return default

    uv = visual.uv
    material = visual.material

    if uv is None:
        print(f"  No UVs, using base color")
        if hasattr(material, "main_color") and material.main_color is not None:
            return np.array(material.main_color, dtype=np.uint8)
        return default

    # Try to get the albedo/base color image
    image = None
    if hasattr(material, "baseColorTexture") and material.baseColorTexture is not None:
        image = material.baseColorTexture
    elif hasattr(material, "image") and material.image is not None:
        image = material.image

    if image is None:
        if hasattr(material, "main_color") and material.main_color is not None:
            return np.array(material.main_color, dtype=np.uint8)
        return default

    image = image.convert("RGBA")
    pixels = np.array(image)
    h, w = pixels.shape[:2]

    # Sample texture at each vertex's UV (nearest-neighbor)
    u = np.clip(uv[:, 0] % 1.0, 0, 1)
    v = np.clip(uv[:, 1] % 1.0, 0, 1)
    px = (u * (w - 1)).astype(int)
    py = ((1.0 - v) * (h - 1)).astype(int)

    vertex_colors = pixels[py, px, :4]
    avg = vertex_colors.astype(np.float64).mean(axis=0).round().astype(np.uint8)
    return avg


def sanitize_name(name):
    """Make a name safe for use as an MTL material name."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def write_mtl(mtl_path, materials):
    """Write a .mtl file. materials is a list of (name, rgba_uint8) tuples."""
    with open(mtl_path, "w") as f:
        for mat_name, rgba in materials:
            r, g, b, a = rgba.astype(np.float64) / 255.0
            f.write(f"newmtl {mat_name}\n")
            f.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
            f.write(f"d {a:.6f}\n")
            f.write(f"illum 1\n")
            f.write("\n")
    print(f"Wrote {len(materials)} materials to {mtl_path}")


def write_obj_with_mtl(obj_path, mtl_filename, mesh_entries):
    """
    Write a combined OBJ file referencing the given .mtl.
    mesh_entries is a list of (mat_name, vertices, faces) tuples.
    """
    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_filename}\n\n")

        vertex_offset = 0
        for mat_name, verts, faces in mesh_entries:
            f.write(f"usemtl {mat_name}\n")
            f.write(f"o {mat_name}\n")

            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            for face in faces:
                # OBJ faces are 1-indexed, offset by prior meshes
                indices = " ".join(str(idx + 1 + vertex_offset) for idx in face)
                f.write(f"f {indices}\n")

            vertex_offset += len(verts)
            f.write("\n")

    total_verts = sum(len(v) for _, v, _ in mesh_entries)
    print(f"Exported {total_verts} vertices to {obj_path}")


def gltf_to_obj_with_mtl(gltf_path, obj_path):
    scene = trimesh.load(gltf_path)

    obj_dir = os.path.dirname(obj_path)
    obj_base = os.path.splitext(os.path.basename(obj_path))[0]
    mtl_filename = obj_base + ".mtl"
    mtl_path = os.path.join(obj_dir, mtl_filename) if obj_dir else mtl_filename

    materials = []  # (name, rgba)
    mesh_entries = []  # (name, vertices, faces)

    if isinstance(scene, trimesh.Scene):
        for name, geom in scene.geometry.items():
            print(f"Processing sub-mesh: {name} ({len(geom.vertices)} verts)")

            # Apply scene graph transform
            node_names = scene.graph.geometry_nodes.get(name, [name])
            transform = scene.graph.get(node_names[0])[0] if node_names else np.eye(4)
            geom.apply_transform(transform)

            avg_color = compute_avg_color(geom)
            mat_name = sanitize_name(name)

            materials.append((mat_name, avg_color))
            mesh_entries.append((mat_name, geom.vertices, geom.faces))
    else:
        avg_color = compute_avg_color(scene)
        mat_name = "material_0"
        materials.append((mat_name, avg_color))
        mesh_entries.append((mat_name, scene.vertices, scene.faces))

    write_mtl(mtl_path, materials)
    write_obj_with_mtl(obj_path, mtl_filename, mesh_entries)


gltf_to_obj_with_mtl("assets/bistrointerior.glb", "assets/bistrointerior-colored.obj")
