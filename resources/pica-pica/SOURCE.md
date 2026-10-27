# PICA PICA Mini Diorama 01

Assets and material parameters by **SEED / Electronic Arts**, licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

- Geometry and TGA images: [official repository](https://github.com/SEED-EA/pica-pica-assets), commit `b433c2edc0c3120b6d61a9c64daa2d24ebc677a9`.
- `published-materials.json`: author-published material channels from the public [Sketchfab viewer](https://sketchfab.com/3d-models/pica-pica-mini-diorama-01-45e26a4ea7874c15b91bd659e656e30d), retrieved 2026-10-26. Retains linear scalar colors and texture references; contains no model payload or authentication data.
- `tools/setup-pica-pica.ps1` downloads the source and converts it using Blender. Generated models and images stay in ignored `external/PicaPica`.

Changes: FBX-to-glTF conversion, restoration of published metallic/roughness
and glossiness (`roughness = 1 - glossiness`) according to the active channel,
base color and dielectric reflectance, alpha-mask decals. Glass is represented
as an opaque reflective surface. Original HDR lighting and ShaderFX are not
reproduced. These assets are for noncommercial use with attribution.

The first conversion incorrectly treated a disabled roughness channel as zero,
ignoring the active glossiness channel on 13 materials (11 nonzero roughness
values were changed incorrectly). The importer now handles
both workflows; `python tools/test-pica-materials.py` checks all exported factors
and image references against the author-published metadata. Earlier image
records with a different scene hash do not represent the corrected materials.
