"""Run with Blender --background --python tools/import-pica-pica.py -- SOURCE OUTPUT.

Geometry and textures: SEED / Electronic Arts, CC BY-NC 4.0.
The FBX's generic white materials are replaced by the author's public viewer
parameters. Scalar colors are already linear; texture colors stay sRGB.
"""
import json
import math
import pathlib
import sys
import bpy

root = pathlib.Path(__file__).resolve().parents[1]
source, output = map(pathlib.Path, sys.argv[sys.argv.index('--') + 1:])
source, output = source.resolve(), output.resolve()
output.mkdir(parents=True, exist_ok=True)
materials = json.loads((root / 'resources/pica-pica/published-materials.json').read_text())
texture_names = {
    'Decal_Caution': 'Caution_Decal_01_D.tga',
    'Decal_Keyboard_01': 'Decal_Keyboard_01_D.tga',
    'Decal_First_Aid_Kit': 'First_Aid_Wall_Kit_01_D.tga',
    'Decal_Note_01': 'StickyNote_02.tga',
    'Wax_Pastel_Label_02': 'Wax_Pastel_Label_Yellow_01.tga',
}
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(source / 'Mini_Diorama_01.fbx'), use_custom_normals=True)
for mat in bpy.data.materials:
    channels = materials[mat.name]  # Fail on missing mappings instead of guessing.
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
    out = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    albedo = channels['AlbedoPBR']
    color = [x * albedo.get('factor', 1) for x in albedo.get('color', [1, 1, 1])]
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    rough = channels['RoughnessPBR']
    gloss = channels['GlossinessPBR']
    assert rough['enable'] != gloss['enable'], f'Ambiguous roughness workflow: {mat.name}'
    assert 'texture' not in gloss or not gloss['enable'], 'Gloss texture requires explicit inversion'
    bsdf.inputs['Roughness'].default_value = rough['factor'] if rough['enable'] else 1.0 - gloss['factor']
    metal = channels['MetalnessPBR']
    bsdf.inputs['Metallic'].default_value = metal['factor'] if metal['enable'] else 0
    f0 = 0.08 * channels['SpecularF0']['factor']
    bsdf.inputs['IOR'].default_value = (1 + math.sqrt(f0)) / (1 - math.sqrt(f0))
    if mat.name in texture_names:
        tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex.image = bpy.data.images.load(str(source / texture_names[mat.name]))
        tex.image.colorspace_settings.name = 'sRGB'
        mat.node_tree.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        if mat.name.startswith('Decal_'):
            mat.node_tree.links.new(tex.outputs['Alpha'], bsdf.inputs['Alpha'])
            mat.surface_render_method = 'DITHERED'
    if mat.name == 'Metal_Aluminium':
        tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex.image = bpy.data.images.load(str(source / 'Plastic_04_R.tga'))
        tex.image.colorspace_settings.name = 'Non-Color'
        # glTF stores the texture and scalar factor separately below.
        mat.node_tree.links.new(tex.outputs['Color'], bsdf.inputs['Roughness'])

bpy.ops.export_scene.gltf(filepath=str(output / 'scene.gltf'), export_format='GLTF_SEPARATE',
    export_yup=True, export_animations=False, export_cameras=False, export_lights=False,
    export_normals=True, export_image_format='AUTO')
path = output / 'scene.gltf'
g = json.loads(path.read_text())
for mat in g['materials']:
    if mat['name'].startswith('Decal_'):
        mat['alphaMode'] = 'MASK'
        mat['alphaCutoff'] = 0.5
    if mat['name'] == 'Metal_Aluminium':
        mat['pbrMetallicRoughness']['roughnessFactor'] = materials[mat['name']]['RoughnessPBR']['factor']
g['asset']['copyright'] = 'SEED / Electronic Arts. CC BY-NC 4.0. FBX converted; published PBR material parameters restored.'
path.write_text(json.dumps(g, indent=2))
bounds = [o.matrix_world @ v.co for o in bpy.data.objects if o.type == 'MESH' for v in o.data.vertices]
report = dict(source='https://github.com/SEED-EA/pica-pica-assets',
    commit='b433c2edc0c3120b6d61a9c64daa2d24ebc677a9',
    materials='https://sketchfab.com/3d-models/pica-pica-mini-diorama-01-45e26a4ea7874c15b91bd659e656e30d',
    license='https://creativecommons.org/licenses/by-nc/4.0/',
    blender=bpy.app.version_string,
    bounds_blender_meters=[[min(p[i] for p in bounds) for i in range(3)], [max(p[i] for p in bounds) for i in range(3)]],
    mesh_objects=sum(o.type == 'MESH' for o in bpy.data.objects),
    materials_count=len(g['materials']),
    limitations=['Opaque approximation for glass; no transmission.', 'Original HDR environment and ShaderFX are not included.'])
(output / 'SOURCE.json').write_text(json.dumps(report, indent=2))
(output / 'ATTRIBUTION.txt').write_text('PICA PICA Mini Diorama 01 — SEED / Electronic Arts\nCC BY-NC 4.0\nhttps://github.com/SEED-EA/pica-pica-assets\nhttps://creativecommons.org/licenses/by-nc/4.0/\nConverted FBX to glTF; restored published material parameters.\n')
print(json.dumps(report))
