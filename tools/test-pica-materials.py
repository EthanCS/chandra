"""Verify the converted asset against both authored PBR parameter workflows."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
source = json.loads((root / 'resources/pica-pica/published-materials.json').read_text())
asset = json.loads((root / 'external/PicaPica/scene.gltf').read_text())
seen = set()
for material in asset['materials']:
    name = material['name']
    channels = source[name]
    rough, gloss = channels['RoughnessPBR'], channels['GlossinessPBR']
    assert rough['enable'] != gloss['enable'], name
    expected = rough['factor'] if rough['enable'] else 1 - gloss['factor']
    pbr = material['pbrMetallicRoughness']
    assert abs(pbr.get('roughnessFactor', 1) - expected) < 1e-6, (name, 'roughness')
    seen.add('roughness' if rough['enable'] else 'glossiness')
    metal = channels['MetalnessPBR']
    assert abs(pbr.get('metallicFactor', 1) - (metal['factor'] if metal['enable'] else 0)) < 1e-6, (name, 'metalness')
    color = channels['AlbedoPBR']
    expected_color = [v * color['factor'] for v in color.get('color', [1, 1, 1])]
    assert max(abs(a - b) for a, b in zip(pbr.get('baseColorFactor', [1, 1, 1, 1])[:3], expected_color)) < 1e-6, (name, 'linear base color')
    if name.startswith('Decal_'):
        assert material['alphaMode'] == 'MASK' and material['alphaCutoff'] == .5, name
    if name == 'Metal_Aluminium':
        assert 'metallicRoughnessTexture' in pbr, 'Missing authored roughness texture'
assert seen == {'roughness', 'glossiness'}
assert len(asset['materials']) == len(source) == 28
assert len(asset['images']) == 6
for image in asset['images']:
    assert (root / 'external/PicaPica' / image['uri']).is_file(), image
print('PASS: 28 authored materials, both roughness/glossiness workflows, 6 textures, and alpha-mask decals.')
