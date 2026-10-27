"""Numerical parity for shared reprojection/visibility work in the motion checker."""
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('motion',Path(__file__).with_name('validate-specular-motion.py'))
motion=importlib.util.module_from_spec(spec);spec.loader.exec_module(motion)
rng=np.random.default_rng(9182)
h,w=32,48;y,x=np.mgrid[:h,:w]
position=np.zeros((h,w,4),np.float32)
position[...,0]=(x+.5-w*.5)*.15;position[...,1]=(y+.5-h*.5)*.15
position[...,2]=5;position[...,3]=1
normal=np.zeros_like(position);normal[...,2]=1
previous=position.copy();previous[...,0]+=.04
previous[3:8,9:12,3]=2;previous[15:18,15:20,2]+=.2
view=dict(position=[0,0,0],transform_columns=np.eye(3).tolist(),scaling=.03)
warp,valid=motion.reprojection(position,normal,previous,normal,view)
support=motion.surface_support(position,normal)
assert valid.any() and not valid.all()
for _ in range(6):
 value=rng.random((h,w),dtype=np.float32)
 expected,expected_valid=motion.checks.reproject(position,normal,previous,normal,view,value)
 np.testing.assert_array_equal(valid,expected_valid)
 np.testing.assert_array_equal(warp(value),expected)
 low,high=motion.reference_bounds(value,support)
 elo,ehi=motion.checks.surface_reference_bounds(value,position,normal)
 np.testing.assert_array_equal(low,elo);np.testing.assert_array_equal(high,ehi)
print('PASS: all six channels preserve exact reprojection and reference bounds.')
