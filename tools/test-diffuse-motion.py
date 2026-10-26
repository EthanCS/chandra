"""Receiver correspondence and reference-footprint tests, without a GPU."""
from pathlib import Path
import runpy
import unittest
import numpy as np

MODULE=runpy.run_path(str(Path(__file__).with_name("validate-diffuse-motion.py")))

class ReceiverTests(unittest.TestCase):
    def setUp(self):
        y,x=np.mgrid[:5,:5]
        self.position=np.stack([x-2,y-2,np.ones_like(x),np.ones_like(x)],axis=-1).astype(np.float32)
        self.normal=np.zeros_like(self.position);self.normal[...,2:]=1

    def test_reference_support_never_crosses_a_receiver_edge(self):
        value=np.zeros((5,5),np.float32);value[2,3]=1
        for mode in ("instance","normal","parallel"):
            position=self.position.copy();normal=self.normal.copy()
            if mode=="instance":position[2,3,3]=2
            if mode=="normal":normal[2,3,:3]=[1,0,0]
            if mode=="parallel":position[2,3,2]=2
            _,high=MODULE["surface_reference_bounds"](value,position,normal)
            self.assertEqual(high[2,2],0,mode)
        _,high=MODULE["surface_reference_bounds"](value,self.position,self.normal)
        self.assertEqual(high[2,2],1,"Same receiver's one-pixel support was lost")

    def test_reprojection_preserves_same_points_and_rejects_parallel_surfaces(self):
        value=np.arange(25,dtype=np.float32).reshape(5,5)
        view={"position":[0,0,0],"transform_columns":np.eye(3).tolist(),"scaling":1}
        warped,valid=MODULE["reproject"](self.position,self.normal,self.position,self.normal,view,value)
        np.testing.assert_array_equal(warped[:4,:4],value[:4,:4])
        self.assertTrue(valid[:4,:4].all())
        previous=self.position.copy();previous[...,2]=2
        _,valid=MODULE["reproject"](self.position,self.normal,previous,self.normal,view,value)
        self.assertFalse(valid.any())

if __name__=="__main__":unittest.main()
