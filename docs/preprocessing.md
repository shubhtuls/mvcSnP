# Preprocessing

## Rendering
To render RGB and depth images for ShapeNet models, specify the ShapeNetV1 folder [here](../preprocess/synthetic/rendering/startup.py) and the path to blender [here](../preprocess/synthetic/rendering/renderer/global_variables.py) and [here](../preprocess/synthetic/rendering/rendererTrans/global_variables.py). Then, run
```
#Rendering chairs, cars and aeroplanes (takes about a day)
cd preprocess/synthetic/rendering
python renderPreprocessShapenet.py

#Rendering chairs, cars and aeroplanes with random translations (takes about a day)
python renderPreprocessShapenetTrans.py
```

## Computing Voxelizations
For evaluation and training the 3D-supervised baseline, we need to compute the groun-truth 3D voxelizations. First, modify the path to ShapeNetV1 [here](../preprocess/synthetic/voxelization/startup.m) and then run
```
#Computing Gt Voxelizations
cd preprocess/synthetic/voxelization
matlab -nodesktop -nosplash
>> precomputeVoxels
```