# Evaluation

We provide scripts to run the training jobs in [benchmark/synthetic/scripts](../benchmark/synthetic/scripts). These can be launched as below.

## Prediction Frame Alignment
Since the predictions using multi-view without pose supervision are arbitrary aligned, we first need to compute the optimal translation and rotation to bring them to the ShapeNet frame.

```
cd benchmark
cat synthetic/scripts/compute_alignment | bash
```

## Rotation Evaluation
```
cd benchmark
cat synthetic/scripts/eval_rot | bash
```
The median errors and prediction accuracy for various settings will be saved in cachedir/resultsDir/pose/shapenet.

## Shape Evaluation
To compute the mean IoU across categories for the different experiments -
```
cd benchmark
#save shape predictions
cat synthetic/scripts/eval_shape | bash

#evaluate shape predictions
cd benchmark/synthetic/
matlab -nodesktop -nosplash
>> evalShape
```
