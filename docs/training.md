# Training

We provide scripts to run the training jobs in [experiments/synthetic/scripts](../experiments/synthetic/scripts). These can be launched as below. However, I'd recommend first looking at/modifying the scripts (e.g. specify gpus, or commenting out some experiments) as simply running the jobs  below will launch numerous jobs that will take a few days.

```
cd experiments

#Train upper bounds with stronger supervision
cat synthetic/scripts/upper_bounds | bash

#Train using multi-view supervision with unknown rotation
cat synthetic/scripts/mv_wo_rot | bash

#Train using multi-view supervision with unknown rotation and translation
cat synthetic/scripts/mv_wo_rot_trans | bash
```
