# UNet

## Current Results

1. **Prediction**
   ![Prediction](https://github.com/Kai-MC/UNet/assets/100511674/258888ca-0025-4a0c-aace-d57bcec03aa7)

   This image shows the prediction results from the model.

2. **Distribution of Prediction**
   ![Distribution of Prediction](https://github.com/Kai-MC/UNet/assets/100511674/e2d8854a-9db0-4a90-aff9-9f9c55e7cbad)

   This image illustrates the distribution of the predictions.

3. **Items in the Data Loader**
   ![Items in the Data Loader](https://github.com/Kai-MC/UNet/assets/100511674/3dbcbb9c-fe82-40df-9884-5a033440963f)

   This image displays the items present in the data loader.


## What is new

1. I implemented a new data pipeline in `multiunet.ipynb` to evaluate the correctness of the previous pipeline. The both pipeline should work in the same way. I used `multiunet.ipynb` to train the new model.
3. The updated France notebook is `france_trainmodel_new.ipynb`

## Potential Bug

Model
1. Mismatch in the heads `head_cmtsk`. As some boundary feature is extracted well in the image of Distance prediction.
2. Pooling. Stitch is complicated.

Pipeline
1. Loss function. The function is not found in the original mxnet notebook, it is instead imported.
2. Distance calculation. Not sure abt 1-dist or dist.


The original structure: https://github.com/waldnerf/decode/
