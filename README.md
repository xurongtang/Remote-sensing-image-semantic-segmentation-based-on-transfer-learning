# Remote-image-segmentation-based-on-transfer-learning
Identify the colored steel houses along the railway to ensure the safety of railway operation. However, since there are fewer colored steel house datasets and a large number of remote sensing image data building datasets, we now train on the building dataset and then use transfer learning to fine-tune on the colored steel house dataset.

# Result Preview(A simple GUI)
## This is segmentation result of pre-trained model training with building dataset
![](/img/pre-trained.png "pre-trained")

## This is segmentation result of fine tune model training with specific dataset
![](/img/fine_tune.png "fune-tune")

# Json data convert
If your dataset label save as json format, you can run this python script to get image label.
> JSON tool/json_label_trans.py

# Create a segmentation model with pre-trained encoder by using pytorch
Firstly, we can use the [smp](https://github.com/qubvel/segmentation_models.pytorch) to construct our model.
how to use the smp build a complete segmentation model, just see the following python script.
>main.py

# fine tune
When there is a pre-trained model, we can use the transfer learning method to fine tune our model with specific dataset.
reading the following script in detail.
>finetune_model.py 

# test
