## Towards Few-Call Model Stealing via Active Self-Paced Knowledge Distillation and Diffusion-Based Image Generation  - Official Implementation

### Setup
Install requirements

``` pip install -r requirements.txt ```

If you want access to our pretrained models (teachers - trained on the true datasets, and students - trained on TinyImagenet200), please download them using:
 ``` bash download_checkpoints.sh ```


### Models
We provide the implementation for our method from the paper, along with the models we used for the experiments. These are:
Students:
    - HalfAlexNet
    - Resnet18
Teachers:
    - AlexNet
    - Resnet18
    - Resnet50


### Experiments can then be run, as follows:
1. Create the proxy dataset using any diffusion model.
2. Add the images in a follder called "images_generated_DATASET_NAME" (you can change DATASET_NAME as you please). The folder structure should be: 2 folders (train and valid) that contains subfolder named after the class with the respective images, as well as 2 json files that contain the labels.
3. Run the following command:

```python start_experiment.py --epochs 200 --batch_size 64 --lr 0.001 --step_size 20 --dataset DATASET_NAME --student resnet --teacher alexnet --use_soft_labels True --use_active_learning True --use_og_labels True --distance cosine --use_all_data True ```

### Acknowledgements
Our work was based on "Black-Box Ripper: Copying black-box models using generative evolutionary algorithms". We forked from their repository, which is available here: https://github.com/antoniobarbalau/black-box-ripper.
