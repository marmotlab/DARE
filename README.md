# DARE
Code for Diffusion Policy to Autonomous Robotics Exploration (DARE)
Paper available @ https://arxiv.org/abs/2410.16687v1

## Installation
Create conda environment from `environment.yml`
```
conda env create -f environment.yml
```

Activate the environment

```
conda activate env_dare
```

## Training
### Dataset Collection
Modify `dataset_parameter.py` to fit your dataset needs.

Run `dataset_driver.py`
```
python dataset_driver.py
```

Dataset will be saved to directory `diffusion_exploration/dataset/name_of_test`.
It will include a `data.zarr` directory which contains the dataset and a `gifs` directory.

### Running the Training Script
Copy desired training config file from `diffusion_exploration/diffusion_policy/config`

Modify desired task config file from `diffusion_exploration/diffusion_policy/config/task`
**Note:** You probably should modify the `zarr_path` to change dataset location

You can run the training script which requires two arguements:
1. `--config-dir` which is the directory to find the config file
2. `--config-name` which is the name of the config file

Example:
```
python train.py --config-dir=. --config-name=train_exploration_transformer_node_discrete.yaml
```

This will create a directory `diffusion_exploration/data/date/time/name_of_run`

## Test
Modify `test_parameter.py` to fit your test needs.

Run `test_driver.py`
```python test_driver.py```

Test results will be printed on terminal and saved as a CSV
`inference_gifs` directory will be created in `diffusion_exploration/data/date/time/name_of_run`.
