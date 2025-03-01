## Dataset Options
DATASET_METHOD = 'ground_truth'  # 'tare', 'ground_truth', 'ground_truth_no_replan'
DATA_TYPE = 'node'  # 'map', 'node'
USE_TEST_DATASET = False  # False for train dataset, True for test dataset
USE_DELTA_POSITION = True # False for absolute position, True for delta position
TEST_N_AGENTS = 1 # SINGLE AGENT keep it 1

## Environment Runner Options
USE_GPU = False  # do you want to use GPUS?
NUM_GPU = 0  # the number of GPUs
NUM_META_AGENT = 24  # the number of processes

NUM_EPISODES = 4000 # number of episodes
NUM_RUN = 1
SAVE_GIFS = False  # do you want to save GIFs

## Name and Path
DESCRIPTION = ''
dataset_path = f'dataset/{DATASET_METHOD}_{DATA_TYPE}_{"test" if USE_TEST_DATASET else "train"}_{NUM_EPISODES}{"_" + DESCRIPTION if DESCRIPTION else ""}'
gifs_path = f'{dataset_path}/gifs'