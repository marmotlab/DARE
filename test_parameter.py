## Test Options
TEST_METHOD = 'DARE' # It's only DARE
DATA_TYPE = 'node'  # 'map', 'node'
USE_TEST_DATASET = True  # False for train dataset, True for test dataset
USE_DELTA_POSITION = True # False for absolute position, True for delta position
USE_EXPLORATION_RATE_FOR_DONE = False  # False for robot util == 0, True for exploration rate
TEST_N_AGENTS = 1 # SINGLE AGENT keep it 1

## Environment Runner Options
USE_GPU = True  # do you want to use GPUS?
NUM_GPU = 1  # the number of GPUs
NUM_META_AGENT = 10  # the number of processes

NUM_TEST = 100
NUM_RUN = 1
SAVE_GIFS = True  # do you want to save GIFs

ACTION_HORIZON = None # None for 1 horizon

## Name and Path
run_path = f'runs/best'
checkpoint_name = 'epoch=0130-val_loss=0.056.ckpt'
gifs_path = f'{run_path}/gifs'