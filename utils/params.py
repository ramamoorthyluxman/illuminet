ACQ_PATH = r'/work/imvia/ra7916lu/illuminet/data/subset/painting1/dense/train'

# Dataset params
MAX_NB_IMAGES_PER_ACQ = 105
COLLIMATED_LIGHT = True 

# SURFACE_PHYSCIAL_SIZE = [(1024*0.250)/2704, (1024*0.160)/1800] #default [0.250, 0.120]
# SURFACE_PHYSCIAL_SIZE = [0.250, 0.160] #default [0.250, 0.160] #width , height
SURFACE_PHYSCIAL_SIZE = [0.05, 0.0421] #default [0.250, 0.120]


# RTI training
RTI_NET_EPOCHS = 1000
RTI_NET_SAVE_MODEL_EVERY_N_EPOCHS = 10

# Model/results paths
RTI_MODEL_PATH = r'/work/imvia/ra7916lu/illuminet/data/subset/painting1/dense/relight_results/illumi-net/model/saved_models_20241231_151829/best_model.pth'
RTI_MODEL_SAVE_DIR = r'/work/imvia/ra7916lu/illuminet/data/subset/painting1/dense/relight_results/illumi-net/model'

# Goal
TRAINING = True # Training: True, Relighting: False. If you choose relighting- ensure the path contains - 1. distances.npy, 2. cosines.npy, 3. albedo and 4. normals


# Model Params
BATCH_SIZE = 4
TRAIN_SHUFFLE = True
VAL_SHUFFLE = False
LEARNING_RATE = 0.001
TRAIN_VAL_SPLIT = 0.2