"""
all parameters used in dnn model are defined in this file
it should be constant value
"""

# the down size for training images
DOWN_SIZE = 4

# the number of features detected by spp keypoint method
NUM_FEAT_DETECT = 400

# the number of the input feature sampling
NUM_FEAT_INPUT = 200

# times of data sampling from detected features
SAMPLES = 5

# the length of input feature, the spp feature length
SPP_FEAT_LEN = 340

# the length of keypoint features output from point feature net
OUT_FEAT_LEN = 128

# the length of image gloabl feature
GLOBAL_FEAT_LEN = 64
