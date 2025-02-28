# map and planning resolution
CELL_SIZE = 0.4  # meter
NODE_RESOLUTION = 4.0  # meter
FRONTIER_CELL_SIZE = 4 * CELL_SIZE

# map representation
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# sensor and utility range
SENSOR_RANGE = 20  # meter
UTILITY_RANGE = 0.8 * SENSOR_RANGE
MIN_UTILITY = 1

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION

# training parameters
MAX_EPISODE_STEP = 128

# Graph parameters
K_SIZE = 25  # the number of neighboring nodes
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value