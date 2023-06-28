from pathlib import Path
CFG_DIR = __file__
BASE_DIR = Path(__file__).parent.parent.parent


class SegmenterConfig:
    _MODEL_DIR = ''
    _GRAD_CLIP = 1.

    _EXAMPLE_IMAGES = False

    # device used for training
    _DEVICE = 'cuda'

    # Use embedding space as prototype
    _PROTOTYPE = False

    # Whether the segmenter is doing zero label training
    _ZERO_LABEL = False

    # Whether the segmenter is doing zero label testing
    _TEST_ZERO_LABEL = False

    # Pretrained backbone weights
    _PRETRAINED_MODEL = None
    _DEFAULT_BB_WEIGHTS = False

    # wether continue training from previous session
    _RESUME = False

    # Data configuration
    _HEIGHT = 500  # 513
    _WIDTH = 500 # 513
    _DEPTH = 3
    _MIN_SCALE = 0.5 # 0.5
    _MAX_SCALE = 2.0 # 2.0
    _IGNORE_LABEL = 255

    # Training configuration
    _EPOCHS_PER_EVAL = 1
    _BATCH_SIZE = 5
    _SEGMENTER_DIR = ""
    _SEGMENTER_IDENT = ""
    _TRAIN = False
    _TEST = False
    _VAL = False

    # Learning rate
    _END_LEARNING_RATE = 0
    _POWER = 0.9

    # Optimizer
    _MOMENTUM = 0.9
    _BATCH_NORM_MOMENTUM = 0.9997
    _WEIGHT_DECAY = 1e-4

    # Embedding Function configuration
    _OUTPUT_STRIDE = 16
    _BACKBONE = 'resnet_v2_101'
    _EFN_OUT_DIM = 256  # embedding function output dimension

    # dataset specific
    _NUM_EPOCHS = 0
    _INITIAL_LEARNING_RATE = 0
    _NUM_TRAIN = 0

    # save
    _SAVE_FOLDER = 'saves/default_folder/'

    @property
    def _MAX_ITER(self, ):
        """ Compute lr max iter based on own params. """
        return self._NUM_TRAIN * self._NUM_EPOCHS / self._BATCH_SIZE
