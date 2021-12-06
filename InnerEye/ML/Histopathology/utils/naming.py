from enum import Enum

class ResultsKey(str, Enum):
    SLIDE_ID = 'slide_id'
    TILE_ID = 'tile_id'
    IMAGE = 'image'
    IMAGE_PATH = 'image_path'
    LOSS = 'loss'
    PROB = 'prob'
    PRED_LABEL = 'pred_label'
    TRUE_LABEL = 'true_label'
    BAG_ATTN = 'bag_attn'
