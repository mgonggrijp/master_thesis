import json as json_lib
from hesp.hierarchy.tree import Tree
from hesp.models.segmenter import Segmenter


def model_factory(config):
    """ Initializes and returns a model based on config and mode."""
    # initialize tree containing target class relationships
    if config.embedding_space._HIERARCHICAL:
        json = json_lib.load(open(config.dataset._JSON_FILE))
        
    else:
        json = {}
        
    device = config.segmenter._DEVICE
        
    tree_params = {'i2c': config.dataset._I2C,
                   'json': json,
                   'device' : device}
    
    class_tree = Tree(**tree_params)
     
    train_embedding_space = True
    prototype_path = ''
    # initialize model

    model_params = {
        'tree': class_tree,
        'config': config,
        'save_folder': config.segmenter._SAVE_FOLDER,
        'seed' : config.segmenter._SEED,
        'device' : device}
    
    return Segmenter(**model_params)
