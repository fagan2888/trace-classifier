import json
import os

MODEL_INPUT_CONFIG = {
  "ID_COL": "id",
  "WORD_VEC_COL": "word_vec",
  "WORD_POS_COL": "word_pos",
  "INPUT_COL": "phrase"
}


def class_weights_todict(config):
    """
    Converts class weights from an array (list, tuple or numpy array) to a dictionary
    as required by Keras' fit_generator.

    Expects to find the class weights in config['TRAIN']['FIT']['class_weight'].

    Parameters
    ----------
    config: Dictionary.
            Configuration for training a model.

    Returns
    -------
    Configuration with model class weight as a dictionary.
    """

    # Convert class weight from array to dictionary
    if not isinstance(config, dict):
        d = {i: w for i, w in enumerate(config['TRAIN']['FIT']['class_weight'])}
        config['TRAIN']['FIT']['class_weight'] = d

    return config


def remove_field(config, field):
    """
    Removes a field from config.

    Parameters
    ----------
    config: Dictionary.
            Configuration for training a model.

    Returns
    -------
    Configuration with the target field removed.
    """

    config2 = config.copy()

    # in-place deletion
    def delete_key(d, key):
        if key in d:
            del d[key]
        for v in d.values():
            if isinstance(v, dict):
                delete_key(v, key)

    delete_key(config2, field)

    return config2


def add_input_dimension(config):
    """
    Computes model input shape based on configurations in config['PREPROCESS']
    and adds this config['INPUT'].

    Parameters
    ----------
    config: Dictionary.
            Configuration for training a model.

    Returns
    -------
    Configuration with model input dimension added.
    """

    # ndims = 1 for 1D signal. Example: temperature.
    # ndims = 2 for 2D signal. Example: R-band in an image.
    ndims = config['INPUT']['ndims']

    # For 1D signal, this is equal to the number of signals.
    # For 2D signal, this is analogous to the width of an image.
    nops  = sum(1 for grp in config['PREPROCESS']['desired_ops'] for _ in grp)

    # For 1D signal, this is equal to the length of the signals.
    # For 2D signal, this is analogous to the height of an image.
    nrows = config['PREPROCESS']['desired_phrase_length']

    input_shape = [nrows, nops]

    if ndims == 2:
        # Add channel to become a single-channel image
        input_shape += 1,

    config['INPUT']['input_shape'] = input_shape

    return config



def load_config(config=None):
    """
    Returns the training config.

    Parameters
    ----------
    config: Dictionary or String.
            Dictionary of training config, or path to a json config file.
            If not provided, returns the sample config.

    Returns
    -------
    A dictionary.
    """

    if not config:
        dir, _ = os.path.split(__file__)
        config = os.path.join(dir, 'sample_model/config.json')

    if isinstance(config, str):
        with open(config, 'r') as f:
            config = json.load(f)

    # Sample config file contains descriptions for ease of comprehension, but some
    # config items are arguments passed to Keras functions as is. Need to remove
    # these description fields to avoid `unexpected keyword argument` error.
    config = remove_field(config, 'DESCRIPTION')

    # Ensure class weights are in dictionary form as requred by Keras
    config = class_weights_todict(config)

    # Compute mode
    config = add_input_dimension(config)

    return config
