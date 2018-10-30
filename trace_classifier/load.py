from keras.models import model_from_json
from keras.models import load_model as kload_model
import tensorflow as tf
import h5py
import json
import os


def get_metadata(saved_model_dir, model_name):
    """
    Returns the metadata appended to a model .h5 file.

    Parameters
    ----------
    saved_model_dir: String.
                     Path to a directory to save the best model.
    model_name: String.
                File basename without the extension.

    Returns
    -------
    A dictionary.
    """

    file = os.path.join(saved_model_dir, model_name)

    f = h5py.File(file + '.h5', mode='r')

    if 'metadata' in f.attrs:
        metadata = json.loads(f.attrs['metadata'])
    else:
        metadata = None

    return metadata


def load_model(saved_model_dir, model_name, custom_objects={}):
    """
    Loads a model from a .h5 file. Also returns the metadata required to use this model.

    The .h5 can either consist of weights + architecture + optimizer state, or
    just the weights. If the latter, an architection json file of the same
    file basename is expected.

    Parameters
    ----------
    saved_model_dir: String.
                     Path to a directory where the model is saved.
    model_name: String.
                Model name, which is the file basename (without the extension).

    Returns
    -------
    Two objects:
    - A keras model, not yet compiled.
    - Metadata required to run this model.
    """

    file = os.path.join(saved_model_dir, model_name)

    metadata = get_metadata(saved_model_dir, model_name)


    if metadata['save_weights_only']: # The .h5 file contains only the weights.
        # load architecture
        with open(file + '.json', 'r') as f2:
            model = model_from_json(f2.read(), custom_objects=custom_objects)

        # load weights
        model.load_weights(file + '.h5')

    else: # The .h5 file contains weights + architecture + optimizer state.
        model = kload_model(file + '.h5', custom_objects=custom_objects)

    return model, metadata


def load_model_metadata(model_path):
    """
    Loads the metadata required to run the model in model_path.

    model_path can either be path to a .h5 file or path to a .pb file. Expects
    the file basename to start with the model name.

    If the model file is a .h5 file, metadata is expected to live inside the file.
    If the model file is a .pb file, metadata is expected to live inside a file
    `<model_name>_metadata.json` that's in the same directory as the .pb file.

    Parameters
    ----------
    model_path: String.
                Path to a model (either .h5 or .pb file).

    Returns
    -------
    A dictionary.
    """

    path, ext = os.path.splitext(model_path)
    saved_model_dir, model_name = os.path.split(path)

    metadata = None
    if ext == '.h5':
        metadata = get_metadata(saved_model_dir, model_name)

    else:
        # Model is in the form of: <model_name>_some_other_strings.pb
        # Search for model name by cutting off the filename part by part
        parts = model_name.split('_')
        for i in range(len(parts)-1, -1, -1):
            model_name = '_'.join(parts[:i])
            metadata_file = os.path.join(saved_model_dir, model_name + '_metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                break
    return metadata
