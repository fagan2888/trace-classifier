from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.callbacks import Callback
import datetime
import h5py
import json
import os


def setup_tensor_board(saved_logs_dir):
    """
    Sets up TensorBoard. See https://keras.io/callbacks/#tensorboard.

    Parameters
    ----------
    saved_logs_dir: String.
                    Path to a directory to save tensorboard logs.

    Returns
    -------
    Two objects:
    - A TensorBoard object,
    - Log name (name of the subfolder under saved_logs_dir which contains the tfevents.MBF file).
    """

    today = datetime.date.today().strftime('%Y%m%d')
    todays_saved_log_dir = os.path.join(saved_logs_dir, today)
    os.makedirs(todays_saved_log_dir, exist_ok=True)

    existing_logs = [log for log in os.listdir(todays_saved_log_dir) if today in log]

    tb_counter = len(existing_logs) + 1

    tb_log_name = '{}-{}'.format(today, tb_counter)
    tensorboard = TensorBoard(log_dir=os.path.join(todays_saved_log_dir, tb_log_name),
                              histogram_freq=0,
                              write_graph=True,
                              write_images=False)

    return tensorboard, tb_log_name


def setup_model_checkpoint(saved_model_dir, model_name, config):
    """
    Sets up model checkpoint to periodically save the model weights with the best performance.

    The metric to compare performance and what to save (see below) is specified in config.

    By Keras' default, save_weights_only = False, i.e. model .h5 contains weights +
    architecture + optimizer state. To use a pretrained model, all we need is weights
    + architecture; optimizer state is only used when we want to further train the
    model from its existing state.

    If config['save_weights_only'] = True, the model .h5 will only contain the weights;
    architecture will need to be saved separately using the setup_save_architecture
    callback function.

    The metric to monitor for best performance is specified by config['monitor'].
    Period is set to 1 (i.e. every epoch).

    Parameters
    ----------
    saved_model_dir: String.
                     Path to a directory to save the best model.
    model_name: String.
                File basename (exclude extension) to save the model weights as.
    config: Dictionary.
            Keras ModelCheckpoint arguments. See https://keras.io/callbacks/#modelcheckpoint.

    Returns
    -------
    Two objects:
    - A ModelCheckpoint object,
    - Full path (<saved_mode_dir>/<model_name>.h5) to the saved model weights.
    """

    # Create folder if not exist
    os.makedirs(saved_model_dir, exist_ok=True)

    weights_file = os.path.join(saved_model_dir, model_name + '.h5')

    checkpoint = ModelCheckpoint(weights_file,
                                 **config,
                                 save_best_only=True,
                                 period=1,
                                 verbose=1)

    return checkpoint, weights_file



def setup_save_architecture(saved_model_dir, model_name, verbose=True):
    """
    Sets up SaveArchitecture to save the model architecture at the end of training.

    Parameters
    ----------
    saved_model_dir: String.
                     Path to a directory to save the best model.
    model_name: String.
                File basename (exclude extension) to save the model architecture as.
    verbose: Boolean.
             Print path to the saved file.

    Returns
    -------
    Two objects:
    - A SaveArchitecture object
    - Full path (<saved_mode_dir>/<model_name>.json) to the saved model architecture.
    """

    class SaveArchitecture(Callback):
        def __init__(self, filepath, verbose=True):
            self.filepath = filepath
            self.verbose  = verbose

        def on_train_end(self, logs=None):
            # Update model with metadata.
            with open(self.filepath, 'w') as f:
                f.write(self.model.to_json())

            if self.verbose:
                print('Saved model architecture to ' + self.filepath)

    # Create folder if not exist
    os.makedirs(saved_model_dir, exist_ok=True)

    architecture_file = os.path.join(saved_model_dir, model_name + '.json')

    saver = SaveArchitecture(architecture_file, verbose=verbose)

    return saver, architecture_file


def setup_early_stopping(config):
    """
    Sets up early stopping to terminate training when metric does not improve by
    at least a certain amount.

    The metric to monitor and minimum required improvement required are specified
    in config.

    Parameters
    ----------
    config: Dictionary.
            Keras EarlyStopping arguments. See https://keras.io/callbacks/#earlystopping.

    Returns
    -------
    An EarlyStopping object.
    """

    return EarlyStopping(**config, verbose=1)


def setup_save_metadata(saved_model_dir, model_name, config):
    """
    Sets up AddMetadataToHDFS to add additional metadata to saved model HDFS file
    at the end of training.

    Parameters
    ----------
    saved_model_dir: String.
                     Path to a directory to save the best model.
    model_name: String.
                File basename (exclude extension) of the saved model.
    config: Dictionary.
            Model training config.

    Returns
    -------
    An AddMetadataToHDFS object.
    """

    class AddMetadataToHDFS(Callback):
        def __init__(self, filepath, metadata, name='metadata'):
            self.filepath = filepath
            self.metadata = metadata
            self.name     = name

        def on_train_end(self, logs=None):
            with h5py.File(self.filepath, mode='a') as fp:
                fp.attrs[self.name] = json.dumps(self.metadata)


    weights_file = os.path.join(saved_model_dir, model_name + '.h5')

    # Metadata to save
    metadata = {**config['PREPROCESS'], **config['INPUT']}
    metadata['save_weights_only'] = config['TRAIN']['MODEL_CHECKPOINT']['save_weights_only']

    saver = AddMetadataToHDFS(weights_file, metadata)

    return saver

