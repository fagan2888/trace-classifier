from pyspark.sql.functions import col
from .preprocessing import preprocessing_part2
from .preprocessing import preprocessing_part3
from .model import construct_model
from .callback import setup_tensor_board
from .callback import setup_model_checkpoint
from .callback import setup_early_stopping
from .callback import setup_save_architecture
from .callback import setup_save_metadata
import numpy as np


def compute_class_weight(df, label_col):
    """
    Computes the weight of each class based on its count in the dataset.

    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame.
    label_col: String.
               Name of the column that contains the integer categorical label.

    Returns
    -------
    A dictionary of weights. The keys are the integer labels, and the values are
    the weight (float) for a label.
    """

    # Total number of records
    n = df.count()

    df_classes = df.groupBy(label_col).count()
    n_classes  = df_classes.count()

    weights = df_classes.select(label_col, (n / (n_classes * df_classes['count'])).alias('recip_freq')) \
                        .toPandas() \
                        .set_index(label_col) \
                        .to_dict()['recip_freq']

    return weights



def train_val_split(df, vfold, instruction):
    """
    Splits df into a training set and a validation set.

    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame.
        Dataframe returned by `preprocessing_part1` function (see preprocessing.py).
    vfold: Integer.
           The fold to reserve as validation set.
    instruction: Dictionary.
                 Parameters for how to process data, either the `PREPROCESS`
                 section of training config json or metadata of a pretrained model.

    Returns
    -------
    Three objects:
    - Training set (pyspark.sql.dataframe.DataFrame),
    - Validation set (pyspark.sql.dataframe.DataFrame),
    - Values used to normalise the datasets (dictionary)
    """

    # Split dataset into two.
    df_trn = df.where(df.fold != vfold)
    df_val = df.where(df.fold == vfold)

    # Process training set.
    df_trn, offset_vals, scale_vals = preprocessing_part2(df_trn,
                                                          instruction,
                                                          offset_vals=None,
                                                          scale_vals=None)

    # Process validation set. Note that this uses normalization parameters
    # `offest_vals` and `scale_vals` computed from the training set.
    df_val, _, _ = preprocessing_part2(df_val,
                                       instruction,
                                       offset_vals=offset_vals,
                                       scale_vals=scale_vals)

    # Save normalization parameters
    params = {
        'offset': offset_vals,
        'scale': scale_vals,
    }

    return df_trn, df_val, params



def get_batches(df, input_shape=(15, 2), batch_size=16, shuffle=True, seed=None):
    """
    An infinite generator, each iteration returns a list of phrases with their labels.

    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame.
    input_shape: Tuple of integers.
                 Shape of the input data (excluding batch size).
    batch_size: Integer.
                Number of phrases per batch.
    shuffle: Boolean.
             Whether to randomly shuffle the dataset.
    seed: Integer (optional).
          Seed integer for random shuffle.


    Returns
    -------
    Two objects:
    - Model input (numpy.ndarray)
    - Array of true integer labels (numpy.array)
    """

    # Helper function
    # (Note: this is kept as a function for future addition of data augmentation.)
    def prepare_inputs(df_in):
        """Process dataframe into numpy arrays."""

        # Form phrases
        df4 = preprocessing_part3(df_in, {'desired_phrase_length': input_shape[0]})

        # Note: this assumes the training set is small an can fit entirely in memeory.
        df4.cache()
        n = df4.count()

        # Convert dataframe into numpy array
        x = np.stack(df4.select('phrase').toPandas().values[:,-1])  # need np.vstack otherwise it's an array of lists
        if len(input_shape) == 3:
            # convert into a single-channel image
            x = np.expand_dims(x, axis=-1)

        # Convert labels into numpy array
        y = df4.select('label').toPandas().values

        # If batch size is larger than dataset, set batch_size to dataset length
        bs = min(n, batch_size)

        indices = np.arange(n)
        df4.unpersist()  # no need to keep the dataframe in memory anymore as relevant data is in numpy arrays

        return x, y, indices, n, bs


    # Initialize
    i = 0
    df2 = df
    x, y, indices, n, bs = prepare_inputs(df2)

    # set seed
    if seed:
        np.random.seed(seed)

    # Infinite loop
    while True:
        if (i >= n):
            i = 0
            if shuffle:
                np.random.shuffle(indices)

        inds = indices[i:i+bs]
        yield x[inds,], y[inds]

        i += batch_size



def train(architecture, config, trn_set, val_set, n_epochs=3, print_summary=True, decorate=False):
    """
    Trains a model.

    Parameters
    ----------
    architecture: Function.
                  A function to construct a Keras Model object.
    config: Dictionary.
            Training configuration.
    trn_set: A pyspark.sql.dataframe.DataFrame.
             Training dataset, which is returned by either the `train_val_split`
             function or `preprocessing_part2` function.
    val_set: A pyspark.sql.dataframe.DataFrame.
             Validation dataset, which is returned by either the `train_val_split`
             function or `preprocessing_part2` function.
    n_epochs: Integer.
              Number of epochs to run.
    print_summary: Boolean.
                   Whether to print model summary.
    decorate: False or a string.
              What print print in the horizontal rule in the log.

    Returns
    -------
    Two objects:
    - Model name (string), which is also the model file name
    - Training history (dictionary)
    """

    # Training set batch generator configuration
    trn_config = config['TRAIN']['BATCH_GENERATOR']
    trn_config['input_shape'] = config['INPUT']['input_shape']

    # Validation set batch generator configuration
    val_config = config['VALIDATE']['BATCH_GENERATOR']
    val_config['input_shape'] = config['INPUT']['input_shape']

    # Create batch generators
    trn_batches = get_batches(trn_set, **trn_config)
    val_batches = get_batches(val_set, **val_config)

    # Construct model
    model = construct_model(architecture=architecture,
                            loss=config['TRAIN']['loss'],
                            metrics=config['TRAIN']['metrics'],
                            optimizer=config['TRAIN']['optimizer'],
                            n_classes=len(config['INPUT']['classes']),
                            input_shape=config['INPUT']['input_shape'],
                            print_summary=print_summary)

    # Decorate
    if decorate:
        print('\n')
        print('**************************************************************************************************')
        print('* ' + decorate)
        print('**************************************************************************************************')

    # Create callback objects
    early_stop               = setup_early_stopping(config['TRAIN']['EARLY_STOPPING'])
    tensorboard, tb_log_name = setup_tensor_board(config['SAVE']['saved_logs_dir'])
    checkpoint, weights_file = setup_model_checkpoint(config['SAVE']['saved_model_dir'], tb_log_name, config['TRAIN']['MODEL_CHECKPOINT'])
    metadata_saver           = setup_save_metadata(config['SAVE']['saved_model_dir'], tb_log_name, config)
    callbacks = [early_stop, checkpoint, metadata_saver, tensorboard]

    if config['TRAIN']['MODEL_CHECKPOINT']['save_weights_only']:
        arch_saver, arch_file = setup_save_architecture(config['SAVE']['saved_model_dir'], tb_log_name)
        callbacks += arch_saver,


    # Train
    # (Note: must use keras version >2.2.1, otherwise will run into this problem https://github.com/keras-team/keras/pull/10673)
    history = model.fit_generator(trn_batches,
                                  **config['TRAIN']['FIT'],
                                  epochs=n_epochs,
                                  validation_data=val_batches,
                                  callbacks=callbacks,
                                  workers=4)

    return tb_log_name, history
