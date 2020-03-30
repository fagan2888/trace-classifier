from .utils import create_label
from .utils import add_id
from .utils import random_int_column
from .word_vec import create_words
from .word_vec import create_word_vecs
from .phrase import create_phrases


def include_id_and_label(df, class_col=None, classes=None, n_folds=1, seed=None):
    """
    Preprocessing (Part 0): Housekeeping.

    Assign each trace a unique ID, and (for training and validation) converts
    string class to integer label.

    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame.
    class_col: String, required for training and validation, but not required for inferencing.
               Name of the column that contains the class name for each trace.
    classes: List of strings, required for training and validation but not for inferencing.
             A list of class names.
    n_folds: Integer (optional).
             Number of folds to randomly assign traces to.
    seed: Integer (optional).
          Seed number for randomly assigning traces to a fold.

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame with new columns
    `id` (integer) for unique trace ID,
    `fold` (integer) for fold number (only if n_folds > 1).
    """


    # Add an id column so that we can split a trace into pieces and still
    # know which piece comes from which trace.
    res_df = add_id(df)

    # NOTE: Inference using the infer module does not follow either branch below
    # Convert class names to integer labels
    if class_col is not None:  # i.e. training and validation data
        assert classes is not None

        # Create label
        res_df = create_label(res_df, class_col, 'label', classes)

    # Assign each trace to a fold
    if n_folds > 1:
        res_df = random_int_column(res_df, 0, n_folds, 'fold', seed=seed)

    return res_df


def include_word_vecs(df, instruction, offset_vals=None, scale_vals=None):
    """
    Preprocessing: Form "word vecs" from "words", "words" from "alphabet".

    A word is a list of N coordinates that may or may not be consecutive, and
    a word vecs is a set of numbers that represents a word.

    offset_vals and scale_vals are only used when data needs to be normalized.
    They are computed during training using *only* the training set only and
    *not* the validation set. Failure to do this results in data leakage;
    validation set should be normalised using the values from the training set.
    See `train_val_split` function in train.py for an example.

    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame
        DataFrame with trace IDs and (if provided) labels
    instruction: Dictionary.
                 Parameters for how to process data, either the `PREPROCESS`
                 section of training config json or metadata of a pretrained model.
    offset_vals: A list of floats (optional).
                 How much to offset each component of a word vec (see `create_word_vecs`
                 function in word_vec.py). Used only when offset_vals are not
                 provided in instruction.
    scale_vals: A list of floats (optional).
                How much to scale each component of a word vec (see `create_word_vecs`
                function in word_vec.py). Used only when offset_vals are not
                provided in instruction.

    Returns
    -------
    Three objects:
    - A pyspark.sql.dataframe.DataFrame with a new column `sentence`.
    - offset_vals used in proprocessing.
    - scale_vals used in proprocessing.
    """

    # include words
    with_words_df = create_words(df, 'coordinates', word_size=instruction['word_size'])

    # Get offset_vals and scale_vals from instruction
    if instruction['normalize']:
        if 'norm_params' in instruction:  # instruction = metadata of a pre-trained model
            offset_vals = instruction['norm_params']['offset']
            scale_vals = instruction['norm_params']['scale']
        else:
            # use offset_vals and scale_vals as provided
            pass

    if 'clip_rng' not in instruction:
        instruction['clip_rng'] = None

    if 'ndigits' not in instruction:
        instruction['ndigits'] = None

    # [Step 2] Create word vecs
    with_word_vecs_df, offset_vals, scale_vals = create_word_vecs(with_words_df,
                                                   'word',
                                                   instruction['desired_ops'],
                                                   normalize=instruction['normalize'],
                                                   offset_vals=offset_vals,
                                                   scale_vals=scale_vals,
                                                   clip_rng=instruction['clip_rng'],
                                                   ndigits=instruction['ndigits'])

    return with_word_vecs_df, offset_vals, scale_vals
