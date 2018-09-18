from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import array
from pyspark.sql.functions import pow as vpow
from pyspark.sql.functions import sqrt as vsqrt
from .utils import explode_array
from .utils import clip
from .utils import round_columns
from .scaler import compute_mean
from .scaler import compute_mad
from .cheap_ruler import cheap_ruler
from itertools import islice


def create_words(df, coordinates_col, word_size, word_pos_col='word_pos', word_col='word'):
    """
    Creates "words" from an array of coordinates.

    A word is a list of N coordinates that may or may not be consecutive.


    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame.
    coordinates_col: String.
                     Name of the column that contains an array of coordinates.
    word_size: Tuple of integers (N, B, S).
               How to create a word.
               N = number of coordinates in a word
               B = stride between coordinates in a word (B=1 for consecutive coordinates)
               S = stride between words (S=N for no overlap between words)
    word_pos_col: String.
                  Name of the new column that contains the position of the word
                  in a sentence (staring from 0).
    word_col: String.
              Name of the new column that contains the word.


    Returns
    -------
    A pyspark.sql.dataframe.DataFrame with one word per record.
    """

    N, B, S = word_size

    @pandas_udf('array<array<array<double>>>')
    def create_words(v):
        """Create an array of words. A word is a list of coordinates."""

        def create(coords):
            alphabets = [islice(coords, i*B, None, S) for i in range(N)]
            words = zip(*alphabets)
            return list(words)

        return v.apply(lambda coords: create(coords))


    othercols = list(set(df.columns) - {coordinates_col})

    # Create an array of words from an array of coordinates.
    df2 = df.withColumn('words', create_words(df[coordinates_col]))

    # Explode the array of words into individual records.
    df3 = explode_array(df2, 'words', [word_pos_col, word_col])

    df4 = df3.select(*othercols, word_pos_col, word_col)

    return df4



def create_word_vecs(df, word_col, desired_ops, word_vec_col='word_vec',
                     normalize=False, offset_vals=None, scale_vals=None, clip_rng=None, ndigits=None):
    """
    Creates a "word vec" from "word".

    A word is a list of N coordinates that may or may not be consecutive, and
    a word vecs is a set of numbers that represents a word.

    Normalizers standardises word vecs by

                                    (original value) - (offset_val)
         (standardized value)  = ------------------------------------
                                              (scale_val)

    Currently only supports 1 normalization method:
    - mean-MAD: offset_val is the mean and scale_val is the MAD


    Paramters
    ---------
    df: A pyspark.sql.dataframe.DataFrame.
    word_col: String.
              Name of the column that contains a word.
    desired_ops: List of tuples, or list of list of tuples.
                 A list of operations to execute on df to get the word vec. Each
                 tuple (OP_NAME, i, j, ...) is an operation where
                   - OP_NAME = name of the operation used in ops_dict,
                   - i, j, ... = parameters to the lambda function for OP_NAME
                 Tuples in the same list are normalized together. See ops_dict
                 in the code for more detail.

    word_vec_col: String.
                  Name of the new word vec column.

    normalize: One of {False, 'mean-mad'}.
               How to normalize word vec.
    offset_vals: List of floats (optional).
                 The offset value for each component in the word vec (used for inferencing).
    scale_vals: List of floats (optional)
                The scale value for each component in the word vec (used for inferencing).
    clip_rng: Tuple of two integers (optional).
              The (min, max) range to clip each component in the word vec. Values
              smaller than min are repaced with min; values greater than max are capped at max.
    ndigits: Integer (optional).
             Number of decimal digits to round each component in the word vec to.


    Returns
    -------
    Three objects:
    - A pyspark.sql.dataframe.DataFrame with the word column replaced by a new word vec column,
    - offset_vals (list of floats)
    - scale_vals (list of floats)
    """

    # Check normalize parameter
    assert normalize in {False, 'mean-mad'}


    othercols = list(set(df.columns) - {word_col})

    df2 = cheap_ruler(df, word_col)


    # A dictionary of operations (features).
    # Must return operations as a single-item list, or else will run into error.
    ops_dict = {
        # Distance project to E-W
        'dx': lambda ii, ff: [(df2[word_col][ff][0] - df2[word_col][ii][0]) * df2.kx],

        # Distance projected to N-S
        'dy': lambda ii, ff: [(df2[word_col][ff][1] - df2[word_col][ii][1]) * df2.ky],

        # Distance
        'd': lambda ii, ff: [vsqrt(vpow((df2[word_col][ff][0] - df2[word_col][ii][0]) * df2.kx, 2) + \
                                   vpow((df2[word_col][ff][1] - df2[word_col][ii][1]) * df2.ky, 2))],

        # Altitude
        'al': lambda ii, ff:[df2[word_col][ff][2] - df2[word_col][ii][2]],

        # Duration
        't': lambda ii, ff: [df2[word_col][ff][3] - df2[word_col][ii][3]],

        # Speed
        's': lambda ii, ff: [(vsqrt(vpow((df2[word_col][ff][0] - df2[word_col][ii][0]) * df2.kx, 2) + \
                                    vpow((df2[word_col][ff][1] - df2[word_col][ii][1]) * df2.ky, 2)) / \
                             (df2[word_col][ff][3] - df2[word_col][ii][3]))]
    }

    # Given a list of desired operations, find the operation in a dictionary and
    # add to an execution plan. Retain group structure for multiple-column
    # normalization.
    ops = []            # a list of operations to be added to the execution plan
    col_grps = []       # a list of column names that stores the result of those operations
    for i, op_grp in enumerate(desired_ops):
        col_grp = []
        for j, op in enumerate(op_grp):
            op_name, ii, ff = op

            # Column to store result of operation
            col_name = '_' + str(i) + '_' + str(j)
            col_grp += col_name,

            # Find the operation in the dictionary and add to ops
            ops += ops_dict[op_name](ii, ff)[0].alias(col_name),

        col_grps += col_grp,

    # Flatten the list
    word_vec_cols = [c for grp in col_grps for c in grp]

    df6 = df2.select(*othercols, *ops)


    # Normalize
    if normalize:
        if scale_vals is None or offset_vals is None:
            # Compute mean and mad for every group
            offset_vals = []
            scale_vals  = []
            for grp in col_grps:
                mu  = compute_mean(df6, grp)
                mad = compute_mad(df6, grp, mean_val=mu)
                offset_vals += [mu] * len(grp)
                scale_vals  += [mad] * len(grp)

        scale_ops = []
        scaled_word_vec_cols = []
        for i, cname in enumerate(word_vec_cols):
            scaled_cname = cname + '_scaled'
            scale_ops += ((df6[cname] - offset_vals[i]) / scale_vals[i]).alias(scaled_cname),
            scaled_word_vec_cols += scaled_cname,

        df7 = df6.select(*othercols, *scale_ops)

        # Clip features
        if clip_rng is not None:
            df7 = clip(df7, scaled_word_vec_cols, clip_rng)

        word_vec_cols = scaled_word_vec_cols

    else:
        df7 = df6
        offset_vals = None
        scale_vals = None


    # Round to desired decimal digits
    if ndigits is not None:
        df7 = round_columns(df7, word_vec_cols, decimals=ndigits)

    # Combine columns into a vec
    df9 = df7.select(*othercols, array(*(df7[col] for col in word_vec_cols)).alias(word_vec_col))

    return df9, offset_vals, scale_vals
