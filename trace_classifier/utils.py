import numpy as np
from pyspark.sql.functions import col
from pyspark.sql.functions import floor
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import posexplode
from pyspark.sql.functions import rand
from pyspark.sql.functions import round as vround
from pyspark.sql.functions import udf
from pyspark.sql.functions import when
from pyspark.sql.types import ArrayType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType


def getGeojsonSchema():
    """
    Returns the schema of a GeoJSON LineString feature.

    Parameters
    ----------
    None

    Returns
    -------
    A pyspark.sql.dataframe schema.
    """

    schema = StructType(
        [
            StructField(
                "geometry",
                StructType(
                    [
                        StructField("coordinates", ArrayType(ArrayType(DoubleType()))),
                        StructField("type", StringType()),
                    ]
                ),
            ),
            StructField(
                "properties", StructType([StructField("modality", StringType())])
            ),
            StructField("type", StringType()),
        ]
    )

    return schema


def round_columns(df, cols, decimals=0):
    """
    Rounds the values in one or more columns.

    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame.
    cols: List of strings.
          A list of column names whose values are to be rounded off.
    decimals: Integer or a list of integers.
              Number of digits to round off. If a list, the list must be the
              same length as cols (i.e. each column has a corresponding item
              in the decimals list).

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame.
    """

    othercols = list(set(df.columns) - set(cols))

    # Convert decimals into a list of the same length as cols
    if isinstance(decimals, int):
        decimals = [decimals] * len(cols)

    assert len(cols) == len(decimals)

    ops = []
    for i, cname in enumerate(cols):
        ops += (vround(col(cname), decimals[i]).alias(cname),)

    return df.select(*othercols, *ops)


def clip(df, cols, rng):
    """
    Clips the values in one or more columns to the desired range. All values
    greater than the desired range maximum is replaced with the desired range
    maximum; all values smaller than the desired range minimum is replaced with
    the desired range minimum.

    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame.
    cols: List of strings.
          Name of the columns to be rounded off.
    rng: Tuple of two numbers (integer or float), or a list of tuples of two numbers.
         The desired (min, max) range to clip values. If a list of tuples, the
         list must be the same length as cols (i.e. each column has a corresponding
         item in the rnd list).

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame.
    """

    othercols = list(set(df.columns) - set(cols))

    # convert rng into a list of the same length as cols
    if not isinstance(rng[0], list) and not isinstance(rng[0], tuple):
        rng = [rng] * len(cols)

    assert len(cols) == len(rng)

    ops = []
    for i, cname in enumerate(cols):
        ops += (
            when(col(cname) < rng[i][0], rng[i][0])
            .when(col(cname) > rng[i][1], rng[i][1])
            .otherwise(col(cname))
            .alias(cname),
        )

    return df.select(*othercols, *ops)


@pandas_udf("int")
def argmax(v):
    """
    Returns the argmax of an array.

    Parameters
    ----------
    v: A pyspark.sql.column.Column.
       A column whose elements are arrays of numbers (<array<float>>).

    Returns
    -------
    A pyspark.sql.column.Column of the argmax (index of the max value in the array).
    """

    return v.apply(lambda arr: np.argmax(arr, axis=-1))


@udf("array<array<float>>")  # Tensorframes requires float instead of double
def pad(arr, n):
    """
    Prepends an array with zero vector.

    Parameters
    ----------
    arr: A pyspark.sql.column.Column.
         Phrase column (array<array<double>>).
    n: A pyspark.sql.column.Column.
       Number of zero vectors to prepend.

    Returns
    -------
    A pyspark.sql.column.Column of zero-padded phrase (array<array<float>>).
    """

    return np.pad(arr, [(n, 0), (0, 0)], "constant", constant_values=0.0).tolist()


def create_label(df, class_col, label_col, classes):
    """
    Creates an integer label column from a string class column.

    Note that df cannot already contain a column named label_col.

    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame.
    class_col: String.
               Name of the column that contains the class name of each record.
    label_col: String
               Name of the column to be created that will contain the integer label of each record.
    classes: List of strings.
             A list of classes. Position of the class in the list is its integer label.

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame with integer labels.
    """

    # Integer labels
    labels = list(map(str, range(len(classes))))

    # Replace only works for same data type
    return (
        df.replace(classes, labels, subset=class_col)
        .withColumn(label_col, col(class_col).cast("integer"))
        .drop(class_col)
    )


def reverse_create_label(df, label_col, class_col, classes):
    """
    Creates a class column from integer labels.

    Note that df cannot already contain a column named class_col.

    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame.
    label_col: String.
               Name of the column that contains the integer label of each record.
    class_col: String.
               Name of the column to be created that will contain the string class name of each record.
    classes: List of strings.
             A list of classes. Position of the class in the list is its integer label.

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame with an integer label for each record.
    """

    # Integer labels
    labels = list(map(str, range(len(classes))))

    # Replace only works for same data type
    return (
        df.withColumn(class_col, col(label_col).cast("string"))
        .replace(labels, classes, subset=class_col)
        .drop(label_col)
    )


def add_id(df, id_col="id"):
    """
    Adds a unique identifier column.

    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame.
    id_col: String.
            Name of the new unique identifier column to be created.

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame with a unique id for each record.
    """

    return df.withColumn(id_col, monotonically_increasing_id())


def random_int_column(df, min_val, max_val, newcol, seed=None):
    """
    Adds a column of i.i.d. random integers.

    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame.
    min_val: Integer.
             Min limit (inclusive) to randomly drawn integer.
    max_val: Integer.
             Max limit (exclusive) to randomly drawn integer.
    newcol: String.
            Name of the new random integers column to be created.
    seed: Integer (optional).
          Seed for rand.

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame with a new random integers column.
    """

    return df.withColumn(newcol, floor(rand(seed=seed) * (max_val - min_val) + min_val))


def explode_array(df, target_col, newcols):
    """
    Explodes an array into one row per item.

    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame.
    target_col: String.
         Name of the array column to explode.
    newcols: Tuple of two strings.
             New columns (position, item) to store array items and their position
             in the original array.

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame with col column replaced by two new columns.
    """

    othercols = list(set(df.columns) - {target_col})

    return df.select(*othercols, posexplode(target_col).alias(*newcols))
