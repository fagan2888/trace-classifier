from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import cos
import numpy as np


def cheap_ruler(df, coordinates_col='coordinates'):
    """
    Computes the multipliers for converting longitude and latitude degrees into distance in km.

    See
    - Code https://github.com/mapbox/cheap-ruler/blob/master/index.js#L55-L70.
    - Explanation https://www.mapbox.com/blog/cheap-ruler/

    Parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame.
    coordinates_col: String.
                     Name of the column that contains the coordinates in array<array<double>> format.

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame with two new columns added:
    `kx` (double type) for the multiplier in the x direction, and
    `ky` (double type) for the multiplier in the y direction.
    """

    df = df.withColumn('cos1', cos(df[coordinates_col][0][1] * np.pi / 180))  # latitude of first coordinate
    df = df.withColumn('cos2', 2 * df.cos1 * df.cos1 - 1)
    df = df.withColumn('cos3', 2 * df.cos1 * df.cos2 - df.cos1)
    df = df.withColumn('cos4', 2 * df.cos1 * df.cos3 - df.cos2)
    df = df.withColumn('cos5', 2 * df.cos1 * df.cos4 - df.cos3)

    m = 1000  # meters
    df = df.withColumn('kx', m * (111.41513 * df.cos1 - 0.09455 * df.cos3 + 0.00012 * df.cos5))
    df = df.withColumn('ky', m * (111.13209 - 0.56605 * df.cos2 + 0.0012 * df.cos4))
    df = df.drop('lat', 'cos1', 'cos2', 'cos3', 'cos4', 'cos5')

    return df
