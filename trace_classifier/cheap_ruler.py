import numpy as np
from pyspark.sql.functions import col
from pyspark.sql.functions import cos


def cheap_ruler(df, coordinates_col="coordinates"):
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

    m = 1000  # meters
    # require latitude of first coordinate
    return (
        df.withColumn("cos1", cos(df[coordinates_col][0][1] * np.pi / 180))
        .withColumn("cos2", 2 * col("cos1") * col("cos1") - 1)
        .withColumn("cos3", 2 * col("cos1") * col("cos2") - col("cos1"))
        .withColumn("cos4", 2 * col("cos1") * col("cos3") - col("cos2"))
        .withColumn("cos5", 2 * col("cos1") * col("cos4") - col("cos3"))
        .withColumn(
            "kx",
            m
            * (111.41513 * col("cos1") - 0.09455 * col("cos3") + 0.00012 * col("cos5")),
        )
        .withColumn(
            "ky", m * (111.13209 - 0.56605 * col("cos2") + 0.0012 * col("cos4"))
        )
        .drop("lat", "cos1", "cos2", "cos3", "cos4", "cos5")
    )
