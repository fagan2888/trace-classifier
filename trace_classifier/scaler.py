from pyspark.sql.functions import col, lit
from pyspark.sql.functions import mean as vmean
from pyspark.sql.functions import abs as vabs
import numpy as np


def compute_mean(df, cols):
    """
    Computes a single mean value from one or more columns of numbers.

    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame.
    cols: A list of strings.
          A list of column names to compute the mean.

    Returns
    -------
    The mean (float).
    """

    ops = []
    for cname in cols:
        ops += vmean(col(cname)),

    # Mean of each column
    df2 = df.select(*ops)

    # Mean across columns
    mean = np.sum(df2.toPandas().values) / len(cols)

    return mean



def compute_mad(df, cols, mean_val=None):
    """
    Computes the MAD for mean-MAD scaler for one or more columns of numbers.

    See https://www.researchgate.net/profile/Adel_Eesa/publication/322146029_A_Normalization_Methods_for_Backpropagation_A_Comparative_Study/links/5a4a96a10f7e9ba868afeec4/A-Normalization-Methods-for-Backpropagation-A-Comparative-Study.pdf

    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame.
    cols: A list of strings.
          A list of column names to compute the MAD.
    mean_val: Float (optional).
              The mean. If not provided, mean is computed from the list of columns.

    Returns
    -------
    The MAD (float).
    """

    # Compute mean if not provided
    if mean_val is None:
        mean_val = compute_mean(df, cols)

    # Add mean as a column
    df2 = df.withColumn('mean', lit(mean_val))

    ops = []
    for cname in cols:
        ops += vabs(df2[cname] - df2.mean),

    # absolute difference for each column
    df3 = df2.select(*ops)

    # sum of difference for each row
    df4 = df3.withColumn('sum_abs_diff', sum(df3[cname] for cname in df3.columns))

    # total absolute difference
    df5 = df4.select('sum_abs_diff').groupBy().sum()

    # MAD
    n = df.count() * len(cols)
    mad = df5.toPandas().values[0,0] / n

    return mad
