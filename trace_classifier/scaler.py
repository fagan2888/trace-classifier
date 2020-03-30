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

    ops = [ vmean(col(cname)) for cname in cols ]
    return np.sum(df.select(*ops).toPandas().values) / len(cols)


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

    ops = [vabs(col(cname) - col("mean")) for cname in cols]

    # Add mean as a column
    with_ops_df = df.withColumn('mean', lit(mean_val)) \
        .select(*ops)
    with_sum_abs_diff_df = with_ops_df.withColumn('sum_abs_diff', sum(col(cname) for cname in with_ops_df.columns)) \
        .select('sum_abs_diff').groupBy().sum()

    n = df.count() * len(cols)
    mad = with_sum_abs_diff_df.toPandas().values[0,0] / n

    return mad
