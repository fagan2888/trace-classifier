import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)sZ [%(levelname)s][%(name)s] %(message)s')
from os import path
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import numpy.testing as npt

FIXTURES_PATH = path.join(path.dirname(path.realpath(__file__)), "fixtures")
MODEL_PATH = path.join(FIXTURES_PATH, "sample_model_optimised_frozen.pb")

COORDINATES_TYPE = T.ArrayType(T.DoubleType())
LINESTRING_TYPE = T.ArrayType(COORDINATES_TYPE)
TRACES_SCHEMA = T.StructType([
    T.StructField("test_id", T.StringType()),
    T.StructField("coordinates", LINESTRING_TYPE)
])

def is_equal_df(expected_df, actual_df, sort_columns=["test_id"], approx=False):
    """
    Test equality of two Spark DataFrames according to their collected contents
    Args:
        expected_df (pyspark.sql.DataFrame): first argument of equality relation
        actual_df (pyspark.sql.DataFrame): second argument of equality relation
        sort_column (String): the column by which to sort the two argument
    Returns
        bool
    """
    expected = expected_df.select(*sorted(expected_df.columns)) \
        .orderBy(*sort_columns).collect()
    actual = actual_df.select(*sorted(actual_df.columns)) \
        .orderBy(*sort_columns).collect()
    # logging.info([ e == a for (e, a) in zip(expected, actual) ])
    # e = expected[0]
    # a = actual[0]
    # logging.info(e)
    # logging.info(a)
    if approx:
        expected = [ e for r in expected for e in r.asDict().items() ]
        actual = [ e for r in actual for e in r.asDict().items() ]
        # comparisons = [ npt.assert_allclose(e, a) for (e, a) in zip(expected, actual) ]
        logging.info(expected)
        logging.info(actual)
        logging.info(comparisons)
        # return npt.assert_allclose([ e.asDict() actual], expected)
    return expected == actual
