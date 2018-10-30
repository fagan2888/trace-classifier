from pyspark.sql import SparkSession


def read_dataframe(path, format='json'):
    """
    Reads a pyspark dataframe from disk.

    The dataframe may be stored as one or many files under a directory.

    Parameters
    ----------
    path: String.
          Directory of files to read in.
    format: String, one of {'json', 'parquet'}.
            File format.

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame.
    """

    assert format in ['json', 'parquet']

    # Get spark session
    spark = SparkSession.builder.getOrCreate()

    if format == 'json':
        df = spark.read.json('{}/*'.format(path))
    elif format == 'parquet':
        df = spark.read.parquet('{}/*'.format(path))

    return df
