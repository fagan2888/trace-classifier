from pyspark.sql.functions import lit
from pyspark.sql.functions import struct
import shutil
import os


def write_dataframe(dst, df, format='json'):
    """
    Writes out dataframe to disk.

    Parameters
    ----------
    dst: String.
         Directory to save dataframe. Dataframe will be saved as one file per partition.
    df: A pyspark.sql.dataframe.DataFrame.
    format: String, one of {'json', 'parquet'}.
            Output format.

    Returns
    -------
    Path to the directory with saved files.
    """

    # Recreate directory
    shutil.rmtree(dst, ignore_errors=True)

    # Write out dataframe
    df.write.format(format).save(dst)

    return dst


def write_traces(dst, df, coordinates_col, properties_cols=None, limit=None, max_record_per_file=None):
    """
    Writes out traces in line-delimited geojson.

    Parameters
    ----------
    dst: String.
         Directory to save line-delimited geojson files.
    df: A pyspark.sql.dataframe.DataFrame.
    coordinates_col: String.
                     Name of the column that contains the coordinates.
    properties_col: List of strings (optional)
                    Name of columns to include as geojson linestring properties.
    limit: Integer (optional).
           Number of records to export.
    max_record_per_file: Integer (optional).
                         Maximum number of records per file.

    Returns
    -------
    None
    """

    # Geometry
    df2 = df.withColumn('type', lit('LineString'))
    df2 = df2.withColumn('geometry', struct(df2['type'], df2[coordinates_col]))

    # Properties
    if properties_cols:
        lst = []
        for c in properties_cols:
            lst += df2[c],
        df2 = df2.withColumn('properties', struct(*lst))

    # Feature
    df3 = df2.withColumn('type', lit('Feature'))
    df4 = df3.select('type', 'geometry', 'properties')


    # Recreate directory
    shutil.rmtree(dst, ignore_errors=True)

    # Write to disk
    if limit:
        df5 = df4.limit(limit)
    else:
        df5 = df4

    if max_record_per_file is None:
        df5.write.format('json').save(dst)
    else:
        df5.write.option('maxRecordsPerFile', max_record_per_file).format('json').save(dst)
