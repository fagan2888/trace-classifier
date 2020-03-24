def is_equal_df(expected_df, actual_df, sort_column="test_id"):
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
        .orderBy(sort_column).collect()
    actual = actual_df.select(*sorted(actual_df.columns)) \
        .orderBy(sort_column).collect()
    return expected == actual
