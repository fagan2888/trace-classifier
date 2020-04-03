import numpy as np
from pyspark.sql.functions import col
from pyspark.sql.functions import collect_list
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import size

from .utils import pad


def create_phrases(
    df,
    word_vec_col,
    sentence_id_col,
    word_pos_col,
    phrase_pos_col="phrase_pos",
    phrase_col="phrase",
    word_count_col="word_count",
    desired_phrase_length=15,
):
    """
    Gathers words (more precisely, the word vecs) into a zero-padded phrase.


    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame
    word_vec_col: String.
                  Name of the column that contains the word vecs.
    sentence_id_col: String.
                     Name of the column that contains the unique ID of a trace.
    word_pos_col: String.
                  Name of the column that contains a word's position (starting
                  with 0) in a sentence
    phrase_pos_col: String.
                    Name of a new column to be created that will contain a phrase's
                    position (starting with 0) in a sentence.
    phrase_col: String.
                Name of a new column to be created that will contain a zero-padded
                phrase (i.e. an array of word vecs).
    word_count_col: String.
                    Name of a new column to be created that will contain the
                    number of words in an unpadded phrase.
    desired_phrase_length: Integer.
                           Standard length of a phrase.


    Returns
    -------
    A pyspark.sql.dataframe.DataFrame with zero-padded phrases, their corresponding
    position in a sentence and the word count (ignoring padding) in each phrase.
    """

    # Helper function
    @pandas_udf("integer")
    def phrase_id(pos):
        """
        Computes the phrase position from word position.

        Parameters
        ----------
        pos: a pyspark.sql.column.Column, word position in integer.

        Returns
        -------
        A a pyspark.sql.column.Column of phrase position.
        """
        return np.floor(pos / desired_phrase_length)

    othercols = list(set(df.columns) - {word_vec_col, word_pos_col})

    # Assign each word to a phrase
    with_phrase_pos_df = df.withColumn(phrase_pos_col, phrase_id(df[word_pos_col]))

    # Collect the word vecs from the same phrase into an array
    with_phrases_df = (
        with_phrase_pos_df.sortWithinPartitions(sentence_id_col, word_pos_col)
        .groupBy([*othercols, phrase_pos_col])
        .agg(collect_list(word_vec_col).alias(phrase_col))
    )

    # Pad phrase to desired length
    return with_phrases_df.withColumn(
        "word_count", size(with_phrases_df[phrase_col])
    ).withColumn(
        phrase_col, pad(col(phrase_col), desired_phrase_length - col("word_count"))
    )
