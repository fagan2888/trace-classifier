import os

import tensorflow as tf
import tensorframes as tfs
from pyspark.sql.functions import array
from pyspark.sql.functions import col
from pyspark.sql.functions import sum as vsum

from .config import MODEL_INPUT_CONFIG
from .load import load_model_metadata
from .phrase import create_phrases
from .preprocessing import include_id_and_label
from .preprocessing import include_word_vecs
from .utils import argmax
from .utils import reverse_create_label


def infer(df, model_file=None, aggregate=True):
    """
    Predict.

    Expects the dataframe to contains a column called `coordinates` of the
    data type array<array<double>>.

    Parameters
    ----------
    model_file: String.
                Path to a .pb tensorflow model file.
    df: A pyspark.sql.dataframe.DataFrame.
        Expects a column called `coordinates` of array<array<double>> type.
    model_file: String.
                Full path to a model .pb file.
                Expects a correspoinding metadata json file in the same directory.
                If not provided, loads the sample model.
    aggregate: Boolean.
               Whether to aggregate piece-wise results into a prediction for the full trace.

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame with two extra columns
    `probas` (array<double>) for the probabilities of each class, and
    `pred_modality` (string) for the class that has the highest probability.
    """

    # Use sample model if a model is not provided.
    if model_file is None:
        dir, _ = os.path.split(__file__)
        model_file = os.path.join(dir, "sample_model/sample_model_optimised_frozen.pb")

    # Load model metadata
    metadata = load_model_metadata(model_file)
    assert metadata is not None

    # Preprocess data
    with_ids_and_labels_df = include_id_and_label(df)  # To be joined with prediction
    with_ids_and_labels_df.persist()

    with_word_vecs_df, _, _ = include_word_vecs(with_ids_and_labels_df, metadata)
    with_phrases_df = create_phrases(
        with_word_vecs_df,
        MODEL_INPUT_CONFIG["WORD_VEC_COL"],
        MODEL_INPUT_CONFIG["ID_COL"],
        MODEL_INPUT_CONFIG["WORD_POS_COL"],
        desired_phrase_length=metadata["desired_phrase_length"],
    )
    with_phrases_df.persist()

    # Read in serialized tensorflow graph
    with tf.gfile.FastGFile(model_file, "rb") as f:
        model_graph = f.read()

    with tf.Graph().as_default() as g:
        # Reconstruct tf graph (parse serialised graph)
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_graph)

        input_op_name = [
            n.name
            for n in graph_def.node
            if n.op.startswith("Placeholder") and n.name.startswith("input")
        ][0]
        output_op_name = [
            n.name
            for n in graph_def.node
            if n.op.startswith("Softmax") and n.name.startswith("output")
        ][0]

        # Add metadata on the input size to the dataframe for tensorframes
        input_shape = [None, *metadata["input_shape"]]
        model_input_df = tfs.append_shape(
            with_phrases_df,
            with_phrases_df[MODEL_INPUT_CONFIG["INPUT_COL"]],
            shape=input_shape,
        )

        # Load graph
        [input_op, output_op] = tf.import_graph_def(
            graph_def, return_elements=[input_op_name, output_op_name]
        )

        # Predict
        model_output_df = tfs.map_blocks(
            output_op.outputs,
            model_input_df,
            feed_dict={input_op.name: MODEL_INPUT_CONFIG["INPUT_COL"]},
        )

        # Rename column
        output_col = list(set(model_output_df.columns) - set(with_phrases_df.columns))[
            0
        ]  # Something like 'import/output/Softmax', but might change
        phrasewise_res_df = model_output_df.withColumnRenamed(
            output_col, "probas"
        ).withColumn("pred_label", argmax(col("probas")))

        if aggregate:
            phrasewise_res_df.persist()

            # Average piece-wise probabilities into full-trace probabilities, and
            # find the label with the highest probability.
            with_avg_prob_df = avg_probability(
                phrasewise_res_df, "id", "probas", len(metadata["classes"])
            )

            # Convert integer labels into string classes
            with_predicted_labels_df = reverse_create_label(
                with_avg_prob_df,
                "sentence_pred_label",
                "pred_modality",
                metadata["classes"],
            ).withColumnRenamed("sentence_probas", "probas")

            # Join prediction with the original dataframe to get the coordinates
            res_df = with_ids_and_labels_df.join(
                with_predicted_labels_df, on=MODEL_INPUT_CONFIG["ID_COL"], how="inner"
            )

        else:
            # TO-DO: return pieces of coordinates rather than phrases
            res_df = reverse_create_label(
                phrasewise_res_df, "pred_label", "pred_modality", metadata["classes"]
            )

        # clean up
        with_ids_and_labels_df.unpersist()
        with_phrases_df.unpersist()
        phrasewise_res_df.unpersist()

        res_df.persist()
        return res_df


def avg_probability(df, sentence_col, probas_col, n_classes):
    """
    Aggregate piece-wise predictions into a full-trace prediction by
    uniform-weight arithmetic average of the predicted probabilities.

    Parameters
    ----------
    df: A pyspark.sql.dataframe.DataFrame.
    sentence_col: String.
                  Name of the column that contains the unique ID of a trace.
    probas_col: String.
                Name of the column that contains the piece-wise predicted probabilities.
    n_classes: Integer.
               Number of classes.

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame with three columns:
     sentence_col, which contains the unique ID of a trace,
    `sentence_probas` (array<float>) which contains the average probabilities for the trace,
    `sentence_pred_label` (integer) which contains the integer label with the highest probability.
    """

    # Dummy class names for naming columns
    classes = list(map(str, range(n_classes)))

    # Prepare agg operations
    ops = []
    for i, klass in enumerate(classes):
        ops += (vsum(df[probas_col][i]).alias(klass),)

    # Add up probabilities
    with_probs_means = (
        df.groupBy(sentence_col)
        .agg(*ops)
        .withColumn("total_probas", sum(col(klass) for klass in classes))
    )

    # Divide by total to get the average
    for klass_col in classes:
        with_probs_means = with_probs_means.withColumn(
            klass_col, col(klass_col) / col("total_probas")
        )

    # Gather probabilities into an array, return argmax
    return with_probs_means.select(
        sentence_col,
        array(*(with_probs_means[klass] for klass in classes)).alias("sentence_probas"),
    ).withColumn("sentence_pred_label", argmax(col("sentence_probas")))
