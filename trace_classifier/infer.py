from pyspark.sql.functions import sum as vsum
from pyspark.sql.functions import array
from .load import load_model_metadata
from .preprocessing import preprocessing_part0
from .preprocessing import preprocessing_part1
from .preprocessing import preprocessing_part2
from .preprocessing import preprocessing_part3
from .utils import reverse_create_label
from .utils import argmax
import tensorframes as tfs
import tensorflow as tf
from pkgutil import get_data
import json


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
    sample_model_metadata = '{"word_size": [2, 1, 1], "desired_ops": [[["d", 0, 1]], [["s", 0, 1]]], "normalize": "mean-mad", "clip_rng": [-1, 1], "ndigits": 2, "desired_phrase_length": 15, "ndims": 1, "classes": ["Not Driving", "Driving", "Noise"], "input_shape": [15, 2], "norm_params": {"offset": [12.793474655679825, 0.006270694753199434], "scale": [11.100786721797306, 0.005729512902929993]}, "save_weights_only": true}'

    if model_file is None:
        metadata = sample_model_metadata
    else:
        parts = model_file.split("_")
        if parts[0] == 'sample' and parts[1] == "model":
            metadata = json.loads(sample_model_metadata)
            model_file = 'sample_model/sample_model_optimized_frozen.pb'
        else:
            # this below doesnt work but we shouldn't get here right now
            print("in the broken load model")
            metadata = json.loads(load_model_metadata(model_file))

    print("model_file is {}".format(model_file))
    print("metadata is {}".format(metadata))
    # Load model metadata
    assert metadata is not None

    # Preprocess data
    df2 = preprocessing_part0(df)  # To be joined with prediction

    df2.persist()

    df3 = preprocessing_part1(df2, metadata)
    df4, _, _ = preprocessing_part2(df3, metadata)
    df5 = preprocessing_part3(df4, metadata)
    df5.persist()
    print('df5 debug {}'.format(df5.take(1)))

    # Name of the column that contains the input
    input_col = 'phrase'

    print('read serizalized tensorflow graph')

    # Read in serialized tensorflow graph
    with tf.gfile.FastGFile(model_file, 'rb') as f:
        model_graph = f.read()

    with tf.Graph().as_default() as g:
        # Reconstruct tf graph (parse serialised graph)
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_graph)

        input_op_name = [n.name for n in graph_def.node if
                         n.op.startswith('Placeholder') and n.name.startswith('input')][0]
        output_op_name = [n.name for n in graph_def.node if
                          n.op.startswith('Softmax') and n.name.startswith('output')][0]

        # Add metadata on the input size to the dataframe for tensorframes
        input_shape = [None, *metadata['input_shape']]
        df6 = tfs.append_shape(df5, df5[input_col], shape=input_shape)

        # Load graph
        [input_op, output_op] = tf.import_graph_def(graph_def,
                                                    return_elements=[input_op_name, output_op_name])

        # Predict
        df7 = tfs.map_blocks(output_op.outputs, df6, feed_dict={input_op.name: input_col})

        # Rename column
        output_col = list(set(df7.columns) - set(df5.columns))[
            0]  # Something like 'import/output/Softmax', but might change
        df8 = df7.withColumnRenamed(output_col, 'probas')

        # Find the label with the highest probability
        rdf = df8.withColumn('pred_label', argmax(df8.probas))

        if aggregate:
            rdf.persist()

            # Average piece-wise probabilities into full-trace probabilities, and
            # find the label with the highest probability.
            rdf2 = avg_probability(rdf, 'id', 'probas', len(metadata['classes']))

            # Convert integer labels into string classes
            rdf3 = reverse_create_label(rdf2, 'sentence_pred_label', 'pred_modality',
                                        metadata['classes'])

            # Join prediction with the original dataframe to get the coordinates
            rdf4 = rdf3.withColumnRenamed('sentence_probas', 'probas')
            rdf5 = df2.join(rdf4, on='id', how='inner')

        else:
            # TO-DO: return pieces of coordinates rather than phrases
            rdf5 = reverse_create_label(rdf, 'pred_label', 'pred_modality', metadata['classes'])

        rdf5.persist()

        # clean up
        df2.unpersist()
        df5.unpersist()
        rdf.unpersist()

        return rdf5


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
        ops += vsum(df[probas_col][i]).alias(klass),

    # Add up probabilities
    df2 = df.groupBy(sentence_col).agg(*ops)

    # Divide by total to get the average
    df3 = df2.withColumn('total_probas', sum(df2[klass] for klass in classes))
    for klass_col in classes:
        df3 = df3.withColumn(klass_col, df3[klass_col] / df3.total_probas)

    # Gather probabilities into an array
    df4 = df3.select(sentence_col,
                     array(*(df3[klass] for klass in classes)).alias('sentence_probas'))

    # Find the label with the highest probability
    df5 = df4.withColumn('sentence_pred_label', argmax(df4.sentence_probas))

    return df5
