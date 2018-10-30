from tensorflow.python.tools import freeze_graph
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework import tensor_util
import keras.backend as K
import tensorflow as tf
import h5py
import json
import shutil
import os


def save_model(model, model_dir, model_name, metadata=None, verbose=True):
    """
    Saves the model in 2 formats
    - A complied model (<model_name>.h5)
    - A serialized model (<model_name>_frozen_optimised.pb)

    A compiled model is a .h5 file that contains model weights, model architecture
    and the optimizer state. It's mainly used for continue training. When used for
    inference, no other files are required as the .h5 file already contains the
    weights, architecture, and parameters for preprocessing a trace into model input
    (which is stored in the metadata).

    A serialised model is a .pb file that only contains the model weight and
    a frozen achitecture optimised for fast inference. Since it does not contain
    the parameters for preprocessing a trace into model input, it must be used
    along with a metadata json file `<model_name>_metadata.json`.

    Parameters
    ----------
    model: A Keras model.
    model_dir: String.
               The directory to save this model.
    model_name: String.
                Model name which is also used as the basename for files.
    metadata: Dictionary (optional).
              Other metadata to save with the model.
    verbose: Boolean.
             Whether to print file locations.

    Returns
    -------
    None
    """

    # Create directory if not already exist
    os.makedirs(model_dir, exist_ok=True)

    byproducts_dir = os.path.join(model_dir, 'byproducts')
    os.makedirs(byproducts_dir, exist_ok=True)

    ######
    # Save a complied model for continue training
    ######

    compiled_model_path = os.path.join(model_dir, model_name + '.h5')
    model.save(compiled_model_path)

    if verbose:
        print('Saved compiled model to ' + compiled_model_path)


    ######
    # Save metadata
    ######

    if metadata:
        # Append metadata to compiled model
        with h5py.File(compiled_model_path, mode='a') as fp:
            fp.attrs['metadata'] = json.dumps(metadata)
            if verbose:
                print('Appended metadata to ' + compiled_model_path)

        # Save metadata to a json file for serialised model
        metadata_file = os.path.join(model_dir, model_name + '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
            if verbose:
                print('Saved metadata to ' + metadata_file)


    ######
    # Save a serialized model for inferencing in Spark
    ######

    # Saves a checkpoint for freezing graph
    checkpt_path = os.path.join(byproducts_dir, model_name + '.ckpt')
    saver = tf.train.Saver()
    saver.save(K.get_session(), checkpt_path)
    if verbose:
        print('Saved checkpoint to ' + checkpt_path)

    # Saves a copy of the graph
    graph_path = os.path.join(byproducts_dir, model_name + '.pbtxt')
    tf.train.write_graph(K.get_session().graph, './', graph_path)
    if verbose:
        print('Saved graph to ' + graph_path)

    # Get node names
    input_node_names  = model.input.op.name
    output_node_names = model.output.op.name
    restore_op_name   = 'save/restore_all'

    # Freeze and save a frozen graph
    # (Modified from https://github.com/tensorflow/tensorflow/issues/8181#issuecomment-309375713)
    input_saver_def_path        = ""
    input_binary                = False
    filename_tensor_name        = 'whatever'    # Unused by updated loading code – see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py#L76
    output_frozen_graph_name    = os.path.join(byproducts_dir, model_name + '_frozen.pb')
    clear_devices               = True

    freeze_graph.freeze_graph(input_graph=graph_path,
                              input_saver=input_saver_def_path,
                              input_binary=input_binary,
                              input_checkpoint=checkpt_path,
                              output_node_names=output_node_names,
                              restore_op_name=restore_op_name,
                              filename_tensor_name=filename_tensor_name,
                              output_graph=output_frozen_graph_name,
                              clear_devices=clear_devices,
                              initializer_nodes="")

    if verbose:
        print('Saved frozen graph to ' + output_frozen_graph_name)

    # Optimize the graph for inference
    # (See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/python/transform_graph_test.py)
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
        # load graph
        data = f.read()
        input_graph_def.ParseFromString(data)

        # list of transforms
        # (See https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#transform-reference)
        transforms = [
            "fold_batch_norms",
            "strip_unused_nodes",
            "remove_device",
            "remove_nodes(op=Identity, op=CheckNumerics)",
            "add_default_attributes"
        ]

        output_graph_def = TransformGraph(input_graph_def,
                                          input_node_names.split(","),  # an array of the input node(s)
                                          output_node_names.split(","), # an array of the output nodes
                                          transforms)

    # Save the optimized graph
    output_optimized_graph_name = os.path.join(model_dir, model_name + '_optimised_frozen.pb')
    with tf.gfile.FastGFile(output_optimized_graph_name, "w") as f:
        f.write(output_graph_def.SerializeToString())
    if verbose:
        print('Saved optimized frozen graph to ' + output_optimized_graph_name)

    # Clean up
    shutil.rmtree(byproducts_dir, ignore_errors=True)
    if verbose:
        print('Removed byproducts in ' + byproducts_dir)


def print_weights(model_file):
    """
    Prints the weights of a frozen model.

    Parameters
    ----------
    model_file: String.
                Path to a frozen .pb model.

    Returns
    -------
    None
    """

    # Read in serialised model
    with tf.gfile.FastGFile(model_file, 'rb') as f:
        model_graph = f.read()

    with tf.Graph().as_default() as g:
        # Reconstruct tf graph (parse serialised graph)
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_graph)

        # Load graph
        tf.import_graph_def(graph_def, name='')

        # Loop through all nodes in the graph
        graph_nodes=[n for n in graph_def.node]
        wts = [n for n in graph_nodes if n.op=='Const']
        for n in wts:
            print('Node Name: ' + n.name)
            print('Value:')
            print(tensor_util.MakeNdarray(n.attr['value'].tensor))
            print('\n')

