from .train import train_val_split
from .train import train
import tensorflow as tf
import keras.backend as K


def k_fold_CV(architecture, config, df, n_epochs=50, n_folds=None):
    """
    Run k-fold cross validation.

    Parameters
    ----------
    architecture: Function.
                  A function to construct a Keras Model object.
    config: Dictionary.
            Configuration for training a model.
    df: pyspark.sql.dataframe.DataFrame.
        The ouptput dataframe from preprocessing_part1 function (see preprocessing.py).
    n_epochs: Integer.
              Maximum number of epochs to train.
    n_folds: Integer (default = None = all folds)
             Number of folds to run.

    Returns
    -------
    A list of k tuples (<model_name>, <best_performance>).
    """

    model_performance = []

    # Number of folds to run
    k = n_folds or config['TRAIN']['k_fold_cv']

    # Number of classes
    n_classes = len(config['INPUT']['classes'])


    for vfold in range(k):

        # Clean session to speed up training
        K.clear_session()
        tf.reset_default_graph()

        # Train-val split
        trn_set, val_set, norm_params = train_val_split(df, vfold, config['PREPROCESS'])

        # Save normalization values for inference.
        config['INPUT']['norm_params'] = norm_params

        # Train model
        model_name, training_history = train(architecture,
                                             config,
                                             trn_set,
                                             val_set,
                                             n_epochs=n_epochs,
                                             print_summary=(vfold == 0),
                                             decorate='MODEL NO. {}'.format(vfold))

        model_performance += (model_name, max(training_history.history[config['TRAIN']['MODEL_CHECKPOINT']['monitor']])),

    return model_performance



def summarize_performance(model_performance):
    """
    Summarize k-fold cross validation result.

    Parameter
    ---------
    model_performance: List of tuples (<model_name>, <best_performance>).
                       The object returned by k_fold_CV function.

    Returns
    -------
    A dictionary.
    """

    best_model = ''
    best_model_vfold = -1
    best_performance = 0
    avg_performance = 0
    worst_performance = float('Inf')

    for i, (model_name, performance) in enumerate(model_performance):
        if performance > best_performance:
            best_performance = performance
            best_model       = model_name
            best_model_vfold = i
        if performance < worst_performance:
            worst_performance = performance
        avg_performance += performance

    avg_performance /= len(model_performance)

    return {
        'best_model_vfold': best_model_vfold,  # vfold = the fold that was used as validation set
        'best_model':       best_model,
        'best':             best_performance,
        'average':          avg_performance,
        'worst':            worst_performance
    }