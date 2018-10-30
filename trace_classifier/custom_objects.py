from keras.optimizers import Adagrad, Adadelta, Adamax, Nadam
from .metric import F1, J


def get_custom_metrics():
    """
    Returns a dictionary of custom metric layers/functions.

    Returns
    --------
    A dictionary.
    """

    return {
        'global_weighted_f1': F1,
        'global_macro_j':     J
    }


def get_optimizers():
    """
    Returns a dictionary of optimizer layers.

    Returns
    -------
    A dictionary.
    """

    return {
        'Adagrad':  Adagrad,
        'Adadelta': Adadelta,
        'Adamax':   Adamax,
        'Nadam':    Nadam
    }


def get_custom_losses():
    """
    Returns a dictionary of custom loss functions.

    Returns
    -------
    A dictionary.
    """
    return {}



def build_metrics(metrics_list, kwargs):
    """
    Builds tf graph objects from a list of metric names.

    Parameters
    ----------
    metrics_list: A string, or a list of strings.
                  Each string is the name of a metric.
    kwargs: Dictionary.
            Parameters to pass into the constructor of these metrics.

    Returns
    -------
    A list of metric objects.
    """

    # If metrics is not a list, put it into a list
    if not isinstance(metrics_list, list) and not isinstance(metrics_list, tuple):
        metrics_list = [metrics_list]

    # A dictionary of custom metrics objects
    custom_objects = get_custom_metrics()

    # If a metric exists in Keras already, do nothing.
    # If a metric is custom, build a new stateful metrics layer.
    metrics = []
    for m in metrics_list:
        if m in custom_objects:
            metrics += custom_objects[m](**kwargs),
        else:
            metrics += m,

    return metrics


def build_optimizer(optimizer_name, kwargs):
    """
    Builds a tf graph object from an optimizer name.

    Parameter
    ---------
    optimizer_name: String.
                    Name of an optimizer.
    kwargs: Dictionary.
            Parameters to pass into the constructor of this optimizer.

    Return
    ------
    An optimizer.
    """

    # A dictionary of optimizers
    optimizers = get_optimizers()

    return optimizers[optimizer_name](**kwargs)


def build_loss(loss_name, kwargs):
    """
    Builds a tf graph object from a loss name.

    Parameter
    ---------
    loss_name: String.
               Name of a loss function.
    kwargs: Dictionary.
            Parameters to pass into the constructor of this loss function.

    Return
    ------
    A loss function.
    """
    losses = get_custom_losses()

    if loss_name in losses:
        return losses[loss_name](**kwargs)
    else:
        return loss_name