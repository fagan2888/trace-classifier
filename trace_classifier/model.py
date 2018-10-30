from .custom_objects import build_metrics
from .custom_objects import build_optimizer
from .custom_objects import build_loss


def construct_model(architecture, loss, metrics, optimizer, n_classes=3, input_shape=(15, 2), print_summary=True):
    """
    Constructs and compiles a Keras model.

    Parameters
    ----------
    architecture: Function.
                  Expects two arguments:
                  - n_classes (integer) for number of classes,
                  - input_shape (tuple of integers) for the model input shape
                  and a keras model object as return.
    loss: String or function.
          Loss function. String name of a keras built-in loss function, or
          a custom loss function.
    metrics: List of strings or custom stateful metric objects.
             Metrics to evaluate the model performance during training. A metric
             is either the string name of a keras built-in metric or a custom
             stateful metric object.
    optimizer: String.
               String name of a keras built-in optimizer. An optimizer is a method
               of computing gradient descent. 
    n_classes: Integer.
               Number of classes.
    input_shape: Tuple of integers.
                 Shape of the input data (excluding batch size).
    print_summary: Boolean.
                   Whether to print model summary.


    Retuns
    ------
    A compiled Keras model.
    """

    # Construct model
    model = architecture(n_classes=n_classes, input_shape=input_shape)

    # Print model summary
    if print_summary:
        model.summary()

    # Get metrics, optimizer and loss function
    metrics   = build_metrics(metrics, {'n_classes': n_classes})
    optimizer = build_optimizer(optimizer, {})
    loss      = build_loss(loss, {})

    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model