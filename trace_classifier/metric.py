from keras.engine.topology import Layer
import keras.backend as K
import numpy as np


class StatefulMetrics(Layer):
    '''
    Computes global stateful metrics for sparse label.

    "Global metric" is to be differentiate with "batch metric": tensorflow
    processes data batch by batch. Batch metric is the performance for the
    current batch only; global metric is the performance of all the batches
    processed so far.

    Parameters
    ----------
    name: String.
          A name for the metric.
    n_classes: Integer.
               Number of classes.
    metric_fn: Function.
               How to compute the metric. Expects 3 arguments:
               - a global confusion matrix (accumulation of results so far),
               - confusion matrix from the current batch (results from the current batch), and
               - n_classes.

    '''

    def __init__(self, name, n_classes, metric_fn, **kwargs):
        super().__init__(name=name, **kwargs)

        self.stateful = True
        self.n_classes = n_classes

        # Placeholder for storing the global confusion matrix (CM)
        self.cm = []
        for i in range(n_classes):
            row = []
            for j in range(n_classes):
                row += K.variable(value=0, dtype='float32', name='cm_{}_{}'.format(i, j)),
            self.cm += row,

        self.metric_fn = metric_fn

    def reset_states(self):
        """
        Clears the global confusion matrix (CM).
        """

        for i in range(self.n_classes):
            for j in range(self.n_classes):
                K.set_value(self.cm[i][j], 0.0)

    def get_config(self):
        config = {'n_classes': self.n_classes}

        base_config = super(StatefulMetrics, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def __call__(self, y_true, y_pred):
        '''
        Update the stateful metric with results for the current batch.

        Expects y_true and y_pred to be in sparse-matrix represenation. See
        https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/.

        Parameters
        ----------
        y_true: Tensor.
                True labels for a batch, stored in sparse format.
        y_pred: Tensor.
                Model output for the current batch of data.

        Returns
        -------
        Global performance so far.
        '''

        # Convert probabilities into integer labels
        # (From https://github.com/keras-team/keras/blob/master/keras/metrics.py#L36-L39)
        pred_labels = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
        true_labels = K.cast(K.max(y_true, axis=-1), K.floatx())

        # Populate batch CM and a list of operations
        # (The list of of operations is just a plan for how to update global CM
        # with batch CM; operations are not executed until later).
        batch_cm = []  # Confusion matrix for the current batch
        updates = []   # A list of operations
        for true in range(self.n_classes):
            row = []
            for pred in range(self.n_classes):
                a = K.cast(K.equal(true_labels, true),  K.floatx())
                b = K.cast(K.equal(pred_labels, pred),  K.floatx())
                c = K.cast(K.sum(a * b, keepdims=False),  K.floatx())
                updates += K.update_add(self.cm[true][pred], c),
                row += c,
            batch_cm += row,

        # Make a copy of the current CM before updating.
        current_cm = []
        for i in range(self.n_classes):
            row = []
            for j in range(self.n_classes):
                row += self.cm[i][j] * 1,
            current_cm += row,

        # Comput metric
        metric = self.metric_fn(current_cm, batch_cm, self.n_classes)

        # Execute update plan
        self.add_update(updates, inputs=[y_true, y_pred])

        return metric


class F1(StatefulMetrics):
    def __init__(self, name='global_weighted_f1', n_classes=3, **kwargs):
        super().__init__(name, n_classes, compute_F1, **kwargs)


class J(StatefulMetrics):
    def __init__(self, name='global_macro_J', n_classes=3, **kwargs):
        super().__init__(name, n_classes, compute_J, **kwargs)


def compute_F1(current_cm, batch_cm, n_classes):
    """
    Compute weighted F1 over all batches.

    (For definition of micro vs. macro vs. weighted, see
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)

    Parameters
    ----------
    current_cm: A list of lists of tf.Variable.
                Global confusion matrix (CM) not yet updated with the CM from
                the latest batch.
    batch_cm: A list of lists of tf.Variable.
              CM from the latest batch.
    n_classes: Integer.
               Number of classes.

    Returns
    -------
    Global weighted-averaged F1.
    """

    # Tally
    total = 0
    f1s   = []
    w     = [] # weights
    for kls in range(n_classes):
        row_sum = 0
        col_sum = 0
        for j in range(n_classes):
            row_sum += current_cm[kls][j] + batch_cm[kls][j]
            col_sum += current_cm[j][kls] + batch_cm[j][kls]

        TP = current_cm[kls][kls] + batch_cm[kls][kls]

        recall = TP / (row_sum + K.epsilon())
        precision = TP / (col_sum + K.epsilon())
        f1 = 2 * precision * recall / (precision + recall)

        f1s   += f1,
        w     += row_sum,
        total += row_sum


    # Weighted F1
    f1 = np.sum(np.array(f1s) * np.array(w)) / total

    return K.cast(f1, K.floatx())



def compute_J(current_cm, batch_cm, n_classes):
    """
    Compute macro Youden's J index over all batches.

    (See https://www.aaai.org/Papers/Workshops/2006/WS-06-06/WS06-06-006.pdf.)


    Parameters
    ----------
    current_cm: A list of lists of tf.Variable.
                Global confusion matrix (CM) not yet updated with the CM from
                the latest batch.
    batch_cm: A list of lists of tf.Variable.
              CM from the latest batch.
    n_classes: Integer.
               Number of classes.

    Returns
    -------
    Global macro-averaged J.
    """

    # Tally
    total = 0
    diag_sum = 0
    row_sums = []
    col_sums = []
    for i in range(n_classes):
        row_sum = 0
        col_sum = 0
        for j in range(n_classes):
            row_sum += current_cm[i][j] + batch_cm[i][j]
            col_sum += current_cm[j][i] + batch_cm[j][i]
        row_sums += row_sum,
        col_sums += col_sum,
        total += row_sum
        diag_sum += current_cm[i][i] + batch_cm[i][i]

    # Compute J for each class
    js = []
    for kls in range(n_classes):
        TP = current_cm[kls][kls] + batch_cm[kls][kls]
        FP = diag_sum - TP
        sensitivity = TP / (row_sums[kls] + K.epsilon())
        specificity = FP / (total - row_sums[kls] + K.epsilon())
        j = sensitivity + specificity - 1
        js += j,

    # Macro-average J
    j = sum(js) / n_classes

    return K.cast(j, K.floatx())
