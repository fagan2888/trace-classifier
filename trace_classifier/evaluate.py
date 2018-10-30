from itertools import product
import matplotlib.pyplot as plt
import numpy as np


def confusion_matrix(df, actual_col, pred_col, classes):
    """
    Computes the confusion matrix.

    Parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame.
    actual_col: String.
                Name of the column that contains the true integer label.
    pred_col: String.
              Name of the column that contains the predicted integer label.
    classes: List of strings.
             String class names that corresponds to the integer labels. Position
             of the string in the list is its integer label.

    Results
    -------
    A numpy ndarray, where the rows are the true labels and columns are the predicted labels.
    """

    n_classes = len(classes)
    class_dict = {klass: i for i, klass in enumerate(classes)}

    cm = np.zeros((n_classes, n_classes), dtype=int)

    pdf = df.crosstab(actual_col, pred_col).toPandas()
    pdf = pdf.set_index('{}_{}'.format(actual_col, pred_col))

    # Not all df has all labels in actual_col and pred_col
    for k1 in pdf.index:
        for k2 in pdf.columns:
            cm[class_dict[k1], class_dict[k2]] += pdf.loc[k1, k2]

    return cm


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix, modified from scikit docs.

    Parameters
    ----------
    cm: A numpy ndarray, or list of lists.
        A confusion matrix where the rows are the true labels and the cols are the predicted labels
    classes: List of strings.
             String class names that corresponds to the integer labels. Position
             of the string in the list is its integer label.
    normalize: One of {False, 'row', 'col'}.
               How to normalize the confusion matrix.
               - False = do not normalize.
               - row = values on a row add up to 1.
               - col = values on a column add up to 1.
    title: String.
           Title for this plot.
    cmap: matplotlib color map.
          Color scale for this plot.

    Returns
    -------
    None (displays a plot)
    """

    np.set_printoptions(suppress=True)  # Suppress scientific notation

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        eps = 1e-5   # Tiny epsilon to prevent division by 0
        axis = 0 if normalize == 'col' else 1
        cm = cm.astype('float') / np.expand_dims(cm.sum(axis=axis) + eps, axis=axis)
        cm = np.round(cm, 3)

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    np.set_printoptions(suppress=False)  # Allow scientific notation
