import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

def generate_confusion_matrix_plot(y_true, y_pred, uid):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    path = f'static/plots/{uid}_cm.png'
    plt.savefig(path)
    plt.close()
    return path
