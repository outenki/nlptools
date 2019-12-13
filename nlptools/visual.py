import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, List


def draw_curves(
        output_loc: str,
        data_list: Iterable[np.ndarray],
        legend_list: List[str],
        title: str, xlabel: str, ylabel: str):
    ''' Draw curves
    :param output_loc: str, path to the file to output
    :param legend_list: List[str], names of each curve
    :param data_list: List[np.ndarray], datas to draw. All of them
        are expected to have same length.
    :param title: str, title of the figure
    :param xlabel: str, label of x axel
    :param ylabel: str, label of y axel
    '''
    for data in data_list:
        plt.plot(data)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend_list, loc='upper left')
    plt.savefig(output_loc)
    plt.clf()
