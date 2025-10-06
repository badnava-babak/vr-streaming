import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from scripts.print_distance_table import prepare_data


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


if __name__ == '__main__':
    bins = 3

    # opt, _ = prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/Optimal.pkl')
    data = {
        'Optimal': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/Optimal.pkl', False)[0],
        'CPPG': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/CPPG.pkl')[0],
        'PPG': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/PPG.pkl', False)[0],
        'EGreedy': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/EGreedy.pkl')[0],
        'PPO': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/PPO.pkl')[0],
    }

    st = {}
    for label, df in data.items():
        stats = df.groupby(by=['quality_decision',
                               'offloading_decision']
                           ).agg(count=('latency', 'count'))
        st[label] = {}
        for i in range(1, 7):
            st[label][i] = {}
            for j in range(4):
                st[label][i][j] = 0.
                if (i, j) in stats.index.to_list():
                    # st += f"{i} : {stats['latency'][i] / stats['latency'].sum():.2f}, "
                    st[label][i][j] = stats['count'].loc[(i, j)] / stats['count'].sum()
                else:
                    st[label][i][j] = 0.

    for label, s in st.items():
        df = pd.DataFrame(s)

        fig, ax = plt.subplots()
        im, cbar = heatmap(df, range(4), range(6, 0, -1), ax=ax, cmap="YlGn", cbarlabel="Action frequency")
        texts = annotate_heatmap(im, valfmt="{x:.2f}")
        plt.title(label)
    plt.show()
