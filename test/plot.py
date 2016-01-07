import matplotlib
import seaborn
import matplotlib.backends.backend_agg


def fig(*args, **kwargs):
    fig = matplotlib.figure.Figure(*args, **kwargs)
    return fig


def axes(*args, **kwargs):
    fig = matplotlib.figure.Figure(*args, **kwargs)
    ax = fig.add_subplot(111)
    return ax


def canvas(*args, **kwargs):
    return matplotlib.backends.backend_agg.FigureCanvasAgg(*args, **kwargs)


def save(figure, filename, format=None, **kwargs):
    if isinstance(figure, matplotlib.axes.Axes):
        figure = figure.figure
    canvas(figure).print_figure(filename, format=format, **kwargs)
