import math

import matplotlib.pyplot as plt
import networkx
import numpy as np
import pingouin as pg


def performance_analysis(frame, field="f1", significance_level=0.05, ascending=False):
    sorted_res = frame.sort_values(["model", "dataset", "i", "trainset_fraction"])
    sorted_res["subject"] = sorted_res[["dataset", "i", "trainset_fraction"]].apply(
        lambda x: f"{x[0]}-{x[1]}-{np.round(x[2], 2)}", axis=1
    )

    f_test = pg.friedman(
        sorted_res, dv=field, within="model", subject="subject", method="chisq"
    )

    if f_test["p-unc"].values[0] > significance_level:
        print("No statistically significant difference between models")
        return

    pt_results = pg.pairwise_tests(
        sorted_res,
        dv=field,
        within="model",
        subject="subject",
        parametric=False,
        padjust="holm",
    )
    pt_results["significant"] = pt_results["p-corr"] < significance_level

    p_values = pt_results[["A", "B", "p-corr", "significant"]].values.tolist()

    ranks = sorted_res.pivot(columns="subject", index="model", values=field).rank(
        ascending=ascending
    )
    average_ranks = ranks.mean(1).sort_values(ascending=False)

    return average_ranks, p_values


def form_cliques(p_values, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)


# inspired from orange3 https://docs.orange.biolab.si/3/data-mining-library/reference/evaluation.cd.html
def graph_ranks(
    avranks,
    names,
    pretty_names,
    p_values,
    ax=None,
    lowv=None,
    highv=None,
    textspace=0.2,
    reverse=False,
    labels=False,
    half_width=3.33,
    **kwargs,
):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
        labels (bool, optional): if set to `True`, the calculated avg rank
        values will be displayed
    """

    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None
    linesblank = 0
    distanceh = 0.1
    if labels:
        space_between_names = 0.3
        # calculate height needed height of an image
        minnotsignificant = 0.4

    else:
        space_between_names = 0.2
        # calculate height needed height of an image
        minnotsignificant = 0.3

    cline += distanceh

    if ax is None:
        width = half_width
        height = cline + ((k + 1) / 2) * space_between_names + minnotsignificant
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_subplot(111)
        ax.set_position([0, 0, 1, 1])

    fig = ax.get_figure()
    width = fig.get_figwidth() * ax.get_position().width
    height = fig.get_figheight() * ax.get_position().height
    ax.set_axis_off()

    textspace = textspace * width
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    hf = 1.0 / height  # height factor
    wf = 1.0 / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color="k", **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=1.5)

    bigtick = 0.3
    smalltick = 0.18
    linewidth = 1.0
    linewidth_sign = 4.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2), (rankpos(a), cline)], linewidth=1.5)

    for a in range(lowv, highv + 1):
        text(
            rankpos(a),
            cline - tick / 2 - 0.08,
            str(a),
            ha="center",
            va="bottom",
            size=9,
        )

    k = len(ssums)

    def filter_names(name):
        return pretty_names[name]

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line(
            [
                (rankpos(ssums[i]), cline),
                (rankpos(ssums[i]), chei),
                (textspace - 0.1, chei),
            ],
            linewidth=linewidth,
        )
        if labels:
            text(
                textspace + 0.11,
                chei - 0.15,
                format(ssums[i], ".2f"),
                ha="right",
                va="center",
                size=8,
            )
        text(
            textspace - 0.15,
            chei,
            filter_names(nnames[i]),
            ha="right",
            va="center",
            size=9,
        )

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line(
            [
                (rankpos(ssums[i]), cline),
                (rankpos(ssums[i]), chei),
                (textspace + scalewidth + 0.1, chei),
            ],
            linewidth=linewidth,
        )
        if labels:
            text(
                textspace + scalewidth - 0.11,
                chei - 0.15,
                format(ssums[i], ".2f"),
                ha="left",
                va="center",
                size=8,
            )
        text(
            textspace + scalewidth + 0.15,
            chei,
            filter_names(nnames[i]),
            ha="left",
            va="center",
            size=9,
        )

    # draw_lines(lines)
    start = cline + 0.2
    height = 0.1

    # draw no significant lines
    # get the cliques
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False

    for clq in cliques:
        if len(clq) == 1:
            continue

        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        line(
            [(rankpos(ssums[min_idx]), start), (rankpos(ssums[max_idx]), start)],
            linewidth=linewidth_sign,
        )
        start += height

    return ax
