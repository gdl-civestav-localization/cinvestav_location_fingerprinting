import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import coverage


def plot(rssi_map, cover, best, figsize=(10, 8)):
    fig, ax = cover.L.showG('s', figsize=figsize)

    # plot the grid
    for k in cover.dap:
        p = cover.dap[k]['p']
        ax.plot(p[0], p[1], 'or')

    vmin = rssi_map.min()
    vmax = rssi_map.max()

    l = cover.grid[0, 0]
    r = cover.grid[-1, 0]
    b = cover.grid[0, 1]
    t = cover.grid[-1, -1]

    img = ax.imshow(rssi_map,
                    extent=(l, r, b, t),
                    origin='lower',
                    cmap='jet',
                    vmin=vmin,
                    vmax=vmax)

    # Put numbers
    for k in range(cover.na):
        ax.annotate(str(k), xy=(cover.pa[0, k], cover.pa[1, k]))
    ax.set_title(cover.title)

    # Put color bar
    divider = coverage.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clb = fig.colorbar(img, cax)
    clb.set_label(cover.legcb)

    # Print dotted line
    if best:
        if cover.mode != 'file':
            if cover.polar == 'o':
                ax.contour(np.sum(cover.bestsvo, axis=2)[cover.f, :].reshape(cover.nx, cover.ny).T, extent=(l, r, b, t),
                           linestyles='dotted')
            if cover.polar == 'p':
                ax.contour(np.sum(cover.bestsvp, axis=2)[cover.f, :].reshape(cover.nx, cover.ny).T, extent=(l, r, b, t),
                           linestyles='dotted')

    # display access points
    if cover.a == -1:
        ax.scatter(cover.pa[0, :], cover.pa[1, :], s=30, c='r', linewidth=0)
    else:
        ax.scatter(cover.pa[0, cover.a], cover.pa[1, cover.a], s=30, c='r', linewidth=0)
    plt.tight_layout()
    return fig, ax


def save_img(fig, ap, j):
    path = os.path.join(os.path.dirname(__file__), "..", "imagen", str(ap) + "- iter" + str(j) + ".png")
    print path
    fig.savefig(path)
    fig.clear()


def run_generate_img():
    c = coverage.Coverage('coverage.ini')  # Max: x=40, y=15
    c.cover()

    rssi_map = c.show(typ='pr', a=-1, polar='p', best=False, noise=False)
    fig, ax = plot(rssi_map=rssi_map, cover=c, best=False, figsize=(10, 8))
    path = os.path.join(os.path.dirname(__file__), "..", "imagen", "all aps without noise.png")
    print path
    fig.savefig(path)
    plt.show()


def run_generate_dataset():
    # Max: x=40, y=15
    map_width = 40
    map_height = 20
    step = 2
    iteration = 20

    # Generate dataset string
    matrix = []
    separator = ','
    dataset_name = os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset_simulation_{}.csv".format(iteration))

    c = coverage.Coverage('coverage.ini')
    c.cover()
    for x in range(0, map_width, step):
        for y in range(0, map_height, step):
            row = []
            # Without noise
            for ap in xrange(0, len(c.dap)):
                rssi_map = c.show(typ='pr', a=ap, polar='p', best=False, noise=False)
                row.append(rssi_map[y][x])
            row.append(x)
            row.append(y)
            matrix.append(row)

            # With noise
            for j in xrange(1, iteration):
                row = []
                for ap in xrange(0, len(c.dap)):
                    rssi_map = c.show(typ='pr', a=ap, polar='p', best=False, noise=True)
                    row.append(rssi_map[y][x])
                row.append(x)
                row.append(y)
                matrix.append(row)

    columns = []
    for ap in xrange(0, len(c.dap)):
        columns.append("ap{}".format(ap))
    columns.append("result_x")
    columns.append("result_y")
    df = pd.DataFrame(data=matrix, columns=columns)
    df.to_csv(dataset_name, sep=separator, encoding='utf-8')


if __name__ == '__main__':
    # run_generate_dataset()
    run_generate_img()
