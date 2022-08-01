"""
Adapted from Brian Hie's Evovelocity: https://github.com/brianhie/evolocity
"""

from scipy.sparse import issparse
import numpy as np
from inspect import signature
from scipy.sparse import csr_matrix, issparse
from scipy.stats import norm as normal
import warnings
from scanpy import Neighbors
from collections import abc
from pandas import unique, Index
from matplotlib import rcParams
from matplotlib.colors import is_color_like
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import SubplotSpec
from matplotlib import patheffects
from cycler import Cycler, cycler
from sklearn.neighbors import NearestNeighbors
import pandas as pd


RANDOM_STATE = 42


import matplotlib.pyplot as pl        

# velocity embedding stream code

def velocity_embedding_stream(
    adata,
    density=None,
    smooth=None,
    cutoff_perc=None,
    arrow_color=None,
    linewidth=None,
    arrowsize=None,
    n_neighbors=None,
    color=None,
    color_map=None,
    colorbar=True,
    size=None,
    alpha=0.3,
    V=None,
    sort_order=True,
    legend_loc="on data",
    legend_fontsize=None,
    legend_fontweight=None,
    fontsize=None,
    layer=None,
    ax=None,
):
    """\
    Stream plot of velocities on the embedding.
    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    density: `float` (default: 1)
        Amount of velocities to show - 0 none to 1 all
    smooth: `float` (default: 0.5)
        Multiplication factor for scale in Gaussian kernel around grid point.
    min_mass: `float` (default: 1)
        Minimum threshold for mass to be shown.
        It can range between 0 (all velocities) and 5 (large velocities only).
    cutoff_perc: `float` (default: `None`)
        If set, mask small velocities below a percentile threshold (between 0 and 100).
    linewidth: `float` (default: 1)
        Line width for streamplot.
    n_neighbors: `int` (default: None)
        Number of neighbors to consider around grid point.
    X: `np.ndarray` (default: None)
        Embedding grid point coordinates
    V: `np.ndarray` (default: None)
        Embedding grid velocity coordinates
    {scatter}
    Returns
    -------
        `matplotlib.Axis` if `show==False`
    """
    basis = 'tsne'
    lkeys = list(adata.layers.keys())
    
    colors = make_unique_list(color, allow_array=True)
    color = colors[0]
    color = default_color(adata) if color is None else color
    
    comps, obsm = np.array([0, 1]), adata.obsm
    X_emb = np.array(obsm[f"X_{basis}"][:, comps])
    V_emb = V[:, :2]
    X_grid, V_grid = compute_velocity_on_grid(
        X_emb=X_emb,
        V_emb=V_emb,
        density=1,
        smooth=smooth,
        n_neighbors=n_neighbors,
        autoscale=False,
        adjust_for_stream=True,
        cutoff_perc=cutoff_perc,
    )
    lengths = np.sqrt((V_grid ** 2).sum(0))
    linewidth = 1 if linewidth is None else linewidth

    scatter_kwargs = {
        "basis": basis,
        "sort_order": sort_order,
        "alpha": alpha,
        "color_map": color_map,
        "legend_loc": legend_loc,
        "legend_fontsize": legend_fontsize,
        "legend_fontweight": legend_fontweight,
        "colorbar": colorbar,
        "fontsize": fontsize,
    }

    density = 1 if density is None else density
    stream_kwargs = {
        "linewidth": linewidth,
        "arrowsize": arrowsize,
        "density": 2 * density,
        "zorder": 3,
        "color": arrow_color,
    }

    ax.streamplot(X_grid[0], X_grid[1], V_grid[0], V_grid[1], **stream_kwargs)

    ax = scatter(
        adata,
        layer=layer,
        color=color,
        size=size,
        ax=ax,
        zorder=0,
        **scatter_kwargs,
    )


def compute_velocity_on_grid(
        X_emb,
        V_emb,
        density=None,
        smooth=None,
        n_neighbors=None,
        min_mass=None,
        autoscale=True,
        adjust_for_stream=False,
        cutoff_perc=None,
        return_mesh=False,
):
    # remove invalid nodes
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, p_mass)[:, None]
    if min_mass is None:
        min_mass = 1

    X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
    ns = int(np.sqrt(len(V_grid[:, 0])))
    V_grid = V_grid.T.reshape(2, ns, ns)

    mass = np.sqrt((V_grid ** 2).sum(0))
    min_mass = 10 ** (min_mass - 6)  # default min_mass = 1e-5
    min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)
    cutoff = mass.reshape(V_grid[0].shape) < min_mass
    retain = mass.reshape(V_grid[0].shape) >= min_mass

    if cutoff_perc is None:
        cutoff_perc = 5
    length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T

    # and now let's remove all other neighbors
    length = length.reshape(ns, ns)
    cutoff |= length < np.percentile(length, cutoff_perc)
    retain &= length >= np.percentile(length, cutoff_perc)

    V_grid[0][cutoff] = np.nan
    V_grid[0][retain] = -1  # set arrow length! 
    
    if return_mesh:
        return X_grid, meshes_tuple, V_grid

    return X_grid, V_grid


def make_dense(X):
    if issparse(X):
        XA = X.A if X.ndim == 2 else X.A1
    else:
        XA = X.A1 if isinstance(X, np.matrix) else X
    return np.array(XA)


def is_categorical(data, c=None):
    from pandas.api.types import is_categorical as cat

    if c is None:
        return cat(data)  # if data is categorical/array
    return isinstance(c, str) and c in data.obs.keys() and cat(data.obs[c])


def is_view(adata):
    return (
        adata.is_view
        if hasattr(adata, "is_view")
        else adata.isview
        if hasattr(adata, "isview")
        else adata._isview
        if hasattr(adata, "_isview")
        else True
    )


def is_int(key):
    return isinstance(key, (int, np.integer))


def is_list(key):
    return isinstance(key, (list, tuple, np.record))


def to_list(key, max_len=20):
    if isinstance(key, Index) or is_list_of_str(key, max_len):
        key = list(key)
    return key if is_list(key) and (max_len is None or len(key) < max_len) else [key]


def is_list_or_array(key):
    return isinstance(key, (list, tuple, np.record, np.ndarray))


def is_list_of_str(key, max_len=None):
    if max_len is not None:
        return (
            is_list_or_array(key)
            and len(key) < max_len
            and all(isinstance(item, str) for item in key)
        )
    else:
        return is_list(key) and all(isinstance(item, str) for item in key)
    
    
def is_list_of_list(lst):
    return lst is not None and any(isinstance(l, list) for l in lst)


def is_list_of_int(lst):
    return is_list_or_array(lst) and all(is_int(item) for item in lst)


def to_val(key):
    return key[0] if isinstance(key, (list, tuple)) and len(key) == 1 else key
    
    
def to_valid_bases_list(adata, keys):
    if isinstance(keys, pd.DataFrame):
        keys = keys.index
    if not isinstance(keys, str):
        keys = list(np.ravel(keys))
    keys = to_list(keys, max_len=np.inf)
    if all(isinstance(item, str) for item in keys):
        for i, key in enumerate(keys):
            if key.startswith("X_"):
                keys[i] = key = key[2:]
            check_basis(adata, key)
        valid_keys = np.hstack(
            [
                adata.obs.keys(),
                adata.var.keys(),
                adata.varm.keys(),
                adata.obsm.keys(),
                [key[2:] for key in adata.obsm.keys()],
                list(adata.layers.keys()),
            ]
        )
        keys_ = keys
        keys = [key for key in keys if key in valid_keys or key in adata.var_names]
        keys_ = [key for key in keys_ if key not in keys]
        if len(keys_) > 0:
            msg_embedding = ""
            if len(keys_) == 1 and keys_[0] in {"diffmap", "umap", "tsne"}:
                msg_embedding = f"You need to run `evo.tl.{keys_[0]}` first."
            logg.warn(", ".join(keys_), "not found.", msg_embedding)
    return keys


def default_color(adata, add_outline=None):
    if (
        isinstance(add_outline, str)
        and add_outline in adata.var.keys()
        and "recover_dynamics" in adata.uns.keys()
        and add_outline in adata.uns["recover_dynamics"]
    ):
        return adata.uns["recover_dynamics"][add_outline]
    return (
        "clusters"
        if "clusters" in adata.obs.keys()
        else "louvain"
        if "louvain" in adata.obs.keys()
        else "grey"
    )


def default_color_map(adata, c):
    cmap = None
    if isinstance(c, str) and c in adata.obs.keys() and not is_categorical(adata, c):
        c = adata.obs[c]
    elif isinstance(c, int):
        cmap = "viridis_r"
    if len(np.array(c).flatten()) == adata.n_obs:
        try:
            if np.min(c) in [-1, 0, False] and np.max(c) in [1, True]:
                cmap = "viridis_r"
        except:
            cmap = None
    return cmap

    
def set_colorbar(smp, ax, orientation="vertical", labelsize=None):
    cax = inset_axes(ax, width="2%", height="90%", loc=4, borderpad=0)
    cb = pl.colorbar(smp, orientation=orientation, cax=cax)
    cb.set_alpha(1)
    cb.ax.tick_params(labelsize=labelsize)
    cb.draw_all()
    cb.locator = MaxNLocator(nbins=18, integer=True)
    cb.update_ticks()


def check_basis(adata, basis):
    if basis in adata.obsm.keys() and f"X_{basis}" not in adata.obsm.keys():
        adata.obsm[f"X_{basis}"] = adata.obsm[basis]
        info(f"Renamed '{basis}' to convention 'X_{basis}' (adata.obsm).")
        

def make_unique_list(key, allow_array=False):
    if isinstance(key, (Index, abc.KeysView)):
        key = list(key)
    is_list = (
        isinstance(key, (list, tuple, np.record))
        if allow_array
        else isinstance(key, (list, tuple, np.ndarray, np.record))
    )
    is_list_of_str = is_list and all(isinstance(item, str) for item in key)
    return key if is_list_of_str else key if is_list and len(key) < 20 else [key]


def make_unique_valid_list(adata, keys):
    keys = make_unique_list(keys)
    if all(isinstance(item, str) for item in keys):
        for i, key in enumerate(keys):
            if key.startswith("X_"):
                keys[i] = key = key[2:]
            check_basis(adata, key)
        valid_keys = np.hstack(
            [
                adata.obs.keys(),
                adata.var.keys(),
                adata.varm.keys(),
                adata.obsm.keys(),
                [key[2:] for key in adata.obsm.keys()],
                list(adata.layers.keys()),
            ]
        )
        keys_ = keys
        keys = [key for key in keys if key in valid_keys or key in adata.var_names]
        keys_ = [key for key in keys_ if key not in keys]
        if len(keys_) > 0:
            warn("IN UNIQUE LIST, ".join(keys_), "not found.")
    return keys


def velocity_embedding_changed(adata, basis, vkey):
    if f"X_{basis}" not in adata.obsm.keys():
        changed = False
    else:
        changed = f"{vkey}_{basis}" not in adata.obsm_keys()
        if f"{vkey}_params" in adata.uns.keys():
            sett = adata.uns[f"{vkey}_params"]
            changed |= "embeddings" not in sett or basis not in sett["embeddings"]
    return changed


def default_arrow(size):
    if isinstance(size, (list, tuple)) and len(size) == 3:
        head_l, head_w, ax_l = size
    elif isinstance(size, (int, np.integer, float)):
        head_l, head_w, ax_l = 12 * size, 10 * size, 8 * size
    else:
        head_l, head_w, ax_l = 12, 10, 8
    return head_l, head_w, ax_l


def get_colors(adata, c):
    if is_color_like(c):
        return c
    else:
        if f"{c}_colors" not in adata.uns.keys():
            palette = default_palette(None)
            palette = adjust_palette(palette, length=len(adata.obs[c].cat.categories))
            n_cats = len(adata.obs[c].cat.categories)
            adata.uns[f"{c}_colors"] = palette[:n_cats].by_key()["color"]
        if isinstance(adata.uns[f"{c}_colors"], dict):
            cluster_ix = adata.obs[c].values
        else:
            cluster_ix = adata.obs[c].cat.codes.values
        return np.array(
            [
                adata.uns[f"{c}_colors"][cluster_ix[i]]
                if cluster_ix[i] != -1
                else "lightgrey"
                for i in range(adata.n_obs)
            ]
        )


def interpret_colorkey(adata, c=None, layer=None, perc=None, use_raw=None):
    if c is None:
        c = default_color(adata)
    if issparse(c):
        c = make_dense(c).flatten()
    if is_categorical(adata, c):
        c = get_colors(adata, c)
    elif isinstance(c, str):
        if is_color_like(c) and not c in adata.var_names:
            pass
        elif c in adata.obs.keys():  # color by observation key
            c = adata.obs[c]
        elif c in adata.var_names or (
            use_raw and adata.raw is not None and c in adata.raw.var_names
        ):  # by gene
            if layer in adata.layers.keys():
                if perc is None and any(
                    l in layer for l in [ "velocity" ]
                ):
                    perc = [1, 99]  # to ignore outliers in non-logarithmized layers
                c = adata.obs_vector(c, layer=layer)
            elif layer is not None and np.any(
                [l in layer or "X" in layer for l in adata.layers.keys()]
            ):
                l_array = np.hstack(
                    [adata.obs_vector(c, layer=l)[:, None] for l in adata.layers.keys()]
                )
                l_array = pd.DataFrame(l_array, columns=adata.layers.keys())
                l_array.insert(0, "X", adata.obs_vector(c))
                c = np.array(l_array.astype(np.float32).eval(layer))
            else:
                if layer is not None and layer != "X":
                    logg.warn(layer, "not found. Using .X instead.")
                if adata.raw is None and use_raw:
                    raise ValueError("AnnData object does not have `raw` counts.")
                c = adata.raw.obs_vector(c) if use_raw else adata.obs_vector(c)
            c = c.A.flatten() if issparse(c) else c
        elif c in adata.var.keys():  # color by observation key
            c = adata.var[c]
        elif np.any([var_key in c for var_key in adata.var.keys()]):
            var_keys = [
                k for k in adata.var.keys() if not isinstance(adata.var[k][0], str)
            ]
            var = adata.var[list(var_keys)]
            c = var.astype(np.float32).eval(c)
        elif np.any([obs_key in c for obs_key in adata.obs.keys()]):
            obs_keys = [
                k for k in adata.obs.keys() if not isinstance(adata.obs[k][0], str)
            ]
            obs = adata.obs[list(obs_keys)]
            c = obs.astype(np.float32).eval(c)
        elif not is_color_like(c):
            raise ValueError(
                "color key is invalid! pass valid observation annotation or a gene name"
            )
        if not isinstance(c, str) and perc is not None:
            c = clip(c, perc=perc)
    else:
        c = np.array(c).flatten()
        if perc is not None:
            c = clip(c, perc=perc)
    return c


def set_frame(ax, frameon):
    frameon = settings._frameon if frameon is None else frameon
    if not frameon:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_frame_on(False)


def set_legend(
    adata,
    ax,
    value_to_plot,
    legend_loc,
    scatter_array,
    legend_fontweight,
    legend_fontsize,
    legend_fontoutline,
    groups,
):
    """
    Adds a legend to the given ax with categorial data.
    """
    # add legend
    if legend_fontoutline is None:
        legend_fontoutline = 1
    obs_vals = adata.obs[value_to_plot]
    obs_vals.cat.categories = obs_vals.cat.categories.astype(str)
    color_keys = adata.uns[f"{value_to_plot}_colors"]
    if isinstance(color_keys, dict):
        color_keys = np.array([color_keys[c] for c in obs_vals.cat.categories])
    valid_cats = np.where(obs_vals.value_counts()[obs_vals.cat.categories] > 0)[0]
    categories = np.array(obs_vals.cat.categories)[valid_cats]
    colors = np.array(color_keys)[valid_cats]

    if groups is not None:
        groups, groupby = get_groups(adata, groups, value_to_plot)
        # only label groups with the respective color
        groups = [g for g in groups if g in categories]
        colors = [colors[list(categories).index(x)] for x in groups]
        categories = groups

    if legend_loc == "on data":
        legend_fontweight = "bold" if legend_fontweight is None else legend_fontweight
        # identify centroids to put labels
        texts = []
        for label in categories:
            x_pos, y_pos = np.nanmedian(scatter_array[obs_vals == label, :], axis=0)
            if isinstance(label, str):
                label = label.replace("_", " ")
            kwargs = dict(verticalalignment="center", horizontalalignment="center")
            kwargs.update(dict(weight=legend_fontweight, fontsize=legend_fontsize))
            pe = [patheffects.withStroke(linewidth=legend_fontoutline, foreground="w")]
            text = ax.text(x_pos, y_pos, label, path_effects=pe, **kwargs)
            texts.append(text)

    else:
        for idx, label in enumerate(categories):
            if isinstance(label, str):
                label = label.replace("_", " ")
            ax.scatter([], [], c=[colors[idx]], label=label)
        ncol = 1 if len(categories) <= 14 else 2 if len(categories) <= 30 else 3
        kwargs = dict(frameon=False, fontsize=legend_fontsize, ncol=ncol)
        if legend_loc == "upper right":
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), **kwargs)
        elif legend_loc == "lower right":
            ax.legend(loc="lower left", bbox_to_anchor=(1, 0), **kwargs)
        elif "right" in legend_loc:  # 'right', 'center right', 'right margin'
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), **kwargs)
        elif legend_loc != "none":
            ax.legend(loc=legend_loc, **kwargs)
            
            
def update_axes(
    ax,
    xlim=None,
    ylim=None,
    fontsize=None,
    is_embedding=False,
    frameon=None,
    figsize=None,
):
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    frameon = _frameon if frameon is None else frameon
    ax.set_xlabel("TSNE 1")
    ax.set_ylabel("TSNE 2")
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    kwargs = dict(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.tick_params(which="both", **kwargs)

    if rcParams["savefig.transparent"]:
        ax.patch.set_alpha(0)


def scatter(
    adata=None,
    basis=None,
    x=None,
    y=None,
    vkey=None,
    color=None,
    use_raw=None,
    layer=None,
    color_map=None,
    colorbar=None,
    palette=None,
    size=None,
    alpha=None,
    linewidth=None,
    linecolor=None,
    perc=None,
    groups=None,
    sort_order=True,
    components=None,
    projection=None,
    legend_loc=None,
    legend_loc_lines=None,
    legend_fontsize=None,
    legend_fontweight=None,
    legend_fontoutline=None,
    xlabel=None,
    ylabel=None,
    title=None,
    fontsize=None,
    figsize=None,
    xlim=None,
    ylim=None,
    add_density=None,
    add_assignments=None,
    add_linfit=None,
    add_polyfit=None,
    add_rug=None,
    add_text=None,
    add_text_pos=None,
    add_outline=None,
    outline_width=None,
    outline_color=None,
    n_convolve=None,
    smooth=None,
    rescale_color=None,
    color_gradients=None,
    dpi=None,
    frameon=None,
    zorder=None,
    ncols=None,
    nrows=None,
    wspace=None,
    hspace=None,
    show=None,
    save=None,
    ax=None,
    **kwargs,
):
    """\
    Scatter plot along observations or variables axes.
    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    x: `str`, `np.ndarray` or `None` (default: `None`)
        x coordinate
    y: `str`, `np.ndarray` or `None` (default: `None`)
        y coordinate
    {scatter}
    Returns
    -------
        If `show==False` a `matplotlib.Axis`
    """
    if adata is None and (x is not None and y is not None):
        adata = AnnData(np.stack([x, y]).T)

    # restore old conventions
    add_assignments = kwargs.pop("show_assignments", add_assignments)
    add_linfit = kwargs.pop("show_linear_fit", add_linfit)
    add_polyfit = kwargs.pop("show_polyfit", add_polyfit)
    add_density = kwargs.pop("show_density", add_density)
    add_rug = kwargs.pop("rug", add_rug)
    basis = kwargs.pop("var_names", basis)

    # keys for figures (fkeys) and multiple plots (mkeys)
    fkeys = ["adata", "show", "save", "groups", "ncols", "nrows", "wspace", "hspace"]
    fkeys += ["ax", "kwargs"]
    mkeys = ["color", "layer", "basis", "components", "x", "y", "xlabel", "ylabel"]
    mkeys += ["title", "color_map", "add_text"]
    scatter_kwargs = {"show": False, "save": False}
    for key in signature(scatter).parameters:
        if key not in mkeys + fkeys:
            scatter_kwargs[key] = eval(key)
    mkwargs = {}
    for key in mkeys:  # mkwargs[key] = key for key in mkeys
        mkwargs[key] = eval("{0}[0] if is_list({0}) else {0}".format(key))

    # use c & color and cmap & color_map interchangeably,
    # and plot each group separately if groups is 'all'
    if "c" in kwargs:
        color = kwargs.pop("c")
    if "cmap" in kwargs:
         color_map = kwargs.pop("cmap")
    if "rasterized" not in kwargs:
        kwargs["rasterized"] = False
    if isinstance(color_map, (list, tuple)) and all(
        [is_color_like(c) or c == "transparent" for c in color_map]
    ):
        color_map = rgb_custom_colormap(colors=color_map)
    if isinstance(groups, str) and groups == "all":
        if color is None:
            color = default_color(adata)
        if is_categorical(adata, color):
            vc = adata.obs[color].value_counts()
            groups = [[c] for c in vc[vc > 0].index]
    if isinstance(add_text, (list, tuple, np.ndarray, np.record)):
        add_text = list(np.array(add_text, dtype=str))

    # create list of each mkey and check if all bases are valid.
    color = to_list(color, max_len=None)
    layer, components = to_list(layer), to_list(components)
    x, y, basis = to_list(x), to_list(y), to_valid_bases_list(adata, basis)

    # get multikey (with more than one element)
    multikeys = eval(f"[{','.join(mkeys)}]")
    if is_list_of_list(groups):
        multikeys.append(groups)
    key_lengths = np.array([len(key) if is_list(key) else 1 for key in multikeys])
    multikey = (
        multikeys[np.where(key_lengths > 1)[0][0]] if np.max(key_lengths) > 1 else None
    )

    # make sure that there are no more lists, e.g. ['clusters'] becomes 'clusters'
    color_map = to_val(color_map)
    color, layer, basis = to_val(color), to_val(layer), to_val(basis)
    x, y, components = to_val(x), to_val(y), to_val(components)
    xlabel, ylabel, title = to_val(xlabel), to_val(ylabel), to_val(title)

    # set color, color_map, edgecolor, basis, linewidth, frameon, use_raw
    if color is None:
        color = default_color(adata, add_outline)
    if "cmap" not in kwargs:
        kwargs["cmap"] = (
            default_color_map(adata, color) if color_map is None else color_map
        )
    if "s" not in kwargs:
        kwargs["s"] = default_size(adata) if size is None else size
    if "edgecolor" not in kwargs:
        kwargs["edgecolor"] = "none"
    is_embedding = ((x is None) | (y is None)) and basis not in adata.var_names
    if basis is None and is_embedding:
        basis = default_basis(adata)
    if linewidth is None:
        linewidth = 1
    if linecolor is None:
        linecolor = "k"
    if frameon is None:
        frameon = True if not is_embedding else False
    if isinstance(groups, str):
        groups = [groups]
    if use_raw is None and basis not in adata.var_names:
        use_raw = layer is None and adata.raw is not None
    if projection == "3d":
        from mpl_toolkits.mplot3d import Axes3D

    if is_embedding:
        X_emb = adata.obsm[f"X_{basis}"][:, np.array([0, 1])]
        x, y = X_emb[:, 0], X_emb[:, 1]

    x, y = make_dense(x).flatten(), make_dense(y).flatten()

    if color == "ascending":
        color = np.linspace(0, 1, len(x))
    elif color == "descending":
        color = np.linspace(1, 0, len(x))

    # set color
    if (
        basis in adata.var_names
        and isinstance(color, str)
        and color in adata.layers.keys()
    ):
        # phase portrait: color=basis, layer=color
        c = interpret_colorkey(adata, basis, color, perc, use_raw)
    else:
        # embedding, gene trend etc.
        c = interpret_colorkey(adata, color, layer, perc, use_raw)

    if c is not None and not isinstance(c, str) and not isinstance(c[0], str):
        # smooth color values across neighbors and rescale
        if smooth and len(c) == adata.n_obs:
            n_neighbors = None if isinstance(smooth, bool) else smooth
            c = get_connectivities(adata, n_neighbors=n_neighbors).dot(c)
        # rescale color values to min and max acc. to rescale_color tuple
        if rescale_color is not None:
            try:
                c += rescale_color[0] - np.nanmin(c)
                c *= rescale_color[1] / np.nanmax(c)
            except:
                logg.warn("Could not rescale colors. Pass a tuple, e.g. [0,1].")

    # set vmid to 0 if color values obtained from velocity expression
    if not np.any([v in kwargs for v in ["vmin", "vmid", "vmax"]]) and np.any(
        [
            isinstance(v, str)
            and "time" not in v
            and (v.endswith("velocity") or v.endswith("transition"))
            for v in [color, layer]
        ]
    ):
        kwargs["vmid"] = 0

    # introduce vmid by setting vmin and vmax accordingly
    if "vmid" in kwargs:
        vmid = kwargs.pop("vmid")
        if vmid is not None:
            if not (isinstance(c, str) or isinstance(c[0], str)):
                lb, ub = np.min(c), np.max(c)
                crange = max(np.abs(vmid - lb), np.abs(ub - vmid))
                kwargs.update({"vmin": vmid - crange, "vmax": vmid + crange})

    x, y = np.ravel(x), np.ravel(y)
    if len(x) != len(y):
        raise ValueError("x or y do not share the same dimension.")

    if not isinstance(c, str):
        c = np.ravel(c) if len(np.ravel(c)) == len(x) else c
        if len(c) != len(x):
            c = "grey"
            if not isinstance(color, str) or color != default_color(adata):
                logg.warn("Invalid color key. Using grey instead.")

    # store original order of color values
    color_array, scatter_array = c, np.stack([x, y]).T

    # set color to grey for NAN values and for cells that are not in groups
    if (
        groups is not None
        or is_categorical(adata, color)
        and np.any(pd.isnull(adata.obs[color]))
    ):
        if isinstance(groups, (list, tuple, np.record)):
            groups = unique(groups)
        zorder = 0 if zorder is None else zorder
        pop_keys = ["groups", "add_linfit", "add_polyfit", "add_density"]
        _ = [scatter_kwargs.pop(key, None) for key in pop_keys]
        ax = scatter(
            adata,
            x=x,
            y=y,
            basis=basis,
            layer=layer,
            color="lightgrey",
            ax=ax,
            **scatter_kwargs,
        )
        if groups is not None and len(groups) == 1:
            if (
                isinstance(groups[0], str)
                and groups[0] in adata.var.keys()
                and basis in adata.var_names
            ):
                groups = f"{adata[:, basis].var[groups[0]][0]}"
        idx = groups_to_bool(adata, groups, color)
        if idx is not None:
            if np.sum(idx) > 0:  # if any group to be highlighted
                x, y = x[idx], y[idx]
                if not isinstance(c, str) and len(c) == adata.n_obs:
                    c = c[idx]
                if isinstance(kwargs["s"], np.ndarray):
                    kwargs["s"] = np.array(kwargs["s"])[idx]
                if (
                    title is None
                    and groups is not None
                    and len(groups) == 1
                    and isinstance(groups[0], str)
                ):
                    title = groups[0]
            else:  # if nothing to be highlighted
                add_linfit, add_polyfit, add_density = None, None, None

    # check if higher value points should be plotted on top
    if not isinstance(c, str) and len(c) == len(x):
        order = None
        if sort_order and not is_categorical(adata, color):
            order = np.argsort(c)
        elif not sort_order and is_categorical(adata, color):
            counts = get_value_counts(adata, color)
            np.random.seed(0)
            nums, p = np.arange(0, len(x)), counts / np.sum(counts)
            order = np.random.choice(nums, len(x), replace=False, p=p)
        if order is not None:
            x, y, c = x[order], y[order], c[order]
            if isinstance(kwargs["s"], np.ndarray):  # sort sizes if array-type
                kwargs["s"] = np.array(kwargs["s"])[order]

    smp = ax.scatter(
        x, y, c=c, alpha=alpha, marker=".", zorder=zorder, **kwargs
    )

    outline_dtypes = (list, tuple, np.ndarray, int, np.int_, str)
    if isinstance(add_outline, outline_dtypes) or add_outline:
        if isinstance(add_outline, (list, tuple, np.record)):
            add_outline = unique(add_outline)
        if (
            add_outline is not True
            and isinstance(add_outline, (int, np.int_))
            or is_list_of_int(add_outline)
            and len(add_outline) != len(x)
        ):
            add_outline = np.isin(np.arange(len(x)), add_outline)
            add_outline = np.array(add_outline, dtype=bool)
            if outline_width is None:
                outline_width = (0.6, 0.3)
        if isinstance(add_outline, str):
            if add_outline in adata.var.keys() and basis in adata.var_names:
                add_outline = f"{adata[:, basis].var[add_outline][0]}"
        idx = groups_to_bool(adata, add_outline, color)
        if idx is not None and np.sum(idx) > 0:  # if anything to be outlined
            zorder = 2 if zorder is None else zorder + 2
            if kwargs["s"] is not None:
                kwargs["s"] *= 1.2
            # restore order of values
            x, y = scatter_array[:, 0][idx], scatter_array[:, 1][idx]
            c = color_array
            if not isinstance(c, str) and len(c) == adata.n_obs:
                c = c[idx]
            if isinstance(kwargs["s"], np.ndarray):
                kwargs["s"] = np.array(kwargs["s"])[idx]
            if isinstance(c, np.ndarray) and not isinstance(c[0], str):
                if "vmid" not in kwargs and "vmin" not in kwargs:
                    kwargs["vmin"] = np.min(color_array)
                if "vmid" not in kwargs and "vmax" not in kwargs:
                    kwargs["vmax"] = np.max(color_array)
            ax.scatter(
                x, y, c=c, alpha=alpha, marker=".", zorder=zorder, **kwargs
            )
        if idx is None or np.sum(idx) > 0:  # if all or anything to be outlined
            plot_outline(
                x, y, kwargs, outline_width, outline_color, zorder, ax=ax
            )
        if idx is not None and np.sum(idx) == 0:  # if nothing to be outlined
            add_linfit, add_polyfit, add_density = None, None, None

    # set legend if categorical categorical color vals
    if is_categorical(adata, color) and len(scatter_array) == adata.n_obs:
        legend_loc = default_legend_loc(adata, color, legend_loc)
        g_bool = groups_to_bool(adata, add_outline, color)
        if not (add_outline is None or g_bool is None):
            groups = add_outline
        set_legend(
            adata,
            ax,
            color,
            legend_loc,
            scatter_array,
            legend_fontweight,
            legend_fontsize,
            legend_fontoutline,
            groups,
        )

    update_axes(ax, xlim, ylim, fontsize, is_embedding, frameon, figsize)

    if colorbar is not False:
        if not isinstance(c, str) and not is_categorical(adata, color):
            labelsize = fontsize * 0.75 if fontsize is not None else None
            set_colorbar(smp, ax=ax, labelsize=labelsize)

