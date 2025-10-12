import logging
from pathlib import Path
import sys
import warnings
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import MDAnalysis as mda
from sklearn.decomposition import PCA
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from reforge import io, mdm
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.utils import clean_dir, logger
import plots


def pca_trajs(sysdir, sysname):
    selection = SELECTION
    step = 1 # in frames
    mdsys = MDSystem(sysdir, sysname)
    mdsys.prepare_files()
    clean_dir(mdsys.datdir, "*")
    tops = io.pull_files(mdsys.mddir, "conv.pdb")
    trajs = io.pull_files(mdsys.mddir, "samples.xtc")
    run_ids = [top.split("/")[-2] for top in tops]
    # Reading 
    logger.info("Reading trajectories")
    u = mda.Universe(tops[0], trajs, in_memory_step=step, ) # in_memory=True)
    ag = u.atoms.select_atoms(selection)
    positions = io.read_positions(u, ag, sample_rate=1, b=0, e=1e9).T
    # PCA
    logger.info("Doing PCA")
    frames = np.arange(len(u.trajectory)) 
    edges = np.cumsum([len(r) for r in u.trajectory.readers])
    traj_ids = np.digitize(frames, edges, right=False)
    pca = PCA(n_components=3)
    x_r = pca.fit_transform(positions) # (n_samples, n_features)
    plot_traj_pca(x_r, 0, 1, traj_ids, run_ids, mdsys, out_tag="runs_pca")
    plot_traj_pca(x_r, 1, 2, traj_ids, run_ids, mdsys, out_tag="runs_pca")
    # Clustering
    cluster(x_r, u, ag, mdsys, n_clusters=2)
    filter_outliers(x_r, u, ag, mdsys)
    logger.info("Done!")


def cluster(data, u, ag, mdsys, n_clusters=2):
    logger.info("Clustering")
    algo = GaussianMixture(n_components=n_clusters, random_state=0, n_init=10)
    # algo = KMeans(n_clusters=n_clusters, random_state=150, n_init=10)
    pred = algo.fit_predict(data)
    labels = []
    for idx, x in enumerate(np.unique(pred)):
        n_samples = np.sum(pred == x)
        label = f"cluster_{idx} with {n_samples} samples"
        labels.append(label)
    plot_traj_pca(data, 0, 1, pred, labels, mdsys, out_tag="clust_pca")
    # plt.scatter(centers[:, 0], centers[:, 1], c="r", s=20)
    for idx, x in enumerate(np.unique(pred)):
        ag.atoms.write(str(mdsys.datdir / f"topology_{idx}.pdb"))
        mask = pred == x
        subset = u.trajectory[mask]
        traj_path = str(mdsys.datdir / f"cluster_{idx}.xtc")
        logger.info(f"Writing cluster %s", idx)
        with mda.Writer(traj_path, ag.n_atoms) as W:
            for ts in subset:   
                W.write(ag) 


def filter_outliers(data, u, ag, mdsys):
    logger.info("Filtering outliers")
    pipe = StandardScaler(with_mean=True, with_std=True)
    Xz = pipe.fit_transform(data)
    ee = EllipticEnvelope(contamination=0.05, support_fraction=0.9,
        assume_centered=True,  random_state=None)
    pred = ee.fit_predict(Xz)               # +1 = inlier (main Gaussian), -1 = outlier
    scores = -ee.score_samples(Xz)          # larger => more outlier-ish
    labels = []
    for idx, x in enumerate(np.unique(pred)):
        n_samples = np.sum(pred == x)
        label = f"cluster_{idx} with {n_samples} samples"
        labels.append(label)
    plot_traj_pca(data, 0, 1, pred, labels, mdsys, out_tag="filtered_pca")
    ag.atoms.write(str(mdsys.datdir / f"filtered.pdb"))
    mask = pred == +1
    subset = u.trajectory[mask]
    traj_path = str(mdsys.datdir / f"filtered.xtc")
    logger.info("Writing filtered cluster")
    with mda.Writer(traj_path, ag.n_atoms) as W:
        for ts in subset:   
            W.write(ag) 


def plot_traj_pca(data, i, j, ids, labels, mdsys, skip=1, alpha=0.3, out_tag="pca",):
    unique_ids = np.unique(ids)
    norm = mcolors.Normalize(vmin=min(ids), vmax=max(ids))
    cmap = plt.get_cmap("viridis")
    plt.figure()
    for tid, label in zip(unique_ids, labels):
        mask = ids == tid
        plt.scatter(data[mask, i][::skip], data[mask, j][::skip],
                    alpha=alpha,
                    color=cmap(norm(tid)),
                    label=label)
    plt.legend()
    plt.xlabel(f"PC{i+1}")
    plt.ylabel(f"PC{j+1}")
    plt.savefig(mdsys.pngdir / f"{out_tag}_{i}{j}.png")
    plt.close()


def clust_cov(sysdir, sysname):
    logger.info("Doing cluster covariance analysis")
    mdsys = MDSystem(sysdir, sysname)
    selection = SELECTION
    clusters = io.pull_files(mdsys.datdir, "cluster*.xtc")
    tops = io.pull_files(mdsys.datdir, "topology*.pdb")
    clusters.append(mdsys.datdir / "filtered.xtc")
    tops.append(mdsys.datdir / "filtered.pdb")
    for idx, (cluster, top) in enumerate(zip(clusters, tops)):
        u = mda.Universe(top, cluster)
        ag = u.atoms.select_atoms(selection)
        dtype = np.float32
        positions = io.read_positions(u, ag, sample_rate=1, b=0, e=1e9, dtype=dtype)
        logger.info("Calculating")
        covmat = mdm.covariance_matrix(positions, dtype=dtype)
        pertmat = mdm.perturbation_matrix_iso(covmat, dtype=dtype)
        dfi_res = mdm.dfi(pertmat)
        idx = 'filt' if cluster == mdsys.datdir / "filtered.xtc" else idx
        np.save(mdsys.datdir / f"cdfi_{idx}_av.npy", dfi_res)
    plots.plot_cluster_dfi(mdsys, tag='cdfi')


def tdlrt_analysis(sysdir, sysname, runname):
    mdrun = MDRun(sysdir, sysname, runname)
    # traj = str(mdrun.rundir / "md.trr")
    # top = str(mdrun.rundir / "md.pdb")
    traj = str(mdrun.rundir / "samples.trr")
    top = str(mdrun.rundir / "topology.pdb")
    u = mda.Universe(top, traj)
    vs = io.read_velocities(u, u.atoms) # (n_atoms*3, nframes)
    ps = io.read_positions(u, u.atoms) # (n_atoms*3, nframes)
    # CALC CCFs
    # adict = {'pv': (ps, vs), 'vv': (vs, vs), }
    adict = {'pv': (ps, vs)}
    for key, item in adict.items():
        v1, v2 = item
        corr = mdm.ccf(v1, v2, ntmax=400, n=1, mode='serial', center=False, dtype=np.float32)  # serial parallel or gpu
        corr_file = mdrun.lrtdir / f'ccf_{key}.npy'
        np.save(corr_file, corr)    
        logger.info("Saved CCFs to %s", corr_file)



