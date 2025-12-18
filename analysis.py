import inspect
import multiprocessing as mp
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from reforge import io, mdm
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.utils import clean_dir, get_logger
import plots

logger = get_logger(__name__)

INPDB = 'input.pdb'
SELECTION = "name CA" 
TRJEXT = 'trr' # 'xtc' or 'trr'

allo_1 = [44, 203, 232, 249, 262, 286]
allo_fta = [244]
allosteric_sites = allo_1 + allo_fta
active_1 = [70, 73, 130, 132, 166, 170, 234]
active_2 = [73, 105, 166, 229, 234, 244, 275]
active_sites = sorted(list(set(active_1 + active_2)))
control_sites = [55, 80, 99, 120, 150, 180, 200, 222, 256]
all_sites = allosteric_sites + active_sites + control_sites


################################################################################
### PCA/Clustering ###
################################################################################
def pca_trajs(sysdir, sysname, selection=SELECTION, step=1):
    mdsys = MDSystem(sysdir, sysname)
    # clean_dir(mdsys.datdir, "*")
    tops = io.pull_files(mdsys.mddir, "topology.pdb")
    trajs = io.pull_files(mdsys.mddir, f"samples.{TRJEXT}")
    run_ids = [top.split("/")[-2] for top in tops]
    # Reading 
    logger.info("Reading trajectories")
    u = mda.Universe(tops[0], trajs, in_memory_step=step, ) # in_memory=True)
    logger.info(f'Selecting atoms for PCA analysis: {selection}')
    ag = u.atoms.select_atoms(selection)
    positions = io.read_positions(u, ag, sample_rate=1, b=0, e=1e9).T
    # PCA
    logger.info("Doing PCA")
    frames = np.arange(len(u.trajectory)) 
    edges = np.cumsum([len(r) for r in u.trajectory.readers])
    traj_ids = np.digitize(frames, edges, right=False)
    pca = PCA(n_components=3)
    x_r = pca.fit_transform(positions) # (n_samples, n_features)
    _plot_traj_pca(x_r, 0, 1, traj_ids, run_ids, mdsys, out_tag="runs_pca")
    _plot_traj_pca(x_r, 1, 2, traj_ids, run_ids, mdsys, out_tag="runs_pca")
    # Clustering
    _cluster(x_r, u, ag, mdsys, n_clusters=2)
    _filter_outliers(x_r, u, ag, mdsys)
    logger.info("Done!")


def _cluster(data, u, ag, mdsys, n_clusters=2):
    logger.info("Clustering")
    algo = GaussianMixture(n_components=n_clusters, random_state=0, n_init=10)
    # algo = KMeans(n_clusters=n_clusters, random_state=150, n_init=10)
    pred = algo.fit_predict(data)
    labels = []
    for idx, x in enumerate(np.unique(pred)):
        n_samples = np.sum(pred == x)
        label = f"cluster_{idx} with {n_samples} samples"
        labels.append(label)
    _plot_traj_pca(data, 0, 1, pred, labels, mdsys, out_tag="clust_pca")
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


def _filter_outliers(data, u, ag, mdsys):
    logger.info("Filtering outliers")
    pipe = StandardScaler(with_mean=True, with_std=True)
    Xz = pipe.fit_transform(data)
    ee = EllipticEnvelope(contamination=0.05, support_fraction=0.90,
        assume_centered=True,  random_state=None)
    pred = ee.fit_predict(Xz)               # +1 = inlier (main Gaussian), -1 = outlier
    scores = -ee.score_samples(Xz)          # larger => more outlier-ish
    labels = []
    for idx, x in enumerate(np.unique(pred)):
        n_samples = np.sum(pred == x)
        label = f"cluster_{idx} with {n_samples} samples"
        labels.append(label)
    _plot_traj_pca(data, 0, 1, pred, labels, mdsys, out_tag="filtered_pca")
    ag.atoms.write(str(mdsys.datdir / f"filtered.pdb"))
    mask = pred == +1
    subset = u.trajectory[mask]
    traj_path = str(mdsys.datdir / f"filtered.xtc")
    logger.info("Writing filtered cluster")
    with mda.Writer(traj_path, ag.n_atoms) as W:
        for ts in subset:   
            W.write(ag) 


def _plot_traj_pca(data, i, j, ids, labels, mdsys, skip=1, alpha=0.3, out_tag="pca",):
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


def clust_cov(sysdir, sysname, selection = SELECTION):
    logger.info("Doing cluster covariance analysis")
    mdsys = MDSystem(sysdir, sysname)
    clusters = io.pull_files(mdsys.datdir, "cluster*.xtc")
    tops = io.pull_files(mdsys.datdir, "topology*.pdb")
    clusters.append(mdsys.datdir / "filtered.xtc")
    tops.append(mdsys.datdir / "filtered.pdb")
    for idx, (cluster, top) in enumerate(zip(clusters, tops)):
        u = mda.Universe(top, cluster)
        logger.info(f'Selecting atoms for cluster DFI analysis: {selection}')
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

################################################################################
### DFI/DCI ###
################################################################################

def cov_analysis(sysdir, sysname, runname, selection=SELECTION):
    mdrun = MDRun(sysdir, sysname, runname)
    mdrun.covdir.mkdir(exist_ok=True, parents=True)
    top = mdrun.rundir / "topology.pdb"
    traj = mdrun.rundir / f"samples.{TRJEXT}"
    u = mda.Universe(top, traj, in_memory=False)
    logger.info(f'Selecting atoms for covariance analysis: {selection}')
    ag = u.atoms.select_atoms(selection)
    clean_dir(mdrun.covdir, "*npy")
    mdrun.get_covmats(u, ag, sample_rate=1, b=0, e=1e12, n=1, outtag="covmat") 
    mdrun.get_pertmats()
    mdrun.get_dfi(outtag="dfi")
    mdrun.get_dci(outtag="dci", asym=False)
    mdrun.get_dci(outtag="asym", asym=True)


def get_means_sems(sysdir, sysname):
    system = MDSystem(sysdir, sysname)   
    system.get_mean_sem(pattern="rmsf*.npy")
    system.get_mean_sem(pattern="dfi*.npy")
    system.get_mean_sem(pattern="dci*.npy")
    system.get_mean_sem(pattern="asym*.npy")
    system.get_mean_sem(pattern="covmat*.npy")
    plots.plot_dfi(system, tag='dfi')
    plots.plot_pdfi(system, tag='dfi')

################################################################################
### RMSD/RMSF Analysis ###
################################################################################

def rms_analysis(sysdir, sysname, runname, selection=SELECTION, step=1):
    mdsys = MDSystem(sysdir, sysname)
    mdrun = MDRun(sysdir, sysname, runname)
    rmsdir = mdrun.rmsdir
    rmsdir.mkdir(exist_ok=True)    
    top = mdrun.rundir / "topology.pdb"
    traj = mdrun.rundir / f"samples.{TRJEXT}"
    # Load trajectory
    u = mda.Universe(str(top), str(traj))
    atoms = u.select_atoms(selection)  
    # Calculate RMSD
    logger.info(f'Calculating RMSD and RMSF for selection: {selection}')
    rmsd_analysis = rms.RMSD(atoms, reference=atoms, select=selection)
    rmsd_analysis.run(step=step)
    # Calculate RMSF
    rmsf_analysis = rms.RMSF(atoms)
    rmsf_analysis.run(step=step)
    # Get residue IDs
    residue_ids = np.array([atom.resid for atom in atoms])
    # Save arrays
    np.save(rmsdir / "rmsd_values.npy", rmsd_analysis.rmsd[:, 2])  # RMSD values (in angstroms)
    np.save(rmsdir / "rmsd_times.npy", rmsd_analysis.rmsd[:, 1])   # Time values (in ps)
    np.save(rmsdir / "rmsf_values.npy", rmsf_analysis.rmsf)        # RMSF values (in angstroms)
    np.save(rmsdir / "residue_ids.npy", residue_ids)               # Residue IDs
    logger.info(f"Saved RMSD and RMSF data to {rmsdir}")
    # Plots
    logger.info("Generating RMSD and RMSF plots")
    plots.plot_rmsd(mdsys)
    plots.plot_rmsf(mdsys)
    
################################################################################
### TDLRT ###
################################################################################

def resid_to_index(pdb, resids):
    u = mda.Universe(pdb)
    cas = u.select_atoms("name CA")
    all_resids = np.array(cas.resids)
    all_ids = np.arange(len(all_resids))
    ids = all_ids[np.isin(all_resids, resids)]
    return ids


def map_to_subset_indices(site_ids, all_ids):
    """
    Map site indices to their positions within the all_sites subset.
    
    Parameters
    ----------
    site_ids : array-like
        Indices of specific sites in the full system
    all_ids : array-like
        Indices of all sites used in the subset
    
    Returns
    -------
    subset_ids : np.ndarray
        Positions of site_ids within all_ids
    """
    subset_ids = np.array([np.where(all_ids == idx)[0][0] for idx in site_ids])
    return subset_ids


def ca_to_3n_indices(ca_indices):
    """
    Convert CA indices to 3N indices (x, y, z components).
    
    Parameters
    ----------
    ca_indices : array-like
        CA atom indices
    
    Returns
    -------
    indices_3n : np.ndarray
        Sorted 3N indices (each CA index expanded to 3 indices for x, y, z)
    """
    indices_3n = np.sort(np.concatenate([ca_indices*3, ca_indices*3+1, ca_indices*3+2]))
    return indices_3n


def tdlrt_analysis(sysdir, sysname, runname, selection=SELECTION):
    mdrun = MDRun(sysdir, sysname, runname)
    mdrun.lrtdir.mkdir(exist_ok=True, parents=True)
    ps_path = mdrun.rundir / "positions.npy"
    vs_path = mdrun.rundir / "velocities.npy"
    pdb_id = '1btl'
    all_ids = resid_to_index(f'systems/{pdb_id}.pdb', all_sites)
    all_3n_ids = np.sort(np.concatenate([all_ids*3, all_ids*3+1, all_ids*3+2]))
    if (ps_path.exists() and vs_path.exists()):
        logger.info("Loading positions and velocities from %s", mdrun.rundir)
        ps = np.load(ps_path)[all_3n_ids]
        vs = np.load(vs_path)[all_3n_ids]
    else:
        traj = mdrun.rundir / f"samples.{TRJEXT}"
        top = mdrun.rundir / "topology.pdb"
        u = mda.Universe(top, traj)
        ag = u.atoms.select_atoms(selection)
        logger.info(f"Reading positions and velocities for selection: {selection}")
        ps = io.read_positions(u, ag) # (n_atoms*3, nframes)
        vs = io.read_velocities(u, ag) # (n_atoms*3, nframes)
    ps = ps - ps[:, 0][..., None]
    # CCF calculations
    adict = {'pv': (ps, vs), } 
    for key, item in adict.items(): # DT = TSTEP * NOUT
        v1, v2 = item
        # corr = mdm.ccf(v1, v2, ntmax=400, n=1, domain='frequency', mode='gpu', center=False, dtype=np.float32, buffer_c=0.9) # falls back on cpu if no cuda
        corr = mdm.cpsd(v1, v2, n=1, mode='gpu', center=False, dtype=np.float32, buffer_c=0.9) # falls back on cpu if no cuda
        corr_file = mdrun.lrtdir / f'cpsd_{key}.npy'
        np.save(corr_file, corr)    
        logger.info("Saved CCFs to %s", corr_file)


def perform_pca_analysis(X, labels, n_components=3):
    """
    Perform PCA on input data.
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix (n_samples, n_features)
    labels : np.ndarray
        Labels for each sample
    n_components : int
        Number of PCA components to compute
    
    Returns
    -------
    X_pca : np.ndarray
        Transformed data in PCA space
    pca : PCA object
        Fitted PCA object
    """
    logger.info(f"Performing PCA on data with shape: {X.shape}")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    logger.info(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")
    
    return X_pca, pca


def pca_cpsd_allosteric(sysdir, sysname, cpsd_file="cpsd_vv_av_ws100.npy"):
    """
    Perform PCA on CPSD signals along allosteric indices.
    Handles complex CPSD data by analyzing magnitude and phase separately.
    - Axis 0: allosteric indices
    - Axis 1: control vs active sites
    """
    mdsys = MDSystem(sysdir, sysname)
    pdb_id = '1btl'
    mdsys.datdir = Path("data") / "1btl_nve"
    mdsys.pngdir = Path("png") / "1btl_nve"
    
    # Get indices for different site types in the full system
    allo_ids = resid_to_index(f'systems/{pdb_id}.pdb', allosteric_sites)
    active_ids = resid_to_index(f'systems/{pdb_id}.pdb', active_sites)
    control_ids = resid_to_index(f'systems/{pdb_id}.pdb', control_sites)
    
    # Get indices for all sites (as used in tdlrt_analysis)
    all_ids = resid_to_index(f'systems/{pdb_id}.pdb', all_sites)
    
    # Map site indices to their positions in the all_sites subset
    allo_subset_ids = map_to_subset_indices(allo_ids, all_ids)
    active_subset_ids = map_to_subset_indices(active_ids, all_ids)
    control_subset_ids = map_to_subset_indices(control_ids, all_ids)
    
    # Load CPSD data
    cpsd_path = mdsys.datdir / cpsd_file
    if not cpsd_path.exists():
        logger.error(f"CPSD file not found: {cpsd_path}")
        return
    
    logger.info(f"Loading CPSD data from {cpsd_path}")
    cpsd = np.load(cpsd_path)  # Shape: (n_sites*3, n_sites*3, n_freq) or (n_sites*3, n_sites*3)
    logger.info(f"CPSD shape: {cpsd.shape}, dtype: {cpsd.dtype}")
    
    # Handle 2D or 3D arrays
    if cpsd.ndim == 3:
        # Average over frequency dimension
        cpsd_avg = np.mean(cpsd, axis=2)
    else:
        cpsd_avg = cpsd
    
    # Convert CA subset indices to 3N indices (x, y, z components)
    allo_3n_ids = ca_to_3n_indices(allo_subset_ids)
    active_3n_ids = ca_to_3n_indices(active_subset_ids)
    control_3n_ids = ca_to_3n_indices(control_subset_ids)
    
    logger.info(f"Allosteric 3N indices: {len(allo_3n_ids)}, Active 3N indices: {len(active_3n_ids)}, Control 3N indices: {len(control_3n_ids)}")
    
    # Extract CPSD submatrices: allosteric rows, active/control columns
    cpsd_allo_active = cpsd_avg[allo_3n_ids][:, active_3n_ids]  # (n_allo*3, n_active*3)
    cpsd_allo_control = cpsd_avg[allo_3n_ids][:, control_3n_ids]  # (n_allo*3, n_control*3)
    
    # Create labels
    labels = np.array(['control'] * len(control_3n_ids) + ['active'] * len(active_3n_ids))
    
    # Check if data is complex
    is_complex = np.iscomplexobj(cpsd_avg)
    
    if is_complex:
        logger.info("CPSD data is complex - analyzing magnitude and phase separately")
        
        # Magnitude analysis
        X_active_mag = np.abs(cpsd_allo_active).T
        X_control_mag = np.abs(cpsd_allo_control).T
        X_mag = np.vstack([X_control_mag, X_active_mag])
        
        X_pca_mag, pca_mag = perform_pca_analysis(X_mag, labels, n_components=3)
        
        # Save magnitude PCA results
        np.save(mdsys.datdir / "cpsd_pca_magnitude_components.npy", X_pca_mag)
        np.save(mdsys.datdir / "cpsd_pca_magnitude_variance.npy", pca_mag.explained_variance_ratio_)
        
        # Plot magnitude PCA
        plots.plot_pca_2d(X_pca_mag, labels, pca_mag.explained_variance_ratio_, mdsys,
                         title_prefix="PCA of CPSD Magnitude", 
                         filename_prefix="cpsd_pca_magnitude")
        
        # Phase analysis
        X_active_phase = np.angle(cpsd_allo_active).T
        X_control_phase = np.angle(cpsd_allo_control).T
        X_phase = np.vstack([X_control_phase, X_active_phase])
        
        X_pca_phase, pca_phase = perform_pca_analysis(X_phase, labels, n_components=3)
        
        # Save phase PCA results
        np.save(mdsys.datdir / "cpsd_pca_phase_components.npy", X_pca_phase)
        np.save(mdsys.datdir / "cpsd_pca_phase_variance.npy", pca_phase.explained_variance_ratio_)
        
        # Plot phase PCA
        plots.plot_pca_2d(X_pca_phase, labels, pca_phase.explained_variance_ratio_, mdsys,
                         title_prefix="PCA of CPSD Phase", 
                         filename_prefix="cpsd_pca_phase")
        
    else:
        logger.info("CPSD data is real - performing single PCA")
        
        # Real data analysis
        X_active = cpsd_allo_active.T
        X_control = cpsd_allo_control.T
        X = np.vstack([X_control, X_active])
        
        X_pca, pca = perform_pca_analysis(X, labels, n_components=3)
        
        # Save PCA results
        np.save(mdsys.datdir / "cpsd_pca_components.npy", X_pca)
        np.save(mdsys.datdir / "cpsd_pca_variance.npy", pca.explained_variance_ratio_)
        
        # Plot PCA
        plots.plot_pca_2d(X_pca, labels, pca.explained_variance_ratio_, mdsys,
                         title_prefix="PCA of CPSD", 
                         filename_prefix="cpsd_pca")
    
    logger.info("PCA analysis complete")


def running_window_average(data, window_size=100):
    """
    Apply running window average to data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    window_size : int
        Size of the running window (default: 100)
    
    Returns
    -------
    smoothed : np.ndarray
        Smoothed data with same shape as input
    """
    if data.ndim == 1:
        # 1D case
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(data, kernel, mode='same')
    elif data.ndim == 2:
        # 2D case - smooth along last axis
        smoothed = np.zeros_like(data)
        kernel = np.ones(window_size) / window_size
        for i in range(data.shape[0]):
            smoothed[i] = np.convolve(data[i], kernel, mode='same')
    elif data.ndim == 3:
        # 3D case - smooth along last axis (frequency)
        smoothed = np.zeros_like(data)
        kernel = np.ones(window_size) / window_size
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.iscomplexobj(data):
                    # Handle complex data separately for magnitude and phase
                    mag = np.abs(data[i, j])
                    phase = np.angle(data[i, j])
                    mag_smooth = np.convolve(mag, kernel, mode='same')
                    phase_smooth = np.convolve(phase, kernel, mode='same')
                    smoothed[i, j] = mag_smooth * np.exp(1j * phase_smooth)
                else:
                    smoothed[i, j] = np.convolve(data[i, j], kernel, mode='same')
    else:
        raise ValueError(f"Unsupported data dimensionality: {data.ndim}")
    
    return smoothed


def apply_running_average_to_files(datdir="data", pattern="*.npy", window_size=100):
    """
    Apply running window average to files matching pattern and save results.
    
    Parameters
    ----------
    sysdir : str or Path
        System directory
    sysname : str
        System name
    pattern : str
        File pattern to match (default: "*.npy")
    window_size : int
        Size of the running window (default: 100)
    """
    datdir = Path(datdir)
    files = io.pull_files(datdir, pattern)
    
    if not files:
        logger.warning(f'No files found matching pattern: {pattern}')
        return
    
    logger.info(f"Found {len(files)} files to process with window size {window_size}")
    
    for file_path in files:
        file_path = Path(file_path)
        logger.info(f"Processing {file_path.name}")
        # Load data
        data = np.load(file_path)
        # Apply running average
        smoothed = running_window_average(data, window_size=window_size)
        # Create output filename
        base_name = file_path.stem  # filename without extension
        out_name = f"{base_name}_ws{window_size}.npy"
        out_path = file_path.parent / out_name
        # Save smoothed data
        np.save(out_path, smoothed)
        logger.info(f"Saved smoothed data to {out_path}")
    
    logger.info("Running average complete")


def get_averages(sysdir, sysname, pattern="ccfs_pp*.npy", dtype=None):
    """Calculate average arrays across files matching pattern."""
    mdsys = MDSystem(sysdir, sysname)
    nprocs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    logger.info("Number of available processors: %s", nprocs)
    files = io.pull_files(sysdir, pattern)[::1]
    if not files:
        logger.info('Could not find files matching given pattern: %s. Maybe you forgot "*"?', pattern)
        return
    logger.info("Found %d files, starting processing: %s", len(files), files[0])
    # Discover minimal common shape (fast, uses mmap to avoid loading full arrays)
    shapes = []
    for f in files:
        try:
            arr = np.load(f, mmap_mode='r')
            if dtype is None:
                dtype = arr.dtype
            shapes.append(arr.shape)
        except Exception as e:
            logger.warning("Could not read shape for %s: %s", f, e)
    if not shapes:
        logger.info('No readable files found for pattern: %s', pattern)
        return
    min_shape = tuple(min(s[i] for s in shapes) for i in range(len(shapes[0])))
    logger.info('Running parallel get_averages with %d processes', nprocs)
    # split files into roughly equal batches
    batches = [files[i::nprocs] for i in range(nprocs)]
    work = [(batch, min_shape) for batch in batches if batch]
    with mp.Pool(processes=len(work)) as pool:
        results = pool.map(_process_batch, work)
    total_sum = np.zeros(min_shape, dtype=dtype)
    total_count = 0
    for local_sum, local_count in results:
        total_sum += local_sum
        total_count += local_count
    average = total_sum / total_count
    outdir = mdsys.datdir
    outdir.mkdir(exist_ok=True, parents=True)
    out_file = outdir / f"{pattern.split('*')[0]}_av.npy"
    np.save(out_file, average)
    logger.info("Saved averages to %s", out_file)


def _process_batch(args, dtype=np.float32):
    """Worker: load assigned files, crop to min_shape and return local sum and count."""
    files, min_shape = args
    s = tuple(slice(0, s) for s in min_shape)
    local_sum = np.zeros(min_shape, dtype=dtype)
    local_count = 0
    for f in files:
        logger.info("Processing %s", f)
        try:
            arr = np.load(f)
        except Exception as e:
            logger.warning("Could not load %s: %s", f, e)
            continue
        local_sum += arr[s]
        local_count += 1
        del arr
    return local_sum, local_count

################################################################################
### ENM analysis ###
################################################################################

def enm_analysis(sysdir, sysname):
    """Calculate ENM-based metrics."""
    system = MDSystem(sysdir, sysname)
    in_pdb = system.syspdb
    u = mda.Universe(in_pdb)
    ag = u.select_atoms("name CA")
    vecs = np.array(ag.positions).astype(np.float64) # (n_atoms, 3)
    hess = mdm.hessian(vecs, spring_constant=5, cutoff=11, dd=0) # distances in Angstroms, dd=0 no distance-dependence
    covmat = mdm.inverse_matrix(hess, device="gpu_dense", k_singular=6, n_modes=1000, dtype=np.float64)
    covmat = covmat * 1.25 # kb*T at 300K in kJ/mol
    outfile = system.datdir / "enm_cov.npy"
    np.save(outfile, covmat)
    pertmat = mdm.perturbation_matrix_iso(covmat)
    rmsf = np.sqrt(np.diag(covmat).reshape(-1, 3).sum(axis=1)) # (n_atoms,)
    dfi = mdm.dfi(pertmat)
    plots.simple_residue_plot(system, [rmsf], outtag="enm_rmsf")
    plots.simple_residue_plot(system, [dfi], outtag="enm_dfi")


def ca_hessian_from_md(sysdir, sysname):
    system = MDSystem(sysdir, sysname)
    covmat = np.load(system.datdir / "covmat_av.npy")
    hess = mdm.inverse_matrix(covmat, device="gpu_dense", k_singular=6, n_modes=1000, dtype=np.float64)
    hess = hess / 1.25 # kb*T at 300K in kJ/mol
    outfile = system.datdir / "md_hess.npy"
    np.save(outfile, hess)


################################################################################
### Bioemu ###
################################################################################

def sample_emu(sysdir, sysname, runname):
    from bioemu.sample import main as sample
    mdrun = MDRun(sysdir, sysname, runname)
    mdrun.prepare_files()
    sequence = _pdb_to_seq(mdrun.sysdir / INPDB)
    sample(sequence=sequence, num_samples=1000, batch_size_100=20, output_dir=mdrun.rundir)


def initiate_systems_from_emu(*args):
    logger.info("Preparing directories from EMU samples")
    emu_dir = Path("systems") / "emu"
    newsys_dir = Path("systems") / "1btl_nve"
    samples = emu_dir / "samples.xtc"
    top = emu_dir / "topology.pdb"
    u = mda.Universe(top, samples)
    step = 10  # every 10 frames
    for i, ts in enumerate(u.trajectory[1::step]):
        idx = i 
        outdir = newsys_dir / f"sample_{idx:03d}"
        outdir.mkdir(parents=True, exist_ok=True)
        outpdb = outdir / "sample.pdb"
        with mda.Writer(outpdb, u.atoms.n_atoms) as W:
            W.write(u.atoms)
        logger.info(f"Saved initial structure {i} to {outpdb}")


def _pdb_to_seq(pdb):
    u = mda.Universe(pdb)
    logger.info('Selecting protein atoms for sequence extraction')
    protein = u.select_atoms("protein")
    seq = "".join(res.resname for res in protein.residues)  # three-letter codes
    seq_oneletter = "".join(mda.lib.util.convert_aa_code(res.resname) for res in protein.residues)
    return seq_oneletter


if __name__ == "__main__":
    from reforge.cli import run_command
    run_command()