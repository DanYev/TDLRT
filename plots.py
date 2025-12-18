import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
from reforge import io, mdm
from reforge.mdsystem import gmxmd
from reforge.plotting import *
from reforge.utils import logger


def plot_pca_2d(X_pca, labels, variance_ratio, mdsys, title_prefix="PCA", filename_prefix="pca", 
                label_colors=None):
    """
    Plot 2D PCA results.
    
    Parameters
    ----------
    X_pca : np.ndarray
        PCA-transformed data
    labels : np.ndarray
        Labels for each sample
    variance_ratio : np.ndarray
        Explained variance ratios
    mdsys : MDSystem
        System object with pngdir attribute
    title_prefix : str
        Prefix for plot title
    filename_prefix : str
        Prefix for output filename
    label_colors : dict, optional
        Dictionary mapping labels to colors
    """
    unique_labels = np.unique(labels)
    if label_colors is None:
        label_colors = {'control': 'blue', 'active': 'red'}
    
    # PC1 vs PC2
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        mask = labels == label
        color = label_colors.get(label, 'gray')
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   alpha=0.6, s=50, c=color, label=label, edgecolors='k')
    plt.xlabel(f'PC1 ({variance_ratio[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({variance_ratio[1]*100:.1f}%)', fontsize=12)
    plt.title(f'{title_prefix}\n(Control vs Active Sites along Allosteric Indices)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    outfile = mdsys.pngdir / f"{filename_prefix}_pc1_pc2.png"
    plt.savefig(outfile, dpi=300)
    plt.close()
    logger.info(f"Saved PCA plot: {outfile}")
    
    # PC2 vs PC3 (if available)
    if X_pca.shape[1] >= 3:
        plt.figure(figsize=(10, 8))
        for label in unique_labels:
            mask = labels == label
            color = label_colors.get(label, 'gray')
            plt.scatter(X_pca[mask, 1], X_pca[mask, 2], 
                       alpha=0.6, s=50, c=color, label=label, edgecolors='k')
        plt.xlabel(f'PC2 ({variance_ratio[1]*100:.1f}%)', fontsize=12)
        plt.ylabel(f'PC3 ({variance_ratio[2]*100:.1f}%)', fontsize=12)
        plt.title(f'{title_prefix}\n(Control vs Active Sites along Allosteric Indices)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        outfile = mdsys.pngdir / f"{filename_prefix}_pc2_pc3.png"
        plt.savefig(outfile, dpi=300)
        plt.close()
        logger.info(f"Saved PCA plot: {outfile}")


def pull_data(datdir, metric):
    files = io.pull_files(datdir, metric)
    datas = [np.load(f) for f in files if '_av' in f]
    errs = [np.load(f) for f in files if '_err' in f]
    fnames = [f.split("/")[-1] for f in files if '_av' in f]
    return datas, errs, fnames


def set_bfactors_by_residue(in_pdb, bfactors, out_pdb=None):
    atoms = io.pdb2atomlist(in_pdb)
    residues = atoms.residues
    for idx, residue in enumerate(residues):
        for atom in residue:
            atom.bfactor = bfactors[idx]
    if out_pdb:
        atoms.write_pdb(out_pdb)
    return atoms


def set_bfactors_by_atom(in_pdb, bfactors, out_pdb=None):
    atoms = io.pdb2atomlist(in_pdb)
    for idx, atom in enumerate(atoms):
        atom.bfactor = bfactors[idx]
    if out_pdb:
        atoms.write_pdb(out_pdb)
    return atoms


def set_ax_parameters(ax, xlabel=None, ylabel=None, axtitle=None, loc=None):
    """
    ax - matplotlib ax object
    """
    # Set axis labels and title with larger font sizes
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(axtitle, fontsize=16)
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5, width=1.5)
    # Increase spine width for a bolder look
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # Add a legend with custom font size and no frame
    legend = ax.legend(fontsize=14, frameon=False, loc=loc)
    # Optionally, add gridlines
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_dfi(system, tag='dfi'):
    datas, errs, fnames = pull_data(system.datdir, f"{tag}*")
    xs = [np.arange(len(data))+26 for data in datas]
    labels = [f.split(".")[0] for f in fnames]
    params = [{'lw':2, 'label':label} for label in labels]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_errorbar(ax, xs, datas, errs, params, alpha=0.7)
    # make_plot(ax, xs, datas, params)
    set_ax_parameters(ax, xlabel='Residue', ylabel='DFI', loc='upper right')
    plot_figure(fig, ax, figname=system.sysname.upper(), figpath=system.pngdir / f"{tag}.png",)


def plot_pdfi(system, tag='dfi'):
    datas, errs, fnames = pull_data(system.datdir, f"{tag}*")
    xs = [np.arange(len(data))+26 for data in datas]
    datas = [mdm.percentile(data) for data in datas]
    labels = [f.split(".")[0] for f in fnames]
    params = [{'lw':2, 'label':label} for label in labels]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, datas, params)
    set_ax_parameters(ax, xlabel='Residue', ylabel='%DFI', loc='lower right')
    plot_figure(fig, ax, figname=system.sysname.upper(), figpath=system.pngdir / f"p{tag}.png",)


def plot_cluster_dfi(system, tag='cdfi'):
    datas, errs, fnames = pull_data(system.datdir, f"{tag}*")
    xs = [np.arange(len(data))+26 for data in datas]
    pdatas = [mdm.percentile(data) for data in datas]
    labels = [f.split(".")[0] for f in fnames]
    colors = ['silver', 'grey', 'black']
    lws = [1, 1, 2]
    params = [{'color':c, 'label':l, 'lw':lw, } for c, l, lw in zip(colors, labels, lws)]
    # Plotting DFI
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, datas, params)
    set_ax_parameters(ax, xlabel='Residue', ylabel='DFI', loc='upper right')
    plot_figure(fig, ax, figname=system.sysname.upper(), figpath=system.pngdir / f"{tag}.png",)
    # Plotting PDFI
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, pdatas, params)
    set_ax_parameters(ax, xlabel='Residue', ylabel='%DFI', loc='lower right')
    plot_figure(fig, ax, figname=system.sysname.upper(), figpath=system.pngdir / f"p{tag}.png",)


def plot_rmsf(system):
    # Pulling data
    datas, errs = pull_data(system.datdir, 'crmsf_B*')
    xs = [np.arange(len(data)) for data in datas]
    datas = [data*10 for data in datas]
    errs = [err*10 for err in errs]
    params = [{'lw':2} for data in datas]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_errorbar(ax, xs, datas, errs, params, alpha=0.7)
    set_ax_parameters(ax, xlabel='Residue', ylabel='RMSF (Angstrom)')
    plot_figure(fig, ax, figname=system.sysname.upper(), figpath='png/rmsf.png',)


def plot_rmsd(system):
    # Pulling data
    files = io.pull_files(system.mddir, 'rmsd*npy')
    datas = [np.load(file) for file in files]
    labels = [file.split('/')[-3] for file in files]
    xs = [data[0]*1e-3 for data in datas]
    ys = [data[1]*10 for data in datas]
    params = [{'lw':2, 'label':label} for label in labels]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, ys, params)
    set_ax_parameters(ax, xlabel='Time (ns)', ylabel='RMSD (Angstrom)')
    plot_figure(fig, ax, figname=system.sysname.upper() , figpath=system.pngdir / 'rmsd.png',)


def plot_dci(system):
    # Pulling data
    datas, errs = pull_data(system.datdir, 'pertmat*')
    param = {'lw':2}
    datas = [data for data in datas]
    data = datas[0]
    data = mdm.dci(data)
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 12))
    make_heatmap(ax, data, cmap='bwr', interpolation=None, vmin=0, vmax=2)
    set_ax_parameters(ax, xlabel='Residue', ylabel='Residue')
    plot_figure(fig, ax, figname='DCI', figpath='png/dci.png',)


def plot_asym(system):
    # Pulling data
    datas, errs = pull_data(system.datdir, 'asym*')
    param = {'lw':2}
    datas = [data for data in datas]
    data = datas[0]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 12))
    make_heatmap(ax, data, cmap='bwr', interpolation=None, vmin=-1, vmax=1)
    set_ax_parameters(ax, xlabel='Residue', ylabel='Residue')
    plot_figure(fig, ax, figname='DCI asymmetry', figpath='png/asym.png',)


def make_pdb(system, label, factor=None):
    data = np.load(os.path.join(system.datdir, f'{label}_av.npy'))
    err = np.load(os.path.join(system.datdir, f'{label}_err.npy'))
    if factor:
        data *= factor
        err *= factor
    data_pdb = os.path.join(system.pngdir, f'{label}.pdb')
    err_pdb = os.path.join(system.pngdir, f'{label}_err.pdb')
    set_bfactors_by_residue(system.inpdb, data, data_pdb)
    set_bfactors_by_residue(system.inpdb, err, err_pdb)


def make_enm_pdb(system, label, factor=None):
    data = np.load(os.path.join(system.datdir, f'{label}_enm.npy'))
    if factor:
        data *= factor
    data_pdb = os.path.join(system.pngdir, f'enm_{label}.pdb')
    set_bfactors_by_residue(system.inpdb, data, data_pdb)


def make_delta_pdb(system_1, system_2, label, out_name, filter=True, factor=None):
    logger.info('Making Delta PDB')
    data_1 = np.load(os.path.join(system_1.datdir, f'{label}_av.npy'))
    err_1 = np.load(os.path.join(system_1.datdir, f'{label}_err.npy'))
    data_2 = np.load(os.path.join(system_2.datdir, f'{label}_av.npy'))
    err_2 = np.load(os.path.join(system_2.datdir, f'{label}_err.npy'))  
    if factor:
        data_1 *= factor
        err_1 *= factor
        data_2 *= factor
        err_2 *= factor
    data = data_1 - data_2
    err = np.sqrt(err_1**2 + err_2**2)
    if filter:
        mask = np.abs(data) < 2.0 * err
        data[mask] = 0
    data_pdb = os.path.join('systems', 'pdb', out_name + '.pdb')
    err_pdb = os.path.join('systems', 'pdb', out_name + '_err.pdb')
    set_bfactors_by_residue(system_1.inpdb, data, data_pdb)
    set_bfactors_by_residue(system_1.inpdb, err, err_pdb) 
    logger.info('Saved Delta PDB to %s', data_pdb)


def rmsf_pdb(system):
    logger.info(f'Making RMSF PDB')
    make_cg_pdb(system, 'rmsf', factor=10)


def dfi_pdb(system):
    logger.info(f'Making DFI PDB')
    make_pdb(system, 'dfi')


def dci_pdbs(system):
    logger.info(f'Making DCI PDB')
    label = f'gdci'
    make_pdb(system, label)


def runs_metric(system, metric):
    files = io.pull_files(system.mddir, metric)
    files = [f for f in files if '.npy' in f]
    datas = [np.load(file) for file in files]
    xs = [np.arange(len(data)) for data in datas]
    datas = [data for data in datas]
    params = [{'lw':2, 'label':fname} for data, fname in zip(datas, files)]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, datas, params)
    set_ax_parameters(ax, xlabel='Residue', ylabel='RMSF (Angstrom)')
    plot_figure(fig, ax, figname=system.sysname.upper(), figpath='png/metric.png',)


def plot_contact_map(inpdb):
    fname = os.path.basename(inpdb).replace('.pdb', '')
    atoms = io.pdb2atomlist(inpdb)
    residues = atoms.residues
    nres = len(residues)
    contact_map = np.zeros((nres, nres))
    for i in range(nres):
        for j in range(nres):
            pos_i = np.array(residues[i].vecs)
            pos_j = np.array(residues[j].vecs)
            pos_ij = np.average(pos_i, axis=0) - np.average(pos_j, axis=0)
            dist = np.linalg.norm(pos_ij)
            contact_map[i, j] = dist
    vmax = 0.8 * np.average(contact_map)
    fig, ax = init_figure(grid=(1, 1), axsize=(8, 8))
    make_heatmap(ax, contact_map, cmap='Greys', interpolation=None, vmin=0, vmax=vmax)
    set_ax_parameters(ax, xlabel='Residue', ylabel='Residue')
    plot_figure(fig, ax, figname='Contact Map', figpath=f'png/{fname}_contact_map.png',)


def plot_ccf(i, j, sysname, dt=10, fbase='ccf', key='vv', outtag=None):
    files = io.pull_files(f'data/{sysname}', f'{fbase}_{key}_av.npy')
    files = [f for f in files if '_av' in f]
    datas = [np.load(file) for file in files]
    datas = [data[i, j, :] for data in datas]
    xs = [np.arange(data.shape[0])*dt for data in datas]
    labels = [file.split('/')[-1].replace('.npy', '') for file in files]
    params = [{'lw':2, 'label':label} for label in labels]
    # Plottingjj
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, datas, params)
    set_ax_parameters(ax, xlabel='Time (fs)', ylabel='CCF')
    outdir = Path('png') / sysname / key
    outdir.mkdir(parents=True, exist_ok=True)
    outtag = outtag if outtag else f'{fbase}'
    plot_figure(fig, ax, figname=f'{key.upper()} CCF {i}_{j}', figpath=outdir / f'{outtag}_{i}_{j}.png')


def plot_fft_ccf(i, j, sysname, fbase='ccf', key='vv'):
    files = io.pull_files(f'data/{sysname}', f'{fbase}_{key}_av_fftn*.npy')
    files = [f for f in files if '_av' in f]
    datas = [np.load(file) for file in files]
    ys = [np.abs(data[i, j, 1:]) for data in datas]
    xs = [np.angle(data[i, j, 1:]) for data in datas]
    labels = [file.split('/')[-1].replace('.npy', '') for file in files]
    params = [{'lw':2, 'label':label} for label in labels]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, ys, params)
    set_ax_parameters(ax, xlabel='Freq (1/fs)', ylabel='CCF')
    outdir = Path('png') / sysname
    outdir.mkdir(parents=True, exist_ok=True)
    plot_figure(fig, ax, figname=f'{key.upper()} CCF {i}_{j}', figpath=outdir / f'{fbase}_{i}_{j}.png')


def plot_test(sysname, fbase='pertmat', key='vv'):
    infile = Path("data") / sysname / f"{fbase}_{key}_av.npy"
    fname = os.path.basename(infile).replace('.npy', '')
    ccf = np.load(infile)
    data = np.average(ccf, axis=-1)
    ys = [np.average(data, axis=1)]
    xs = [np.arange(len(y)) + 26 for y in ys]
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, ys)
    set_ax_parameters(ax, xlabel='Residue', ylabel='PM_av', loc='upper right')
    plot_figure(fig, ax, figname=sysname.upper(), figpath=Path('png') / f"{sysname}_pm_av.png",)


def resid_to_index(pdb, resids):
    u = mda.Universe(pdb)
    cas = u.select_atoms("name CA")
    all_resids = np.array(cas.resids)
    all_ids = np.arange(len(all_resids))
    ids = all_ids[np.isin(all_resids, resids)]
    return ids


def plot_allosteric_control(sysname, **kwargs):
    for pert in allosteric_ids:
        for resp in active_ids:
            plot_ccf(pert, resp, sysname, outtag='active', **kwargs)
        for resp in control_ids:
            plot_ccf(pert, resp, sysname, outtag='control', **kwargs)


def plot_cpsd_magnitude(i, j, sysname, dt=200, filename='cpsd_vv_av.npy', outtag=None):
    """
    Plot CPSD magnitude (amplitude) for given indices.
    
    Parameters
    ----------
    i, j : int
        Row and column indices in the CPSD matrix
    sysname : str
        System name
    dt : float
        Time step in femtoseconds (default: 200 fs)
    filename : str
        Data filename (default: 'cpsd_vv_av.npy')
    outtag : str, optional
        Output tag for filename
    """
    files = io.pull_files(f'data/{sysname}', filename)
    if not files:
        logger.warning(f"No files found matching {filename}")
        return
    datas = [np.load(file) for file in files]
    
    # Extract magnitude for the specified indices
    datas_mag = [np.abs(data[i, j, :]) if data.ndim == 3 else np.abs(data[i, j]) for data in datas]
    
    # Handle both 1D and 2D cases
    if isinstance(datas_mag[0], np.ndarray) and datas_mag[0].ndim > 0:
        # Convert frequency index to actual frequency
        # freq = k / (N * dt) where k is the index, N is the length, dt is time step
        # For CPSD: freq in units of 1/fs, then convert to THz (1 THz = 1000 fs^-1)
        n_freq = datas_mag[0].shape[0]
        freq_spacing = 1.0 / (n_freq * dt)  # in 1/fs
        xs = [np.arange(data.shape[0]) * freq_spacing * 1000 for data in datas_mag]  # Convert to THz
    else:
        xs = [[0] for _ in datas_mag]
        datas_mag = [[d] for d in datas_mag]
    
    labels = [file.split('/')[-1].replace('.npy', '') for file in files]
    params = [{'lw':2, 'label':label} for label in labels]
    
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, datas_mag, params)
    set_ax_parameters(ax, xlabel='Frequency (THz)', ylabel='CPSD Magnitude')
    
    # Use filename base (without extension) for output directory
    filebase = Path(filename).stem.replace('_av', '')
    outdir = Path('png') / sysname / filebase / 'magnitude'
    outdir.mkdir(parents=True, exist_ok=True)
    outtag = outtag if outtag else f'{filebase}_magnitude'
    plot_figure(fig, ax, figname=f'{filebase.upper()} CPSD Magnitude {i}_{j}', 
                figpath=outdir / f'{outtag}_{i}_{j}.png')


def plot_cpsd_phase(i, j, sysname, dt=200, filename='cpsd_vv_av.npy', outtag=None):
    """
    Plot CPSD phase for given indices.
    
    Parameters
    ----------
    i, j : int
        Row and column indices in the CPSD matrix
    sysname : str
        System name
    dt : float
        Time step in femtoseconds (default: 200 fs)
    filename : str
        Data filename (default: 'cpsd_vv_av.npy')
    outtag : str, optional
        Output tag for filename
    """
    files = io.pull_files(f'data/{sysname}', filename)
    if not files:
        logger.warning(f"No files found matching {filename}")
        return
    datas = [np.load(file) for file in files]
    
    # Extract phase for the specified indices
    datas_phase = [np.angle(data[i, j, :]) if data.ndim == 3 else np.angle(data[i, j]) for data in datas]
    
    # Handle both 1D and 2D cases
    if isinstance(datas_phase[0], np.ndarray) and datas_phase[0].ndim > 0:
        # Convert frequency index to actual frequency
        # freq = k / (N * dt) where k is the index, N is the length, dt is time step
        # For CPSD: freq in units of 1/fs, then convert to THz (1 THz = 1000 fs^-1)
        n_freq = datas_phase[0].shape[0]
        freq_spacing = 1.0 / (n_freq * dt)  # in 1/fs
        xs = [np.arange(data.shape[0]) * freq_spacing * 1000 for data in datas_phase]  # Convert to THz
    else:
        xs = [[0] for _ in datas_phase]
        datas_phase = [[d] for d in datas_phase]
    
    labels = [file.split('/')[-1].replace('.npy', '') for file in files]
    params = [{'lw':2, 'label':label} for label in labels]
    
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_plot(ax, xs, datas_phase, params)
    set_ax_parameters(ax, xlabel='Frequency (THz)', ylabel='CPSD Phase (radians)')
    
    # Use filename base (without extension) for output directory
    filebase = Path(filename).stem.replace('_av', '')
    outdir = Path('png') / sysname / filebase / 'phase'
    outdir.mkdir(parents=True, exist_ok=True)
    outtag = outtag if outtag else f'{filebase}_phase'
    plot_figure(fig, ax, figname=f'{filebase.upper()} CPSD Phase {i}_{j}', 
                figpath=outdir / f'{outtag}_{i}_{j}.png')


def plot_cpsd_active_control(sysname, allosteric_ids, active_ids, control_ids, 
                            all_sites_ids=None, pdb_file=None, **kwargs):
    """
    Plot CPSD magnitude and phase for allosteric perturbations on active and control sites.
    
    Parameters
    ----------
    sysname : str
        System name
    allosteric_ids : array-like
        Indices of allosteric sites in the full system
    active_ids : array-like
        Indices of active sites in the full system
    control_ids : array-like
        Indices of control sites in the full system
    all_sites_ids : array-like, optional
        Indices of all sites used in CPSD computation (subset)
        If None, will be computed from pdb_file
    pdb_file : str, optional
        Path to PDB file to compute all_sites_ids
    **kwargs : dict
        Additional keyword arguments passed to plot functions (e.g., filename, dt)
    """
    logger.info(f"Plotting CPSD magnitude and phase for {sysname}")
    
    # If all_sites_ids not provided, compute it
    if all_sites_ids is None:
        if pdb_file is None:
            logger.error("Either all_sites_ids or pdb_file must be provided")
            return
        # Import from analysis to get site definitions
        from analysis import allosteric_sites, active_sites, control_sites, all_sites
        all_sites_ids = resid_to_index(pdb_file, all_sites)
    
    # Map site indices to their positions in the all_sites subset
    def map_to_subset(site_ids, all_ids):
        return np.array([np.where(all_ids == idx)[0][0] for idx in site_ids])
    
    allo_subset = map_to_subset(allosteric_ids, all_sites_ids)
    active_subset = map_to_subset(active_ids, all_sites_ids)
    control_subset = map_to_subset(control_ids, all_sites_ids)
    
    logger.info(f"Mapped {len(allosteric_ids)} allosteric, {len(active_ids)} active, "
                f"{len(control_ids)} control sites to subset indices")
    
    # Plot magnitude
    for pert in allo_subset:
        for resp in active_subset:
            plot_cpsd_magnitude(pert, resp, sysname, outtag='active_magnitude', **kwargs)
        for resp in control_subset:
            plot_cpsd_magnitude(pert, resp, sysname, outtag='control_magnitude', **kwargs)
    
    # Plot phase
    for pert in allo_subset:
        for resp in active_subset:
            plot_cpsd_phase(pert, resp, sysname, outtag='active_phase', **kwargs)
        for resp in control_subset:
            plot_cpsd_phase(pert, resp, sysname, outtag='control_phase', **kwargs)
    
    logger.info("CPSD plotting complete")


if __name__ == '__main__':
    pdb_id = '1btl'
    pdb_file = f'systems/{pdb_id}.pdb'
    
    # Site definitions (residue IDs)
    allo_1 = [44, 203, 232, 249, 262, 286]
    allo_fta = [244]
    allosteric_resids = allo_1 + allo_fta
    active_1 = [70, 73, 130, 132, 166, 170, 234]
    active_2 = [73, 105, 166, 229, 234, 244, 275]
    active_resids = sorted(list(set(active_1 + active_2)))
    control_resids = [55, 80, 99, 120, 150, 180, 200, 222, 256]
    all_resids = allosteric_resids + active_resids + control_resids
    
    # Convert to indices
    allosteric_ids = resid_to_index(pdb_file, allosteric_resids)
    active_ids = resid_to_index(pdb_file, active_resids)
    control_ids = resid_to_index(pdb_file, control_resids)
    all_sites_ids = resid_to_index(pdb_file, all_resids)
    
    # PLOTS 
    # plot_contact_map('systems/1btl.pdb')
    # plot_test('1btl_nve_nikhil', fbase='pertmat', key='pv')
    # plot_allosteric_control('1btl_nve', fbase='pmat', key='pv', dt=20)
    
    # # Apply running window average to CPSD files
    # from analysis import apply_running_average_to_files
    # apply_running_average_to_files('data/1btl_nve', pattern="cpsd*.npy", window_size=100)
    
    # CPSD plots - magnitude and phase (dt in femtoseconds)
    # Now using filename directly instead of fbase and key
    plot_cpsd_active_control('1btl_nve', allosteric_ids, active_ids, control_ids,
                            all_sites_ids=all_sites_ids, 
                            filename='cpsd_vv_av_ws100.npy', dt=200)
