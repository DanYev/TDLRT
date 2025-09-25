import os
from pathlib import Path
import sys
import MDAnalysis as mda
import numpy as np
import pandas as pd
from reforge import io, mdm
from reforge.mdsystem import gmxmd
from reforge.plotting import *
from reforge.utils import logger


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


def plot_ccf(i, j, sysname, fbase='ccf', key='vv', outtag=None):
    files = io.pull_files(f'data/{sysname}', f'{fbase}_{key}_av.npy')
    files = [f for f in files if '_av' in f]
    datas = [np.load(file) for file in files]
    datas = [data[i, j, :] for data in datas]
    xs = [np.arange(data.shape[0])*10 for data in datas]
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
    # Plottingjj
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


def plot_allosteric_control(sysname, fbase='pertmat', key='pv'):
    for pert in allosteric_ids:
        for resp in active_ids:
            plot_ccf(pert, resp, sysname, fbase=fbase, key=key, outtag='active')
        for resp in control_ids:
            plot_ccf(pert, resp, sysname, fbase=fbase, key=key, outtag='control')


if __name__ == '__main__':
    pdb_id = '1btl'
    allosteric_sites = [44, 203, 232, 249, 262, 286]
    active_sites = [70, 130, 160]
    control_sites = [55, 150, 226, 256]
    allosteric_ids = resid_to_index(f'systems/{pdb_id}.pdb', allosteric_sites)
    active_ids = resid_to_index(f'systems/{pdb_id}.pdb', active_sites)
    control_ids = resid_to_index(f'systems/{pdb_id}.pdb', control_sites)
    # PLOTS 
    # plot_contact_map('systems/1btl.pdb')
    # plot_test('1btl_nve_nikhil', fbase='pertmat', key='pv')
    plot_allosteric_control('1btl_nve_nikhil', fbase='pertmat', key='vv')
