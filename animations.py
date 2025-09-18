import logging
import os
from pathlib import Path
import sys
import numpy as np
from reforge import io, mdm
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.rfgmath.rpymath import gfft_conv
from reforge.utils import logger
import reforge.plotting as rfplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 


def _grid_labels(mdsys):
    atoms = io.pdb2atomlist(mdsys.solupdb)
    backbone_anames = ["BB"]
    bb = atoms.mask(backbone_anames, mode='name')
    bb.renum() # Renumber atids form 0, needed to mask numpy arrays
    groups = bb.segments.atids # mask for the arrays
    labels = [segids[0] for segids in bb.segments.segids]
    line_positions = [group[0] for group in groups]
    line_positions.append(groups[-1][-1])
    label_positions = [group[len(group)//2] for group in groups]
    return line_positions, label_positions, labels
    

def set_hm_parameters(ax, xlabel=None, ylabel=None, axtitle=None):
    """
    ax - matplotlib ax object
    """
    # Set axis labels and title with larger font sizes
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(axtitle, fontsize=16)
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5, width=1.5)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False)
    # line_positions, label_positions, labels = _grid_labels(mdsys)
    # max_line_pos = max(line_positions)
    # # Add grid
    # for line_pos, label_pos, label in zip(line_positions, label_positions, labels):
    #     ax.axvline(x=line_pos, color='k', linestyle=':', label=None)
    #     ax.axhline(y=line_pos, color='k', linestyle=':', label=None)
    #     ax.text(label_pos/max_line_pos-0.008, 1.01, label, transform=ax.transAxes, 
    #         rotation=90, fontsize=14) 
    #     ax.text(1.01, 0.992-label_pos/max_line_pos, label, transform=ax.transAxes, 
    #         fontsize=14) 
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    legend = ax.legend(fontsize=14, frameon=False)
   # Autoscale the view to the data
    ax.relim()         # Recalculate limits based on current artists
    ax.autoscale_view()  # Update the view to the recalculated limits
    ax.margins(0) # Remove padding around the data


def animate_graph(fig, ax, lines, datas, outfile="data/ani1d.mp4", dt=0.2):
    print("Working on animation", file=sys.stderr)
    def update(frame):
        for line, data in zip(lines, datas):
            line.set_ydata(data[frame])  # Update y-values for each frame
            ax.set_title(f"Time {dt * frame:.2f}, ns")
        return tuple(lines)
    ani = animation.FuncAnimation(
        fig, update, frames=len(datas[0]), interval=50, blit=False
    )
    ani.save(outfile, writer="ffmpeg")  # Save as mp4
    print("Done!", file=sys.stderr)


def make_graph_animation(sysdir, sysnames):
    print("Plotting", file=sys.stderr)
    datas = []
    for n, sysname in enumerate(sysnames):
        system = gmxSystem(sysdir, sysname)
        infile = os.path.join(system.datdir, f"corr_pp_slow.npy")
        data = make_2d_data(infile, nframes=2000)
        np.save(f"data/arr_{n}.npy", data)
        datas.append(data)
    # datas = [np.load('data/arr_0.npy'), np.load('data/arr_1.npy'),]
    averages = [np.average(data[0]) for data in datas]
    av = np.average(averages)
    datas = [data / av for data in datas]
    outfile = os.path.join("png", f'pp_{"_".join(sysnames)}.mp4')
    fig, ax, lines = make_plot_t_2d(datas, sysnames, outfile="png/test.png")
    animate_1d(fig, ax, lines, datas, outfile, dt=0.04)


############################
## Heatmap animations
############################

def make_hm_data(infile, nframes=1000):
    print(f"Processing {infile}", file=sys.stderr)
    matrices = []
    mat_t = np.load(infile) # [:, 1412:1466, :]
    print(mat_t.shape)
    if nframes > mat_t.shape[2]:  # Plot only the valid part
        nframes = mat_t.shape[2]
    print('Checkpoint')
    pertmat_t = mdm.td_perturbation_matrix(mat_t)  
    print(pertmat_t.shape)
    # resp_t = response_force(mat_t)
    for i in range(0, nframes):
        resp = pertmat_t[:, :, i]
        # resp = response(resp)
        # resp = mdm.dci(resp)
        matrices.append(resp)
    print("Finished computing matrices", file=sys.stderr)
    return matrices


def sliding_window_average(data, window_size=5):
    """
    Apply sliding window average along the last axis of the array.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input array of any dimension
    window_size : int
        Size of the sliding window
    
    Returns
    -------
    numpy.ndarray
        Smoothed array of same shape as input
    """
    window = np.ones(window_size) / window_size  # simple box window
    # For a more sophisticated window, you could use:
    # window = np.hanning(window_size)  # Hanning window
    # window = np.hamming(window_size)  # Hamming window
    # window = np.blackman(window_size)  # Blackman window
    
    # Reshape to 2D if needed
    orig_shape = data.shape
    if data.ndim > 2:
        data_2d = data.reshape(-1, data.shape[-1])
    else:
        data_2d = data
    
    # Apply convolution along last axis
    smoothed = np.array([np.convolve(row, window, mode='same') for row in data_2d])
    
    # Restore original shape if needed
    if data.ndim > 2:
        smoothed = smoothed.reshape(orig_shape)
    
    return smoothed


def plot_hm(data, cmap='bwr'):
    """
    Plot heatmap with the given colormap.
    Available cmaps: 'bwr' (blue-white-red), 'viridis', 'rainbow', 'RdBu_r' (reversed red-blue),
                    'plasma', 'seismic', 'coolwarm'
    """
    logger.info("Plotting heatmap")
    fig, ax = rfplot.init_figure(grid=(1, 1), axsize=(6, 6))
    scale = np.average(np.abs(data))
    factor = 4
    img = rfplot.make_heatmap(ax, data, cmap=cmap, interpolation=None, vmin=-factor*scale, vmax=factor*scale)
    set_hm_parameters(ax, xlabel='Perturbed Residue', ylabel='Coupled Residue')
    return fig, img  


def animate_hm(fig, img, data, title, dt=0.02, outfile="data/hm_ani.mp4"):
    logger.info("Working on animation")
    def update(frame):
        img.set_array(data[:, :, frame])
        # img.axes.set_title(f"Time {dt * frame:.2f} ns")
        fig.suptitle(f"{title} :{dt * frame:.2f} fs", fontsize=16)
        return img
    ani = animation.FuncAnimation(
        fig, update, frames=data.shape[-1], interval=50, blit=False
    )
    ani.save(outfile, writer="ffmpeg")  # Save as mp4
    logger.info("Done. Saved to %s", outfile)


def make_hm_animation(sysname, key):
    infile = Path(sysdir) / "data" / f"ccf_{key}_av.npy"
    ccf = np.load(infile)
    fname = os.path.basename(infile).replace('.npy', '')
    data = ccf - ccf[:, :, 400][..., None]
    pertmat = mdm.td_perturbation_matrix(data)
    data = pertmat # - pertmat[:, :, 400][..., None]
    # data = sliding_window_average(data, window_size=10)
    # outfile = mdsys.pngdir / 'pp_corr.mp4'
    fig, img = plot_hm(data[:, :, 1], cmap='bwr')
    fig.savefig(f"png/{fname}.png")
    title = f'{key.upper()} CCF'
    outfile = f"png/{sysname}_{fname}.mp4"
    animate_hm(fig, img, data[:, :, :400], title, dt=20, outfile=outfile)
  

if __name__ == '__main__':
    sysdir = 'systems/blac_nve' 
    sysnames =['blac_nve']
    for sysname in sysnames:
        mdsys = MDSystem(sysdir, sysname)
        alist = ['pv', ]
        for key in alist:
            make_hm_animation(sysname, key)
