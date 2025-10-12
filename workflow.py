import os
import shutil
import sys
from pathlib import Path
import numpy as np
import cupy as cp
import multiprocessing as mp
import MDAnalysis as mda
import openmm as mm
from openmm import app, Platform, unit
from reforge import io, mdm
from reforge.martini import martini_openmm
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.mdsystem.mmmd import MmSystem, MmRun, MmReporter, convert_trajectories
from reforge.mdsystem.gmxmd import GmxSystem
from reforge.utils import clean_dir, get_logger
import plots
from enm_toy_md import setup_enm

logger = get_logger(__name__)

# Global settings
INPDB = '1btl.pdb'
# Production parameters
TEMPERATURE = 300 * unit.kelvin  # for equilibraion
GAMMA = 1 / unit.picosecond
PRESSURE = 1 * unit.bar
TOTAL_TIME = 100 * unit.picoseconds
TSTEP = 20 * unit.femtoseconds
TOTAL_STEPS = int(TOTAL_TIME / TSTEP)
# Report intervals
TRJ_NOUT = 1              # Trajectory   
LOG_NOUT = 10000            # Log file   
CHK_NOUT = 100000           # Checkpoint
OUT_SELECTION = "name BB"
TRJEXT = 'trr'              # trr saves positions, velocities, forces
SELECTION = "name BB" 


def workflow(sysdir, sysname, runname):
    md_nve(sysdir, sysname, runname)
    trjconv(sysdir, sysname, runname)
    save_pos_vel_to_numpy(sysdir, sysname, runname, selection=SELECTION, dtype=np.float32)
    tdlrt_analysis(sysdir, sysname, runname)

###########################################################
### Setup EMU ###
###########################################################

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
        idx = i + 98
        outdir = newsys_dir / f"sample_{idx:03d}"
        outdir.mkdir(parents=True, exist_ok=True)
        outpdb = outdir / "sample.pdb"
        with mda.Writer(outpdb, u.atoms.n_atoms) as W:
            W.write(u.atoms)
        logger.info(f"Saved initial structure {i} to {outpdb}")


###########################################################
### Setup AA ###
###########################################################

def setup_aa(sysdir, sysname):
    mdsys = MmSystem(sysdir, sysname)
    inpdb = mdsys.sysdir / INPDB
    mdsys.prepare_files()
    mdsys.clean_pdb(inpdb, add_missing_atoms=True, add_hydrogens=True)
    pdb = app.PDBFile(str(mdsys.inpdb))
    model = app.Modeller(pdb.topology, pdb.positions)
    forcefield = app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    logger.info("Adding solvent and ions")
    model.addSolvent(forcefield, 
        model='tip3p', 
        boxShape='dodecahedron', #  ‘cube’, ‘dodecahedron’, and ‘octahedron’
        padding=1.0 * unit.nanometer,
        ionicStrength=0.1 * unit.molar,
        positiveIon='Na+',
        negativeIon='Cl-')
    with open(mdsys.syspdb, "w", encoding="utf-8") as file:
        app.PDBFile.writeFile(model.topology, model.positions, file, keepIds=True)    
    logger.info("Saved solvated system to %s", mdsys.syspdb)
    # Build a system WITHOUT any motion remover/barostat/thermostat. Add them later as needed.
    logger.info("Generating topology...")
    system = forcefield.createSystem(
        model.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        removeCMMotion=False,     # important for strict NVE
        ewaldErrorTolerance=1e-5
    )
    _save_system_to_xml(system, mdsys.sysxml)
    
###########################################################
### Setup Martini ###
###########################################################

def setup_martini_gmx(sysdir, sysname):
    mdsys = GmxSystem(sysdir, sysname)
    inpdb = mdsys.sysdir / INPDB
    mdsys.prepare_files(pour_martini=True) # be careful it can overwrite later files
    # mdsys.clean_pdb_mm(inpdb, add_missing_atoms=True, add_hydrogens=True, pH=7.0) 
    mdsys.clean_pdb_gmx(inpdb, clinput="8\n 7\n", ignh="no", renum="yes") 
    mdsys.split_chains()
    mdsys.martinize_proteins_go(go_eps=12.0, go_low=0.3, go_up=1.1, from_ff='charmm', append=False) # Martini + Go-network FF
    # mdsys.martinize_proteins_en(ef=400, el=0.3, eu=0.9, from_ff='charmm', p="none", append=False)  
    mdsys.make_cg_topology() # CG topology. Returns mdsys.systop ("mdsys.top") file
    mdsys.make_cg_structure() # CG structure. Returns mdsys.solupdb ("solute.pdb") file
    mdsys.make_box(d="1.0", bt="dodecahedron")
    solvent = mdsys.root / "water.gro"
    mdsys.solvate(cp=mdsys.solupdb, cs=solvent, radius="0.17") # all kwargs go to gmx solvate command
    mdsys.add_bulk_ions(conc=0.10, pname="NA", nname="CL")


def setup_martini(sysdir, sysname):
    mdsys = MmSystem(sysdir, sysname)
    setup_martini_gmx(sysdir, sysname)
    # 1.5. GMX -> OpenMM
    top_file = str(mdsys.systop)
    conf = app.GromacsGroFile(str(mdsys.sysgro))
    box_vectors = conf.getPeriodicBoxVectors()
    top = martini_openmm.MartiniTopFile(top_file, periodicBoxVectors=box_vectors, epsilon_r=15.0)
    system = top.create_system(nonbonded_cutoff=1.1*unit.nanometer)
    pdb = app.PDBFile(str(mdsys.syspdb))
    _save_system_to_xml(system, mdsys.sysxml)

###########################################################
### MD ###
###########################################################

def md_nve(sysdir, sysname, runname):
    mdsys = MmSystem(sysdir, sysname)
    mdrun = MmRun(sysdir, sysname, runname)
    mdrun.rundir.mkdir(parents=True, exist_ok=True)
    logger.info(f"WDIR: %s", mdrun.rundir)
    # Prep
    pdb = app.PDBFile(str(mdsys.syspdb))
    system = _load_system_from_xml(mdsys.sysxml)
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, GAMMA, 0.5*TSTEP) # NVT integrator for equilibration
    simulation = app.Simulation(pdb.topology, system, integrator) 
    # --- Initialize state, minimize, equilibrate ---
    logger.info("Minimizing energy...")
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=1000)  
    logger.info("Equilibrating...")
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)
    simulation.step(10000)  # equilibrate 
    # --- Run NVE (need to change the integrator and reset simulation) ---
    logger.info("Running NVE production...")
    integrator = mm.VerletIntegrator(TSTEP)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setState(state)
    logger.info(f'Saving reference PDB with selection: {OUT_SELECTION}')
    mda.Universe(mdsys.syspdb).select_atoms(OUT_SELECTION).write(mdrun.rundir / "md.pdb") # SAVE PDB FOR THE SELECTION
    reporters = _get_reporters(mdrun, append=False, prefix="md")
    simulation.reporters.extend(reporters)
    simulation.step(TOTAL_STEPS)  
    logger.info("Done!")


def trjconv(sysdir, sysname, runname):
    system = MDSystem(sysdir, sysname)
    mdrun = MDRun(sysdir, sysname, runname)
    logger.info(f"WDIR: %s", mdrun.rundir)
    # INPUT
    top = mdrun.rundir / "md.pdb"
    # top = mdrun.syspdb  # use original topology if needed
    traj = mdrun.rundir / f"md.{TRJEXT}"
    ext_trajs = sorted([f for f in mdrun.rundir.glob(f"md_*.{TRJEXT}")])
    trajs = [traj] + ext_trajs
    logger.info(f'Input trajectory files: {trajs}')
    # CONVERT
    out_top = mdrun.rundir / "topology.pdb"
    out_traj = mdrun.rundir / f"samples.{TRJEXT}"
    convert_trajectories(top, trajs, out_top, out_traj, selection=OUT_SELECTION, step=1)
    logger.info("Done!")


def save_pos_vel_to_numpy(sysdir, sysname, runname, selection=SELECTION, dtype=np.float32):
    mdrun = MDRun(sysdir, sysname, runname)
    traj = mdrun.rundir / "samples.trr"
    top = mdrun.rundir / "topology.pdb" 
    outdir = Path(mdrun.rundir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info("Reading trajectory %s with topology %s", traj, top)
    u = mda.Universe(str(top), str(traj))
    ag = u.select_atoms(selection)
    n_atoms = ag.n_atoms
    if n_atoms == 0:
        logger.warning("Selection '%s' matched no atoms; nothing to save", selection)
        return
    positions = []
    velocities = []
    for ts in u.trajectory:
        # copy to avoid referencing the underlying arrays
        positions.append(ag.positions.copy())
        vel = getattr(ag, 'velocities', None)
        if vel is None:
            # try frame attribute
            vel = getattr(ts, 'velocities', None)
        if vel is None:
            # fill with zeros if velocities are not present
            velocities.append(np.zeros_like(ag.positions, dtype=dtype))
        else:
            velocities.append(vel.copy())
    # Stack into arrays: shape (n_atoms, 3, n_frames)
    pos_arr = np.stack(positions, axis=2).astype(dtype)
    vel_arr = np.stack(velocities, axis=2).astype(dtype)
    n_frames = pos_arr.shape[2]
    # Reshape to (n_atoms*3, n_frames) ordering: atom0_x, atom0_y, atom0_z, atom1_x, ...
    pos_flat = pos_arr.reshape(n_atoms * 3, n_frames)
    vel_flat = vel_arr.reshape(n_atoms * 3, n_frames)
    pos_file = outdir / 'positions.npy'
    vel_file = outdir / 'velocities.npy'
    np.save(pos_file, pos_flat)
    np.save(vel_file, vel_flat)
    logger.info('Saved positions (%s) and velocities (%s) for %d atoms and %d frames', pos_file, vel_file, n_atoms, n_frames)

###########################################################
### TDLRT ###
###########################################################

def tdlrt_analysis(sysdir, sysname, runname):
    mdrun = MDRun(sysdir, sysname, runname)
    mdrun.prepare_files()
    ps_path = str(mdrun.rundir / f"positions.npy")
    vs_path = str(mdrun.rundir / f"velocities.npy")
    if (Path(ps_path).exists() and Path(vs_path).exists()):
        logger.info("Loading positions and velocities from %s", mdrun.rundir)
        ps = np.load(ps_path)
        vs = np.load(vs_path)
    else:
        traj = str(mdrun.rundir / f"samples.trr")
        top = str(mdrun.rundir / "topology.pdb")
        u = mda.Universe(top, traj)
        ps = io.read_positions(u, u.atoms) # (n_atoms*3, nframes)
        vs = io.read_velocities(u, u.atoms) # (n_atoms*3, nframes)
    # ps = ps - ps[:, 0][..., None]
    # ps -= ps.mean(axis=1)[..., None]
    # CCF calculations
    adict = {'vv': (vs, vs), } #  adict = {'pv': (ps, vs)}
    for key, item in adict.items(): # DT = TSTEP * NOUT
        v1, v2 = item
        corr = mdm.ccf(v1, v2, ntmax=200, n=1, mode='gpu', center=False, dtype=np.float32, buffer_c=0.8) # falls back on cpu if no cuda
        corr_file = mdrun.lrtdir / f'ccfs_{key}.npy'
        np.save(corr_file, corr)    
        logger.info("Saved CCFs to %s", corr_file)


def ffts(dtype=None, ntmax=None, center=False):
    logger.info("Computing FFTs on GPU.")
    infile = 'data/1btl_nve_nikhil/pertmat_vv_av.npy'
    data = np.load(infile)
    if dtype is None:
        dtype = data.dtype
    nt = data.shape[-1]
    nx = data.shape[0]
    ny = data.shape[1]
    if ntmax is None or ntmax > (nt + 1) // 2:
        ntmax = (nt + 1) // 2
    if center:
        data = data - np.mean(data, axis=-1, keepdims=True)
    data = cp.asarray(data, dtype=dtype)
    data_f = cp.fft.fft(data, n=2 * nt, axis=-1)
    np.save(infile.replace('.npy', f'_fftn{ntmax}.npy'), cp.asnumpy(data_f[:, :, :ntmax]))
    logger.info("Saved FFTs to %s", infile.replace('.npy', f'_fftn{ntmax}.npy'))


def read_nikhils_files():
    dpath = Path("/scratch/nrames19/Time-Dependent/BioEmuRuns/1BTL-RS2")
    sysdir = Path("systems/1btl_nve_nikhil")
    p_files = sorted(list(dpath.glob("*aligned_displacements.npy")))
    v_files = sorted(list(dpath.glob("*aligned_velocities.npy")))
    for pfile, vfile in zip(p_files, v_files):
        pbase = pfile.name.split("_aligned_")[0]
        vbase = vfile.name.split("_aligned_")[0]
        if pbase != vbase:
            logger.warning("Base names do not match: %s vs %s", pbase, vbase)
            continue
        base = pbase
        outdir = Path(sysdir) / base / "mdruns" / "mdrun"
        outdir.mkdir(parents=True, exist_ok=True)
        logger.info("Reading %s and %s", pfile, vfile)
        ps = np.load(pfile).astype(np.float32)
        vs = np.load(vfile).astype(np.float32)
        ntmax = min(ps.shape[0], vs.shape[0])
        tstep = 1 # frames
        ps = ps[:ntmax:tstep, ...]
        vs = vs[:ntmax:tstep, ...]
        # logger.info("Shapes: %s and %s", ps.shape, vs.shape)
        psr = ps.transpose(1, 2, 0).reshape(-1, ps.shape[0])
        vsr = vs.transpose(1, 2, 0).reshape(-1, vs.shape[0])
        logger.info("Updated shapes: %s and %s", psr.shape, vsr.shape)
        pos_file = outdir / 'positions.npy'
        vel_file = outdir / 'velocities.npy'
        np.save(pos_file, psr)
        np.save(vel_file, vsr)
        logger.info("Saved to %s and %s", pos_file, vel_file)


def pca_data():
    pass


##############################################################################################
### Private funcs ############################################################################
##############################################################################################

def _save_system_to_xml(system, filename):
    with open(str(filename), "w", encoding="utf-8") as file:
        file.write(mm.XmlSerializer.serialize(system))
    logger.info(f"Saved system to {filename}")


def _load_system_from_xml(filename):
    with open(str(filename), 'r') as file:
        system = mm.XmlSerializer.deserialize(file.read())
    logger.info(f"Loaded system from {filename}")
    return system


def _get_reporters(mdrun, append=False, prefix="md"):
    """Get reporters for MD simulation using custom MmReporter for velocities"""
    mdrun.rundir.mkdir(parents=True, exist_ok=True)
    # Log reporter (file)
    log_reporter = app.StateDataReporter(
        str(mdrun.rundir / f"{prefix}.log"), 
        LOG_NOUT, step=True, time=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True, append=append)
    # Error reporter (stderr)
    err_reporter = app.StateDataReporter(
        sys.stderr, LOG_NOUT, time=True, step=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True, append=append)
    # Custom trajectory reporter with velocities using MmReporter
    logger.info(f'Setting up trajectory reporter with selection: {OUT_SELECTION}')
    traj_reporter = MmReporter(str(mdrun.rundir / f"{prefix}.{TRJEXT}"), 
        reportInterval=TRJ_NOUT, selection=OUT_SELECTION)
    # State/checkpoint reporter
    state_reporter = app.CheckpointReporter(str(mdrun.rundir / f"{prefix}.xml"), CHK_NOUT, writeState=True)
    return log_reporter, err_reporter, traj_reporter, state_reporter


def _add_bb_restraints(system, pdb, bb_aname='CA'):
    restraint = mm.CustomExternalForce('bb_fc*periodicdistance(x, y, z, x0, y0, z0)^2')
    restraint.setName('BackboneRestraint')
    restraint.addGlobalParameter('bb_fc', 1000.0*unit.kilojoules_per_mole/unit.nanometer)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')
    system.addForce(restraint)
    for atom in pdb.topology.atoms():
        if atom.name == bb_aname:
            restraint.addParticle(atom.index, pdb.positions[atom.index])


def _pdb_to_seq(pdb):
    u = mda.Universe(pdb)
    protein = u.select_atoms("protein")
    seq = "".join(res.resname for res in protein.residues)  # three-letter codes
    seq_oneletter = "".join(mda.lib.util.convert_aa_code(res.resname) for res in protein.residues)
    return seq_oneletter


if __name__ == "__main__":
    from reforge.cli import run_command
    run_command()