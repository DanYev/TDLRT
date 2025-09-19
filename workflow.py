import inspect
import logging
import os
import shutil
import sys
from pathlib import Path
import warnings
import numpy as np
import multiprocessing as mp
import MDAnalysis as mda
from MDAnalysis.transformations.fit import fit_rot_trans
import openmm as mm
from openmm import app, Platform, unit
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.mdsystem.mmmd import MmSystem, MmRun, MmReporter
from reforge.utils import clean_dir, logger
from reforge import io, mdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global settings
INPDB = '1btl.pdb'
# Production parameters
TEMPERATURE = 300 * unit.kelvin  # for equilibraion
GAMMA = 1 / unit.picosecond
PRESSURE = 1 * unit.bar
TOTAL_TIME = 200 * unit.picoseconds
TSTEP = 2 * unit.femtoseconds
NOUT = 10 # save every NOUT steps
OUT_SELECTION = "name CA" 
SELECTION = "name CA" 


def workflow(sysdir, sysname, runname):
    # md_nve(sysdir, sysname, runname)
    # trjconv(sysdir, sysname, runname)
    # save_pos_vel_to_numpy(sysdir, sysname, runname, selection=SELECTION, dtype=np.float32)
    tdlrt_analysis(sysdir, sysname, runname)


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


def setup(sysdir, sysname):
    mdsys = MmSystem(sysdir, sysname)
    # inpdb = mdsys.sysdir / INPDB
    inpdb = mdsys.root / 'sample.pdb'
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


def md_nve(sysdir, sysname, runname):
    # --- Inputs ---
    mdsys = MmSystem(sysdir, sysname)
    mdrun = MmRun(sysdir, sysname, runname)
    logger.info(f"WDIR: %s", mdrun.rundir)
    mdrun.prepare_files()
    inpdb = mdsys.root / 'system.pdb'
    pdb = app.PDBFile(str(inpdb))
    ff  = app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    # --- Build a system WITHOUT any motion remover; no barostat/thermostat added ---
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
    nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        removeCMMotion=False,     # important for strict NVE
        ewaldErrorTolerance=1e-5
    )
    # --- NVT integrator (for short equilibration) ---
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, GAMMA, 0.5*TSTEP)
    integrator.setConstraintTolerance(1e-6)
    simulation = app.Simulation(pdb.topology, system, integrator) #  platform, properties)
    # --- Reporters (energies to monitor drift) ---
    log_reporters = [
        app.StateDataReporter(
            str(mdrun.rundir / "md.log"), 1000, step=True, potentialEnergy=True, kineticEnergy=True,
            totalEnergy=True, temperature=True, speed=True
        ),
        app.StateDataReporter(
            sys.stderr, 1000, step=True, potentialEnergy=True, kineticEnergy=True,
            totalEnergy=True, temperature=True, speed=True
        ),
    ]
    simulation.reporters.extend(log_reporters)
    # --- Initialize state, minimize, equilibrate ---
    logger.info("Minimizing energy...")
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=1000)  
    logger.info("Equilibrating...")
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)  # set initial kinetic energy
    simulation.step(10000)  # equilibrate for 10 ps
    # --- Run NVE (need to change the integrator and reset simulation) ---
    logger.info("Running NVE production...")
    integrator = mm.VerletIntegrator(TSTEP)
    integrator.setConstraintTolerance(1e-6)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setState(state)
    mda.Universe(mdsys.syspdb).select_atoms(OUT_SELECTION).write(mdrun.rundir / "md.pdb") # SAVE PDB FOR THE SELECTION
    traj_reporter = MmReporter(str(mdrun.rundir / "md.trr"), reportInterval=NOUT, selection=OUT_SELECTION)
    simulation.reporters.append(traj_reporter)
    simulation.reporters.extend(log_reporters)
    simulation.step(100000)  
    logger.info("Done!")


def extend(sysdir, sysname, runname):    
    mdrun = MmRun(sysdir, sysname, runname)
    logger.info(f"WDIR: %s", mdrun.rundir)
    pdb = app.PDBFile(str(mdrun.syspdb))
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, GAMMA, TSTEP)
    with open(str(mdrun.sysxml)) as f:
        system = mm.XmlSerializer.deserialize(f.read())
    simulation = app.Simulation(pdb.topology, system, integrator)
    barostat = mm.MonteCarloBarostat(PRESSURE, TEMPERATURE)
    simulation.system.addForce(barostat)
    enum = enumerate(simulation.system.getForces()) 
    idx, bb_restraint = [(idx, f) for idx, f in enum if f.getName() == 'BackboneRestraint'][0]
    simulation.system.removeForce(idx)
    simulation.context.reinitialize(preserveState=True)
    mdrun.extend(simulation, until_time=TOTAL_TIME)


def trjconv(sysdir, sysname, runname):
    system = MDSystem(sysdir, sysname)
    mdrun = MDRun(sysdir, sysname, runname)
    logger.info(f"WDIR: %s", mdrun.rundir)
    traj = str(mdrun.rundir / "md.trr")
    top = str(mdrun.rundir / "md.pdb")
    # top = mdrun.syspdb  # use original topology to avoid missing atoms
    conv_top = str(mdrun.rundir / "topology.pdb")
    if SELECTION != OUT_SELECTION:
        conv_traj = str(mdrun.rundir / f"md_selection.trr")
        _trjconv_selection(traj, top, conv_traj, conv_top, selection=SELECTION, step=1)
    else:
        conv_traj = traj
        shutil.copy(top, conv_top)
    out_traj = str(mdrun.rundir / f"samples.trr")
    _trjconv_fit(conv_traj, conv_top, out_traj, transform_vels=True)


def tdlrt_analysis(sysdir, sysname, runname):
    mdrun = MDRun(sysdir, sysname, runname)
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
    ps = ps - ps[:, 0][..., None]
    # ps -= ps.mean(axis=1)[..., None]
    # CCF calculations
    adict = {'pv': (ps, vs), 'vv': (vs, vs), } #  adict = {'pv': (ps, vs)}
    for key, item in adict.items(): # DT = TSTEP * NOUT
        v1, v2 = item
        corr = mdm.ccf(v1, v2, ntmax=2000, n=1, mode='gpu', center=False, dtype=np.float32) # falls back on cpu if no cuda
        corr_file = mdrun.lrtdir / f'ccf_1_{key}.npy'
        np.save(corr_file, corr)    
        logger.info("Saved CCFs to %s", corr_file)


def get_averages(sysdir, pattern, dtype=None, *args):
    """Calculate average arrays across files matching pattern."""
    nprocs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    logger.info("Number of available processors: %s", nprocs)
    files = io.pull_files(sysdir, pattern)[::]
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
    outdir = Path('data') / Path(sysdir).relative_to('systems')
    outdir.mkdir(exist_ok=True, parents=True)
    out_file = outdir / f"{pattern.split('*')[0]}_av.npy"
    np.save(out_file, average)
    logger.info("Saved averages to %s", out_file)


def _slicer(shape):
    return tuple(slice(0, s) for s in shape)


def _process_batch(args, dtype=np.float32):
    """Worker: load assigned files, crop to min_shape and return local sum and count."""
    files, min_shape = args
    s = _slicer(min_shape)
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


##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

def _add_bb_restraints(system, pdb, bb_aname='CA'):
    restraint = mm.CustomExternalForce('bb_fc*periodicdistance(x, y, z, x0, y0, z0)^2')
    restraint.setName('BackboneRestraint')
    restraint.addGlobalParameter('bb_fc', 1000.0*kilojoules_per_mole/nanometer)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')
    system.addForce(restraint)
    for atom in pdb.topology.atoms():
        if atom.name == bb_aname:
            restraint.addParticle(atom.index, pdb.positions[atom.index])


def _trjconv_selection(input_traj, input_top, output_traj, output_top, selection="name CA", step=1):
    u = mda.Universe(input_top, input_traj)
    selected_atoms = u.select_atoms(selection)
    n_atoms = selected_atoms.n_atoms
    selected_atoms.write(output_top)
    with mda.Writer(output_traj, n_atoms=n_atoms) as writer:
        for ts in u.trajectory[::step]:
            writer.write(selected_atoms)
    logger.info("Saved selection '%s' to %s and topology to %s", selection, output_traj, output_top)


def _trjconv_fit(input_traj, input_top, output_traj, transform_vels=False):
    u = mda.Universe(input_top, input_traj)
    ag = u.atoms
    ref_u = mda.Universe(input_top) 
    ref_ag = ref_u.atoms
    u.trajectory.add_transformations(fit_rot_trans(ag, ref_ag,))
    logger.info("Converting/Writing Trajecory")
    with mda.Writer(output_traj, ag.n_atoms) as W:
        for ts in u.trajectory:   
            if transform_vels:
                transformed_vels = _tranform_velocities(ts.velocities, ts.positions, ref_ag.positions)
                ag.velocities = transformed_vels
            W.write(ag)
            if ts.frame % 1000 == 0:
                frame = ts.frame
                time_ns = ts.time // 1000
                logger.info(f"Current frame: %s at %s ns", frame, time_ns)
    logger.info("Done!")


def _tranform_velocities(vels, poss, ref_poss):
    R = _kabsch_rotation(poss, ref_poss)
    vels_aligned = vels @ R
    return vels_aligned
    

def _kabsch_rotation(P, Q):
    """
    Return the 3x3 rotation matrix R that best aligns P onto Q (both Nx3),
    after removing centroids (i.e., pure rotation via Kabsch).
    """
    # subtract centroids
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    # covariance and SVD
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # right-handed correction
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R


def _get_platform_info():
    """Report OpenMM platform and hardware information."""
    info = {}
    # Get number of available platforms and their names
    num_platforms = mm.Platform.getNumPlatforms()
    info['available_platforms'] = [mm.Platform.getPlatform(i).getName() 
                                 for i in range(num_platforms)]
    # Try to get the fastest platform (usually CUDA or OpenCL)
    platform = None
    for platform_name in ['CUDA', 'OpenCL', 'CPU']:
        try:
            platform = mm.Platform.getPlatformByName(platform_name)
            info['platform'] = platform_name
            break
        except Exception:
            continue 
    if platform is None:
        platform = mm.Platform.getPlatform(0)
        info['platform'] = platform.getName()
    # Get platform properties
    info['properties'] = {}
    try:
        if info['platform'] in ['CUDA', 'OpenCL']:
            info['properties']['device_index'] = platform.getPropertyDefaultValue('DeviceIndex')
            info['properties']['precision'] = platform.getPropertyDefaultValue('Precision')
            if info['platform'] == 'CUDA':
                info['properties']['cuda_version'] = mm.version.cuda
            info['properties']['gpu_name'] = platform.getPropertyValue(platform.createContext(), 'DeviceName')
        info['properties']['cpu_threads'] = platform.getPropertyDefaultValue('Threads')
    except Exception as e:
        logger.warning(f"Could not get some platform properties: {str(e)}")
    # Get OpenMM version
    info['openmm_version'] = mm.version.full_version
    # Log the information
    logger.info("OpenMM Platform Information:")
    logger.info(f"Available Platforms: {', '.join(info['available_platforms'])}")
    logger.info(f"Selected Platform: {info['platform']}")
    logger.info(f"OpenMM Version: {info['openmm_version']}")
    logger.info("Platform Properties:")
    for key, value in info['properties'].items():
        logger.info(f"  {key}: {value}")
    return info


def _pdb_to_seq(pdb):
    u = mda.Universe(pdb)
    protein = u.select_atoms("protein")
    seq = "".join(res.resname for res in protein.residues)  # three-letter codes
    seq_oneletter = "".join(mda.lib.util.convert_aa_code(res.resname) for res in protein.residues)
    return seq_oneletter


def _get_module_functions(module):
    """Get all non-private functions from a module"""
    return {name: obj for name, obj in inspect.getmembers(module, inspect.isfunction)
            if not name.startswith('_')}


def _main():
    if len(sys.argv) < 2:
        print("Usage: <script> <command> [args...]")
        sys.exit(1)
    command = sys.argv[1]
    args = sys.argv[2:]
    module = sys.modules[__name__] # current module
    functions = _get_module_functions(module)
    if command not in functions:
        raise ValueError(f"Unknown command: {command}. Available commands for {module_name}: {', '.join(functions.keys())}")
    try:
        functions[command](*args)
    except Exception as e:
        print(f"Error executing {module_name}.{command}: {str(e)}")
        raise


if __name__ == "__main__":
    _main()