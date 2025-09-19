import os
from pathlib import Path
from reforge.cli import sbatch, run

def dojob(submit, *args, **kwargs):
    """
    Submit a job if 'submit' is True; otherwise, run it via bash.
    
    Parameters:
        submit (bool): Whether to submit (True) or run (False) the job.
        *args: Positional arguments for the job.
        **kwargs: Keyword arguments for the job.
    """
    if submit:
        sbatch(*args, **kwargs)
    else:
        run('bash', *args)


def setup(submit=False, **kwargs): 
    """Set up the md model for each system name."""
    kwargs.setdefault('mem', '3G')
    for sysname in sysnames:
        dojob(submit, shscript, pyscript, 'setup', sysdir, sysname, 
              J=f'setup_{sysname}', **kwargs)


def md(submit=True, **kwargs):
    """Run molecular dynamics simulations for each system and mdrun."""
    kwargs.setdefault('t', '00-04:00:00')
    kwargs.setdefault('N', '1')
    kwargs.setdefault('n', '1')
    kwargs.setdefault('c', ntomp)
    kwargs.setdefault('mem', '3G')
    kwargs.setdefault('e', 'slurm_output/error.%A.err')
    kwargs.setdefault('o', 'slurm_output/output.%A.out')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, shscript, pyscript, 'md', sysdir, sysname, runname, ntomp, 
                  J=f'md_{sysname}', **kwargs)


def extend(submit=True, **kwargs):
    """Extend simulations by processing each system and mdrun."""
    kwargs.setdefault('t', '00-04:00:00')
    kwargs.setdefault('N', '1')
    kwargs.setdefault('n', '1')
    kwargs.setdefault('c', ntomp)
    kwargs.setdefault('G', '1')
    kwargs.setdefault('mem', '3G')
    kwargs.setdefault('e', 'slurm_output/error.%A.err')
    kwargs.setdefault('o', 'slurm_output/output.%A.out')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, shscript, pyscript, 'extend', sysdir, sysname, runname, ntomp, 
                  J=f'ext_{sysname}', **kwargs)
                

def trjconv(submit=True, **kwargs):
    """Convert trajectories for each system and mdrun."""
    kwargs.setdefault('t', '00-01:00:00')
    kwargs.setdefault('mem', '2G')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, shscript, pyscript, 'trjconv', sysdir, sysname, runname,
                  J=f'trjconv_{sysname}', **kwargs)


def tdlrt_analysis(submit=True, **kwargs):
    """Perform tdlrt analysis for each system and run."""
    kwargs.setdefault('t', '00-01:00:00')
    kwargs.setdefault('mem', '30G')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, shscript, pyscript, 'tdlrt_analysis', sysdir, sysname, runname,
                  J=f'tdlrt_{sysname}', **kwargs)
 

def get_averages(pattern, submit=False, **kwargs):
    """Calculate average values for each system."""
    kwargs.setdefault('mem', '80G')
    dojob(submit, shscript, pyscript, 'get_averages', sysdir, pattern,
            J=f'averages', **kwargs)


def sys_job(func, submit=False, **kwargs):
    """Submit or run a system-level job for each system.
    Parameters:
        jobname (str): The name of the job.
        submit (bool): Whether to submit the job.
        **kwargs: Additional keyword arguments.
    """
    for sysname in sysnames:
        dojob(submit, shscript, pyscript, func, sysdir, sysname, 
              J=f'{jobname}', **kwargs)


def run_job(func, submit=False, **kwargs):
    """Submit or run a run-level job for each system and run. 
    Parameters:
        jobname (str): The name of the job.
        submit (bool): Whether to submit the job.
        **kwargs: Additional keyword arguments.
    """
    kwargs.setdefault('t', '00-02:00:00')
    kwargs.setdefault('mem', '17G')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, shscript, pyscript, func, sysdir, sysname, runname,
                  J=f'{jobname}', **kwargs)


def ajob(func, submit=False, **kwargs):
    kwargs.setdefault('t', '00-01:00:00')
    dojob(submit, shscript, pyscript, func, sysdir, J=f'{jobname}', **kwargs)


def workflow(submit, **kwargs):
    kwargs.setdefault('t', '00-04:00:00')
    kwargs.setdefault('N', '1')
    kwargs.setdefault('n', '1')
    kwargs.setdefault('c', '1')
    kwargs.setdefault('mem', '3G')
    kwargs.setdefault('e', 'slurm_jobs/error.%A.err')
    kwargs.setdefault('o', 'slurm_jobs/output.%A.out')
    for sysname in sysnames:
        for runname in runs:
            dojob(submit, shscript, pyscript, 'workflow', sysdir, sysname, runname,
                  J='workflow', **kwargs)


if __name__ == "__main__":
    shscript = 'sbatch.sh'
    pyscript = 'workflow.py'


    sysdir = 'systems/1btl_nve'
    sysnames = [p.name for p in sorted(Path(sysdir).iterdir())]
    # sysnames = [p.name for p in sorted(Path(sysdir).iterdir()) if p.is_dir() and not (p / 'mdruns' / 'mdrun' / 'md.trr').exists()]
    # sysnames = ['sample_000']
    runs = ['mdrun']


    # ajob('workflow', 'initiate_systems_from_emu', submit=False)
    # setup(submit=True, md_module=md_module, mem='2G', q='public', p='htc', t='00:10:00',)
    # workflow(True, q='public', p='htc', t='00-00:20:00', c='1', mem='12G')
    # workflow(True, q='public', p='htc', t='00-00:10:00', c='1', mem='12G', G='1')
    # md(submit=False, md_module=md_module, ntomp=4, mem='2G', q='public', p='htc', t='00-01:00:00', G=1)
    # md(submit=True, md_module=md_module, ntomp=4, mem='4G', q='public', p='htc', t='00-00:15:00', G=1)
    # extend(submit=True, md_module=md_module, ntomp=8, mem='2G', q='public', p='htc', t='00-04:00:00', G=1)
    # extend(submit=True, md_module=md_module, ntomp=8, mem='2G', q='grp_sozkan', p='general', t='01-00:00:00', G=1)
    # trjconv(submit=True, md_module=md_module, t='00-00:20:00', q='public', p='htc', c='1', mem='2G')
    # tdlrt_analysis(submit=True, mem='7G', t='00-00:20:00',)
    get_averages(pattern='ccf_pv*.npy', submit=True, mem='4G') 
