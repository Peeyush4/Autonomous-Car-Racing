#!/bin/bash
# The line above this is the "shebang" line.  It must be first line in script
#-----------------------------------------------------
#       OnDemand Job Template for Hello-UMD, MPI version
#       Runs a simple MPI enabled hello-world code
#-----------------------------------------------------
#
# Slurm sbatch parameters section:
#       Request 60 MPI tasks with 1 CPU core each
#SBATCH -n 1
#SBATCH -c 1
#       Request 5 minutes of walltime
#SBATCH -t 1-00:00:00
#       Request 16 GB of memory per CPU core
#SBATCH --mem-per-cpu=32768
##      Do not allow other jobs to run on same node
##SBATCH --exclusive
#       Run on debug partition for rapid turnaround.  You will need
#       to change this (remove the line) if walltime > 15 minutes
#SBATCH --partition=gpu
#SBATCH --gpus=a100_1g.5gb:1
#       Do not inherit the environment of the process running the
#       sbatch command.  This requires you to explicitly set up the
#       environment for the job in this script, improving reproducibility
#SBATCH --export=NONE

# This job will run the MPI enabled version of hello-umd
# We create a directory on parallel filesystem from where we actually
# will run the job.
# Section to ensure we have the "module" command defined
unalias tap >& /dev/null
if [ -f ~/.bash_profile ]; then
        source ~/.bash_profile
elif [ -f ~/.profile ]; then
        source ~/.profile
fi
# Set SLURM_EXPORT_ENV to ALL.  This prevents the --export=NONE flag
# from being passed to mpirun/srun/etc, which can cause issues.
# We want the environment of the job script to be passed to all
# tasks/processes of the job
export SLURM_EXPORT_ENV=ALL
# Module load section
# First clear our module list
module purge
# and reload the standard modules

module load hpcc/deepthought2

# Load the desired compiler,  MPI, and package modules
# NOTE: You need to use the same compiler and MPI module used
# when compiling the MPI-enabled code you wish to run (in this
# case hello-umd).  The values # listed below are correct for the
# version of hello-umd we will be using, but you may need to
# change them if you wish to run a different package.
module load gcc/11.3.0
# module load openmpi/3.1.5
module load hello-umd/1.5
module load python/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
module load cuda/12.3.0/gcc/11.3.0/zen2
module load cudnn/8.9.7.29-12/gcc/11.3.0/zen2
# Section to make a scratch directory for this job
# Because different MPI tasks, which might be on different nodes, and will need
# access to it, we put it in a parallel file system.
# We include the SLURM jobid in the directory name to avoid interference
# if multiple jobs running at same time.
# TMPWORKDIR="/lustre/$USER/ood-job.${SLURM_JOBID}"
# mkdir $TMPWORKDIR
# cd $TMPWORKDIR

# Section to output information identifying the job, etc.
echo "Slurm job ${SLURM_JOBID} running on"
hostname
echo "To run on ${SLURM_NTASKS} CPU cores across ${SLURM_JOB_NUM_NODES} nodes"
echo "All nodes: ${SLURM_JOB_NODELIST}"
date
pwd
echo "Loaded modules are:"
module list
echo "Job will be started out of $TMPWORKDIR"
# Setting this variable will suppress the warnings
# about lack of CUDA support on non-GPU enabled nodes.  We
# are not using CUDA, so warning is harmless.
export OMPI_MCA_mpi_cuda_support=0
export WANDB_DEBUG=debug

# Get the full path to our hello-umd executable.  It is best
# to provide the full path of our executable to mpirun, etc.
MYEXE=`which hello-umd`
echo "Using executable $MYEXE"
# Run our script using mpirun
# We do not specify the number of tasks here, and instead rely on
# it defaulting to the number of tasks requested of Slurm
# mpirun  ${MYEXE}  > hello.out 2>&1
cd /scratch/zt1/project/msml642/shared/Group9/pytorch_car_caring/code
source /scratch/zt1/project/msml642/shared/Group9/.venv/bin/activate
echo "venv started"

python training_with_leakyReLU.py  > training_with_leakyReLU.out 2>&1
# Save the exit code from the previous command
ECODE=$?
# Output from the above command was placed in a work directory in a parallel
# filesystem.  That parallel filesystem does _not_ get cleaned up automatically.

# And it is not normally visible from the Job Composer.

# To deal with this, we make a symlink from the job submit directory to

# the work directory for the job.

#

# NOTE: The work directory will continue to exist until you delete it.  It will

# not get deleted when you delete the job in Job Composer.

# ln -s ${TMPWORKDIR} ${SLURM_SUBMIT_DIR}/work-dir
echo "Job finished with exit code $ECODE." # Work dir is $TMPWORKDIR"
date
# Exit with the cached exit code

exit $ECODE
