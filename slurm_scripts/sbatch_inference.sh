#!/bin/bash
#SBATCH --job-name=neodaas_request_21_02_serial_inference      # Job name
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=anla@pml.ac.uk   # Where to send mail	
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                   # Run a single task		
#SBATCH --cpus-per-task=10           # Number of CPU cores per task
#SBATCH --mem=20gb                   # Job memory request
#SBATCH --time=2:00:00              # Time limit hrs:min:sec
#SBATCH --output=/Lustre/user_scratch/anla/shiptrack_request_21_02/slurm_logs/inference%j.log     # Standard output and error log
#SBATCH --gres=gpu:1                 # get hold of a GPU
echo "Running shiptrack inference on $SLURM_CPUS_ON_NODE CPU cores"

# enable the module command
source /etc/profile.d/z00_lmod.sh

# load the singularity module
module load linux-ubuntu18.04-broadwell/singularity-3.5.3-gcc-7.5.0-ype77gs

# assign the path to the container to the variable, container
# container = /Lustre/user_scratch/anla/singularity/mageoexperimental_latest.sif

# run infer shiptracks within the singularity container, using singularity exec
singularity exec --nv --bind /Lustre,/users/rsg/anla/.cache:/home/.cache /Lustre/user_scratch/anla/singularity/mageoexperimental_latest.sif\
python /users/rsg/anla/code/shiptrack-detection/tf_dataset_inference.py --infile /Lustre/user_scratch/anla/shiptrack_request_21_02/infiles_2010.txt