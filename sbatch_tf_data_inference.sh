#!/bin/bash
#SBATCH --job-name=neodaas_request_21_02_serial_inference      # Job name
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=anla@pml.ac.uk   # Where to send mail	
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                   # Run a single task		
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --gres=gpu:1                 # get hold of a GPU
#SBATCH --mem=80gb                   # Job memory request
#SBATCH --time=01:00:00              # Time limit hrs:min:sec
#SBATCH --output=/Lustre/user_scratch/anla/shiptrack_request_21_02/slurm_logs/inference%j.log     # Standard output and error log

INFILE=$1

echo performing_inference_against_$INFILE

export TMPDIR=/raid/var/tmp

# enable the module command
source /etc/profile.d/z00_lmod.sh

# load the singularity module
module load linux-ubuntu18.04-broadwell/singularity-3.5.3-gcc-7.5.0-ype77gs

# assign the path to the container to the variable, container
CONTAINER=/Lustre/user_scratch/anla/singularity/mageoexperimental_latest.sif

# model could be an argument but for now just hardcode
MODEL=/Lustre/user_scratch/anla/shiptrack_request_21_02/model/20211210_112435_new_resnet152_bce_jaccard_loss_augmented/model/1

BATCHSIZE=2

# run infer shiptracks within the singularity container, using singularity exec
singularity exec --bind /Lustre,/users/rsg/anla/.cache:/home/.cache \
$CONTAINER \
python /users/rsg/anla/code/shiptrack-detection/tf_dataset_inference.py \
--infile $INFILE \
--model $MODEL \
--batch_size $BATCHSIZE \
--contour_level 0.9 0.5