#!/bin/bash
#SBATCH --job-name=anla_png      # Job name
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=anla@pml.ac.uk   # Where to send mail	
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                   # Run a single task		
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=50gb                   # Job memory request
#SBATCH --time=01:00:00              # Time limit hrs:min:sec
#SBATCH --output=/Lustre/user_scratch/anla/shiptrack_request_21_02/slurm_logs/png_creator%j.log     # Standard output and error log
#SBATCH --partition=batch
#SBATCH --array=1-12%6

# create a job array, one for each month
# given directory to the year
YEAR_DIR=$1
MONTH=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

# find the netcdf files, process contours
python make_daily_png.py --path $YEAR_DIR/$MONTH\
 --level 0.8 --latmin -80 --latmax 80