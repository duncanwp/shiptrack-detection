#!/bin/bash
#SBATCH --job-name=anla_21_02_contours      # Job name
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=anla@pml.ac.uk   # Where to send mail	
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                   # Run a single task		
#SBATCH --cpus-per-task=10            # Number of CPU cores per task
#SBATCH --mem=20gb                   # Job memory request
#SBATCH --time=03:00:00              # Time limit hrs:min:sec
#SBATCH --output=/Lustre/user_scratch/anla/shiptrack_request_21_02/slurm_logs/inference%j.log     # Standard output and error log
#SBATCH --partition=batch

# compute contours for the netcdf files in given dir
DIR=$1

# find the netcdf files, process contours
find $DIR -name "*.nc" | xargs -I {} -P 10 \
python contour_nc_to_gpkg.py --input {} \
--vars shiptracks shiptracks --levels 0.8 0.5
