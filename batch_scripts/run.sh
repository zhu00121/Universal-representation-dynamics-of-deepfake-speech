
#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --gres=gpu:v100l:4     
#SBATCH --mem=128G       
#SBATCH --time=10:0:0    
#SBATCH --mail-user=<YOU EMAIL>
#SBATCH --mail-type=ALL

# move repo to the compute node
cp -R $project/Universal-representation-dynamics-of-deepfake-speech $SLURM_TMPDIR
cp ~/scratch/LA.zip $SLURM_TMPDIR/Universal-representation-dynamics-of-deepfake-speech/data/ASVspoof_2019
cp ~/scratch/ASVspoof2021_DF_eval_*.tar.gz $SLURM_TMPDIR/Universal-representation-dynamics-of-deepfake-speech/data/ASVspoof_2021

# unzip dataset
# the first one is the 2019 ASVspoof dataset, the second to the fifth are the 2021 ASVspoof evaluation set
unzip $SLURM_TMPDIR/Universal-representation-dynamics-of-deepfake-speech/data/ASVspoof_2019/LA.zip -d $SLURM_TMPDIR/Universal-representation-dynamics-of-deepfake-speech/data/ASVspoof_2019
tar -xvf $SLURM_TMPDIR/Universal-representation-dynamics-of-deepfake-speech/data/ASVspoof_2021/ASVspoof2021_DF_eval_part00.tar.gz -C $SLURM_TMPDIR/Universal-representation-dynamics-of-deepfake-speech/data/ASVspoof_2021/
tar -xvf $SLURM_TMPDIR/Universal-representation-dynamics-of-deepfake-speech/data/ASVspoof_2021/ASVspoof2021_DF_eval_part01.tar.gz -C $SLURM_TMPDIR/Universal-representation-dynamics-of-deepfake-speech/data/ASVspoof_2021/
tar -xvf $SLURM_TMPDIR/Universal-representation-dynamics-of-deepfake-speech/data/ASVspoof_2021/ASVspoof2021_DF_eval_part02.tar.gz -C $SLURM_TMPDIR/Universal-representation-dynamics-of-deepfake-speech/data/ASVspoof_2021/
tar -xvf $SLURM_TMPDIR/Universal-representation-dynamics-of-deepfake-speech/data/ASVspoof_2021/ASVspoof2021_DF_eval_part03.tar.gz -C $SLURM_TMPDIR/Universal-representation-dynamics-of-deepfake-speech/data/ASVspoof_2021/

# initiate virtual env
module load python/3.10 scipy-stack
source ~/myenv-speech/bin/activate

# lauch job on node
cd $SLURM_TMPDIR/Universal-representation-dynamics-of-deepfake-speech
python -m torch.distributed.launch --nproc_per_node=4 ./exps/ASVspoof_2019/train.py ./exps/ASVspoof_2019/hparams/LA_v10.yaml --distributed_launch
# copy results to $scratch
mkdir ~/scratch/DF_v10
cp -R ./exps/ASVspoof_2019/brain-logs ~/scratch/DF_v10