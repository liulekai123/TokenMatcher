

module load nvidia/cuda/11.6

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_sysu.py -b 128 -a vit_base -d sysu_all --num-instances 16 --data-dir "/scratch/chenjun3/liulekai/ADCA/data/sysu" --iters 200 --self-norm --conv-stem \
-pp /scratch/chenjun3/liulekai/Tokens/examples/pretrained/vit_base_ics_cfs_lup.pth --cls-token-num 4 --lamba-cross 0.4 --lamba-neighbor 0.5 --lamba-mate 0.03 --epochs 50

# First, run the first stage of train_sysu to obtain the baseline, then run train_sysu_IRcam and train_sysu_RGBcam to obtain reliable local clusters, 
# and finally, run the second stage of train_sysu.