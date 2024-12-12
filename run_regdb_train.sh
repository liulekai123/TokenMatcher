

module load nvidia/cuda/11.6

for trial in 1 2 3 4 5 6 7 8 9 10
do 
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_regdb.py -b 128 -a vit_base -d regdb_rgb --num-instances 16 --iters 100 --self-norm --conv-stem \
-pp /scratch/chenjun3/liulekai/Tokens/examples/pretrained/vit_base_ics_cfs_lup.pth --cls-token-num 6 --lamba-neighbor 0.1 --lamba-cross 0.4 --lamba-mate 0.01 --epochs 35 --trial $trial
done
echo 'Done'

