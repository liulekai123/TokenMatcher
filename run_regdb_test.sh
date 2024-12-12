#!/usr/bin/bash
#SBATCH --partition=gpu       # 指定分配GPU分区
#SBATCH --account=chenjun3    # 指定账户名
#SBATCH --ntasks=1            # 指定任务数，这里为1个任务
#SBATCH --ntasks-per-node=4   # 每个节点的任务数，这里为4
#SBATCH --cpus-per-task=5     # 每个任务的CPU核心数，这里为5
#SBATCH --gres=gpu:4          # 指定GPU资源数，这里为4块GPU
#SBATCH --exclude=g0045,g0017,g0015  # 排除的节点
#SBATCH -o test_regdb_vis_query.log    # 输出日志文件
# module load scl/gcc5.3      # 加载指定的软件模块（在这里被注释掉了）


module load nvidia/cuda/11.6

CUDA_VISIBLE_DEVICES=0,1,2,3 python test_regdb_vis_query.py -b 128 -a vit_base -d regdb_rgb --iters 100 --num-instances 16 --self-norm --hw-ratio 2 --conv-stem \
-pp /scratch/chenjun3/liulekai/PGM-ReID-main/examples/pretrained/vit_base_ics_cfs_lup.pth --cls-token-num 6
 
#-m torch.distributed.launch --nproc_per_node=4