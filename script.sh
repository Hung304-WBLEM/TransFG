mass_shape_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_shape_comb_feats_omit'
mass_margins_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_margins_comb_feats_omit'
calc_type_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_type_comb_feats_omit'
calc_dist_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_dist_comb_feats_omit'

# Test Purpose

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Shape --img_size 224 --split overlap --criterion bce --wc --num_steps 10 --fp16 --name test --data_root . --pretrained_dir 'ViT-B_16.npz'

# MASS SHAPE
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Shape --img_size 224 --split overlap --criterion bce --wc --num_steps 1300 --fp16 --name mass_shape_224x224_bce_wc --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Shape --train_batch_size 32 --img_size 224 --split overlap --criterion bce --wc --num_steps 1300 --fp16 --name mass_shape_224x224_b32_bce_wc --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Shape --train_batch_size 16 --img_size 224 --split overlap --slide_step 6 --criterion bce --wc --num_steps 1300 --fp16 --name mass_shape_224x224_b16_bce_wc_slidestep-6 --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Shape --split overlap --criterion bce --wc --num_steps 1300 --fp16 --name mass_shape_448x448_bce_wc --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Shape --img_size 224 --split overlap --slide_step 10 --num_steps 1300 --fp16 --name mass_shape_224x224_slidestep-10 --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Shape --img_size 224 --split overlap --slide_step 8 --num_steps 1300 --fp16 --name mass_shape_224x224_slidestep-8 --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Shape --img_size 224 --split overlap --slide_step 6 --num_steps 1300 --fp16 --name mass_shape_224x224_slidestep-6 --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Shape --img_size 224 --split overlap --slide_step 4 --num_steps 1300 --fp16 --name mass_shape_224x224_slidestep-4 --data_root . --pretrained_dir 'ViT-B_16.npz'

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Shape --split overlap --num_steps 1300 --fp16 --name mass_shape_448x448 --data_root . --pretrained_dir 'ViT-B_16.npz'

# MASS MARGINS
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Margins --img_size 224 --split overlap --criterion bce --wc --num_steps 1200 --fp16 --name mass_margins_224x224_bce_wc --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Margins --train_batch_size 32 --img_size 224 --split overlap --criterion bce --wc --num_steps 1300 --fp16 --name mass_margins_224x224_b32_bce_wc --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Margins --split overlap --criterion bce --wc --num_steps 1300 --fp16 --name mass_margins_448x448_bce_wc --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Margins --train_batch_size 16 --img_size 224 --split overlap --slide_step 6 --criterion bce --wc --num_steps 1300 --fp16 --name mass_margins_224x224_b16_bce_wc_slidestep-6 --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Margins --split overlap --num_steps 1300 --fp16 --name mass_margins_448x448 --data_root . --pretrained_dir 'ViT-B_16.npz'

# CALC TYPE
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Calc_Type --img_size 224 --split overlap --criterion bce --wc --num_steps 1300 --fp16 --name calc_type_224x224_bce_wc --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Calc_Type --train_batch_size 32 --img_size 224 --split overlap --criterion bce --wc --num_steps 1300 --fp16 --name calc_type_224x224_b32_bce_wc --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Calc_Type --split overlap --criterion bce --wc --num_steps 1300 --fp16 --name calc_type_448x448_bce_wc --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Calc_Type --train_batch_size 16 --img_size 224 --split overlap --slide_step 6 --criterion bce --wc --num_steps 1300 --fp16 --name calc_type_224x224_b16_bce_wc_slidestep-6 --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Calc_Type --split overlap --num_steps 1300 --fp16 --name calc_type_448x448 --data_root . --pretrained_dir 'ViT-B_16.npz'

# CALC DIST
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Calc_Dist --img_size 224 --split overlap --criterion bce --num_steps 1200 --fp16 --name calc_dist_224x224_bce_wc --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Calc_Dist --train_batch_size 32 --img_size 224 --split overlap --criterion bce --wc --num_steps 1300 --fp16 --name calc_dist_224x224_b32_bce_wc --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Calc_Dist --split overlap --criterion bce --wc --num_steps 1300 --fp16 --name calc_dist_448x448_bce_wc --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Calc_Dist --train_batch_size 16 --img_size 224 --split overlap --slide_step 6 --criterion bce --wc --num_steps 1300 --fp16 --name calc_dist_224x224_b16_bce_wc_slidestep-6 --data_root . --pretrained_dir 'ViT-B_16.npz'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Calc_Dist --split overlap --num_steps 1300 --fp16 --name calc_dist_448x448 --data_root . --pretrained_dir 'ViT-B_16.npz'






# # 5000 epochs
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Shape --img_size 224 --split overlap --num_steps 5000 --fp16 --name mass_shape_224x224_e5000 --data_root . --pretrained_dir 'ViT-B_16.npz'

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Mass_Margins --img_size 224 --split overlap --num_steps 5000 --fp16 --name mass_margins_224x224_e5000 --data_root . --pretrained_dir 'ViT-B_16.npz'

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Calc_Type --img_size 224 --split overlap --num_steps 5000 --fp16 --name calc_type_224x224_e5000 --data_root . --pretrained_dir 'ViT-B_16.npz'

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset Calc_Dist --img_size 224 --split overlap --num_steps 5000 --fp16 --name calc_dist_224x224_e5000 --data_root . --pretrained_dir 'ViT-B_16.npz'


##############################################################################
######################### Copy to experiment folders #########################
##############################################################################
# mkdir -p ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_224x224
# cp -r logs/mass_margins_224x224 ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_224x224/logs
# cp output/mass_margins_224x224_checkpoint.bin ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_224x224/ckpt.bin

# mkdir -p ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_224x224
# cp -r logs/calc_type_224x224 ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_224x224/logs
# cp output/calc_type_224x224_checkpoint.bin ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_224x224/ckpt.bin

# mkdir -p ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_224x224
# cp -r logs/calc_dist_224x224 ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_224x224/logs
# cp output/calc_dist_224x224_checkpoint.bin ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_224x224/ckpt.bin

# mkdir -p ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e5000_224x224
# cp -r logs/mass_margins_224x224_e5000 ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e5000_224x224/logs
# cp output/mass_margins_224x224_e5000_checkpoint.bin ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e5000_224x224/ckpt.bin

# mkdir -p ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_stepsize-10
# cp -r logs/mass_shape_224x224_slidestep-10 ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_stepsize-10/logs
# cp output/mass_shape_224x224_slidestep-10_checkpoint.bin ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_stepsize-10/ckpt.bin

# mkdir -p ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_stepsize-8
# cp -r logs/mass_shape_224x224_slidestep-8 ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_stepsize-8/logs
# cp output/mass_shape_224x224_slidestep-8_checkpoint.bin ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_stepsize-8/ckpt.bin


# mkdir -p ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_stepsize-6
# cp -r logs/mass_shape_224x224_slidestep-6 ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_stepsize-6/logs
# cp output/mass_shape_224x224_slidestep-6_checkpoint.bin ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_stepsize-6/ckpt.bin

# mkdir -p ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc
# cp -r logs/mass_margins_224x224_bce_wc ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc/logs
# cp output/mass_margins_224x224_bce_wc_checkpoint.bin ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc/ckpt.bin

# mkdir -p ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc
# cp -r logs/calc_type_224x224_bce_wc ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc/logs
# cp output/calc_type_224x224_bce_wc_checkpoint.bin ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc/ckpt.bin

# mkdir -p ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc
# cp -r logs/calc_dist_224x224_bce_wc ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc/logs
# cp output/calc_dist_224x224_bce_wc_checkpoint.bin ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc/ckpt.bin


# mkdir -p ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc
# cp -r logs/mass_shape_448x448_bce_wc ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc/logs
# cp output/mass_shape_448x448_bce_wc_checkpoint.bin ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc/ckpt.bin

# mkdir -p ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc
# cp -r logs/mass_margins_448x448_bce_wc ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc/logs
# cp output/mass_margins_448x448_bce_wc_checkpoint.bin ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc/ckpt.bin

# mkdir -p ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc
# cp -r logs/calc_type_448x448_bce_wc ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc/logs
# cp output/calc_type_448x448_bce_wc_checkpoint.bin ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc/ckpt.bin

# mkdir -p ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc
# cp -r logs/calc_dist_448x448_bce_wc ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc/logs
# cp output/calc_dist_448x448_bce_wc_checkpoint.bin ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc/ckpt.bin


# mkdir -p ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6
# cp -r logs/mass_shape_224x224_b16_bce_wc_slidestep-6 ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6/logs
# cp output/mass_shape_224x224_b16_bce_wc_slidestep-6_checkpoint.bin ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6/ckpt.bin

# mkdir -p ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6
# cp -r logs/mass_margins_224x224_b16_bce_wc_slidestep-6 ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6/logs
# cp output/mass_margins_224x224_b16_bce_wc_slidestep-6_checkpoint.bin ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6/ckpt.bin

# mkdir -p ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6
# cp -r logs/calc_type_224x224_b16_bce_wc_slidestep-6 ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6/logs
# cp output/calc_type_224x224_b16_bce_wc_slidestep-6_checkpoint.bin ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6/ckpt.bin

# mkdir -p ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6
# cp -r logs/calc_dist_224x224_b16_bce_wc_slidestep-6 ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6/logs
# cp output/calc_dist_224x224_b16_bce_wc_slidestep-6_checkpoint.bin ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6/ckpt.bin

mkdir -p ${mass_shape_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc
cp -r logs/mass_shape_224x224_b32_bce_wc ${mass_shape_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc/logs
cp output/mass_shape_224x224_b32_bce_wc_checkpoint.bin ${mass_shape_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc/ckpt.bin

mkdir -p ${mass_margins_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc
cp -r logs/mass_margins_224x224_b32_bce_wc ${mass_margins_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc/logs
cp output/mass_margins_224x224_b32_bce_wc_checkpoint.bin ${mass_margins_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc/ckpt.bin

mkdir -p ${calc_type_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc
cp -r logs/calc_type_224x224_b32_bce_wc ${calc_type_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc/logs
cp output/calc_type_224x224_b32_bce_wc_checkpoint.bin ${calc_type_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc/ckpt.bin

mkdir -p ${calc_dist_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc
cp -r logs/calc_dist_224x224_b32_bce_wc ${calc_dist_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc/logs
cp output/calc_dist_224x224_b32_bce_wc_checkpoint.bin ${calc_dist_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc/ckpt.bin
