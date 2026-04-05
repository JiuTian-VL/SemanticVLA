featurizer_cfg="{
  'use_film': True,
  'use_reg': True,
  'num_vision_queries': 32,
  'use_mm_sampler': False,
  'text_encoder_path': null,
  'vision_topk': -1,
  'text_topk': -1,
  'freeze': False
}"

fused_featurizer_cfg="{
  'use_film': False,
  'use_reg': False,
  'num_vision_queries': -1,
  'use_mm_sampler': True,
  'text_encoder_path': '/data1/Models/siglip-so400m-patch14-384',
  'vision_topk': 32,
  'text_topk': 5,
  'freeze': False
}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes 1 --nproc-per-node 8 --master_port=5558 vla-scripts/finetune.py \
  --vla_path /data1/Models/openvla-7b \
  --data_root_dir /data1/ALOHA/tensorflow_datasets \
  --dataset_name change_the_name \
  --run_root_dir /data/ckpt \
  --use_l1_regression True \
  --use_diffusion False \
  --vision_aggregate_type concat \
  --num_action_tokens 6 \
  --enable_cross_vit_interact True \
  --cross_vit_target_pairs "[[2, 3], [11, 13], [20, 23]]" \
  --featurizer_cfg "$featurizer_cfg" \
  --fused_featurizer_cfg "$fused_featurizer_cfg" \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 80005 \
  --save_freq 10000 \
  --merge_lora_during_training False \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --lora_alpha 64 \
  --wandb_entity "entity" \
  --wandb_project "openvla" \
  --run_id_note "run_id"