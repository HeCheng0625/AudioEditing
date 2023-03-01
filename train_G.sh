export MODEL_NAME="/blob/v-yuancwang/AudioEditingModel/Diffusion_SG/checkpoint-40000"
export TRAIN_DIR=""

accelerate launch --mixed_precision="no" \
  --gpu_ids="all" \
  --num_machines=1 \
  --num_processes=8 \
  train_diffusion_G.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=1000000 \
  --checkpointing_steps=2000 \
  --learning_rate=5e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/blob/v-yuancwang/AudioEditingModel/Diffusion_SG"\
  --resume_from_checkpoint="/blob/v-yuancwang/AudioEditingModel/Diffusion_SG/checkpoint-240000"