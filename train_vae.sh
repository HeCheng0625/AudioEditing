export MODEL_NAME="/home/v-yuancwang/AudioEditing/MyPipeline"
export TRAIN_DIR=""

accelerate launch --mixed_precision="fp16" \
  --gpu_ids="all" \
  --num_machines=1 \
  --num_processes=4 \
  train_vae.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --max_train_steps=500000 \
  --checkpointing_steps=10000 \
  --learning_rate=7.5e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/blob/v-yuancwang/AudioEditing/Finetune_VAE_2"\