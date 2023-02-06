export MODEL_NAME="/home/v-yuancwang/AudioEditing/MyPipeline"
export TRAIN_DIR=""

accelerate launch --mixed_precision="fp16" \
  --gpu_ids="all" \
  --num_machines=1 \
  --num_processes=8 \
  train_vae.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
<<<<<<< HEAD
  --gradient_accumulation_steps=2 \
=======
  --gradient_accumulation_steps=1 \
>>>>>>> 3a3bede98f0a42c1d3cefd67f4336358039cd9c8
  --gradient_checkpointing \
  --max_train_steps=500000 \
  --checkpointing_steps=10000 \
  --learning_rate=7.5e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/blob/v-yuancwang/AudioEditing/Finetune_VAE_2"\