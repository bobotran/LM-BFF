# Required environment variables:
# TAG: tag for the trail
# TYPE: finetune / prompt / prompt-demo  
# TASK: SST-2 / sst-5 / mr / cr / mpqa / subj / trec / CoLA / MNLI / SNLI / QNLI / RTE / MRPC / QQP / STS-B
# BS: batch size (recommendation: 2 / 4 / 8)
# LR: learning rate (recommendation: 1e-5 / 2e-5 / 5e-5)
# SEED: random seed (13 / 21 / 42 / 87 / 100)
# MODEL: pre-trained model name (roberta-*, bert-*), see Transformers model list

# Number of training instances per label
if [ -z "$K" ]
then
      K=3615
fi

NUM_EPOCHS=4
# Training steps
MAX_STEP=$(($K * 2 / $BS * $NUM_EPOCHS))

# Validation steps
EVAL_STEP=$(($K * 2 / ($BS * 1)))
LOG_STEP=$(($EVAL_STEP / 10))

# Task specific parameters
# The default length is 128 and the default number of samples is 16.
# For some tasks, we use longer length or double demo (when using demonstrations, double the maximum length).
# For some tasks, we use smaller number of samples to save time (because of the large size of the test sets).
# All those parameters are set arbitrarily by observing the data distributions.
TEMPLATE="*cls**sent_0*._It_was*mask*.*sep+*"
MAPPING="{0:'relevant',1:'irrelevant'}"


# Gradient accumulation steps
# For medium-sized GPUs (e.g., 2080ti with 10GB memory), they can only take 
# a maximum batch size of 2 when using large-size models. So we use gradient
# accumulation steps to achieve the same effect of larger batch sizes.
REAL_BS=2
GS=$(expr $BS / $REAL_BS)

if [ -z "$EPISODE" ]
then
      DATA_DIR="data/k-shot/$TASK/stuffed_binary_v1.0/$K-$SEED"
else
      DATA_DIR=data/$EPISODE/k-shot/$TASK/$K-$SEED
fi
# Use a random number to distinguish different trails (avoid accidental overwriting)
TRIAL_IDTF=$RANDOM

CHECKPOINT="result/it_was_mask_starved-spoilers-prompt-167-21-roberta-large-2e-5-4/"
# --resume_from $CHECKPOINT \
# --dataloader_num_workers 4 \

python run.py \
  --save_logit \
  --save_logit_dir result/$TAG \
  --task_name $TASK \
  --data_dir $DATA_DIR \
  --do_train \
  --do_predict \
  --do_eval \
  --evaluate_during_training \
  --model_name_or_path $MODEL \
  --few_shot_type $TYPE \
  --num_k $K \
  --max_seq_length 512 \
  --per_device_train_batch_size $REAL_BS \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps $GS \
  --learning_rate $LR \
  --max_steps $MAX_STEP \
  --logging_steps $LOG_STEP \
  --eval_steps $EVAL_STEP \
  --num_train_epochs 0 \
  --output_dir result/$TAG \
  --seed $SEED \
  --tag $TAG \
  --template $TEMPLATE \
  --mapping $MAPPING \
  $TASK_EXTRA \
  $1 

# Delete the checkpoint 
# Since we need to run multiple trials, saving all the checkpoints takes 
# a lot of storage space. You can find all evaluation results in `log` file anyway.
# rm -r result/$TAG \
# mv result/$TAG result/$TAG \
