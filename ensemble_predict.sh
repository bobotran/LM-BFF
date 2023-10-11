#!/bin/bash
for template_id in {49..50}
do
    # for seed in 13 21 42 87 100
    for seed in 21
    do
        # To save time, we fix these hyper-parameters
        bs=32
        lr=1e-5
        
        TAG=irrelevant-relevant-template-ensemble-predict
        TYPE=prompt
        TASK=spoilers
        BS=$bs
        LR=$lr
        SEED=$seed
        MODEL=roberta-large
        TEMPLATE_ID=$template_id
    
        K=3615
        NUM_EPOCHS=20
        # Training steps
        MAX_STEP=$(($K * 2 / $BS * $NUM_EPOCHS))
        
        # Validation steps
        EVAL_STEP=$(($K * 2 / $BS * 1))
        LOG_STEP=$(($EVAL_STEP / 10))
        
        TASK_EXTRA=""

        MAPPING="{0:'relevant',1:'irrelevant'}"
        
        REAL_BS=2
        GS=$(expr $BS / $REAL_BS)
        
        DATA_DIR="data/unlabeled_v1.0.1"
        
        CHECKPOINT=result/$template_id/
        
        # Since we only use dev performance here, use --no_predict to skip testing
        python run.py \
          --resume_from $CHECKPOINT \
          --save_logit \
          --save_logit_dir result/autolabeled_$template_id \
          --task_name $TASK \
          --data_dir $DATA_DIR \
          --do_predict \
          --do_eval \
          --model_name_or_path $MODEL \
          --few_shot_type $TYPE \
          --num_k $K \
          --max_seq_length 512 \
          --per_device_eval_batch_size 128 \
          --gradient_accumulation_steps $GS \
          --learning_rate $LR \
          --max_steps $MAX_STEP \
          --logging_steps $EVAL_STEP \
          --eval_steps $EVAL_STEP \
          --num_train_epochs 0 \
          --output_dir result/autolabeled_$template_id \
          --seed $SEED \
          --tag $TAG \
          --mapping $MAPPING \
          --max_steps $MAX_STEP \
          --template_path spoilers_auto_template/irrelevant_relevant/16-$seed.txt \
          --template_id $template_id \
          --first_sent_limit 502 \
          --other_sent_limit 502
    done
done
