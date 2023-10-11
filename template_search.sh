#!/bin/bash
for template_id in {80..93}
do
    # for seed in 13 21 42 87 100
    for seed in 21
    do
        # To save time, we fix these hyper-parameters
        bs=32
        lr=1e-5

        # Since we only use dev performance here, use --no_predict to skip testing
        TAG=irrelevant-relevant-template-v5 \
        TYPE=prompt \
        TASK=spoilers \
        BS=$bs \
        LR=$lr \
        SEED=$seed \
        MODEL=roberta-large \
        TEMPLATE_ID=$template_id \
        bash run_experiment.sh "--template_path spoilers_auto_template/irrelevant_relevant/16-$seed.txt --template_id $template_id --mapping {0:'relevant',1:'irrelevant'} --first_sent_limit 502 --other_sent_limit 502"
    done
done
