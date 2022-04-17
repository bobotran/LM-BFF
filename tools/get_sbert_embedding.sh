MODEL=$1
K=16

python tools/get_sbert_embedding.py --sbert_model $MODEL --task spoilers
python tools/get_sbert_embedding.py --sbert_model $MODEL --seed 42 --do_test --task spoilers

for seed in 13 21 87 100
do
    for task in spoilers
    do
        cp data/k-shot-10x/$task/$K-42/test_sbert-$MODEL.npy  data/k-shot-10x/$task/$K-$seed/
    done

    # cp data/k-shot-10x/MNLI/$K-42/test_matched_sbert-$MODEL.npy  data/k-shot-10x/MNLI/$K-$seed/
    # cp data/k-shot-10x/MNLI/$K-42/test_mismatched_sbert-$MODEL.npy  data/k-shot-10x/MNLI/$K-$seed/
done
