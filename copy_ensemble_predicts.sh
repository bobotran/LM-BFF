#!/bin/bash
for template_id in {44..79}
do
    cp result/$template_id/spoilers--1--1-eval.npy ensemble_predict_results/prompt/val/spoilers-$template_id-eval.npy
    cp result/$template_id/spoilers--1--1-test.npy ensemble_predict_results/prompt/test/spoilers-$template_id-test.npy
done