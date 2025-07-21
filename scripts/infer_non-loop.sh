# baseline
model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# non-loop finetuned
model_name=CauchyLovesU/LLM-reasoning-Non-loop-finetuned

python3 infer_mp.py \
    --model_name $model_name \
    --num_loops 1 \
    --test_size 2000 --batch_size 32 \
    --dp_size 8 \
    --dataset_name {gsm8k,primary}

echo "model: $model_name"

# baseline
# gsm8k: 70.89%
# primary: 47.30%

# non-loop finetuned
# gsm8k: 77.71%
# primary: 58.60%