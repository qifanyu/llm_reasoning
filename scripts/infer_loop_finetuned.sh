model_name=CauchyLovesU/LLM-reasoning-loop-finetuned
# loop-finetuning on mistral
model_name=WuRen123/mistral-loop-finetuned

python3 infer_mp.py \
    --model_name $model_name \
    --test_size 2000 --batch_size 32 \
    --dp_size 8 \
    --dataset_name {gsm8k,primary}

echo "model: $model_name"

# loop finetuned
# gsm8k: 82.26%
# primary: 74.30%
