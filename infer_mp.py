from multiprocessing import Process, Queue
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle
from datasets import load_dataset, Dataset
from tqdm import tqdm
from src.config import parse_args
import json

from IPython import embed


def generate_answer(
    model,
    tokenizer,
    batch,
    device,
    max_length=8,
    num_loops=3,
    alpha=0.001,
    dataset_name='gsm8k',
    add_cot_prompt=False,
    cot=True
):
    if add_cot_prompt:
        cot_prompt = f"Please solve the problem step by step and enclose the final answer in \\boxed{{}}.\n\n" if cot else "Please output the final answer directly: "
    else:
        cot_prompt = ""

    prompts = [f"<｜User｜>{cot_prompt}{question}\n<｜Assistant｜>" for question in batch['question']]
    batch_inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)
    
    input_ids = batch_inputs.input_ids
    attention_mask = batch_inputs.attention_mask
    
    batch_generated_ids = input_ids.clone()
    is_finished = torch.zeros(input_ids.size(0), dtype=torch.bool).to(device)

    for _ in range(max_length):
        if is_finished.all():
            break

        current_input_ids = batch_generated_ids[~is_finished]
        current_attention_mask = attention_mask[~is_finished]

        with torch.no_grad():
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                output_hidden_states=True,
            )
            original_hidden_states = outputs.hidden_states[-1]
            hidden_states = original_hidden_states.clone()

        for _ in range(num_loops - 1):
            with torch.no_grad():
                outputs = model(
                    inputs_embeds=hidden_states,
                    attention_mask=current_attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]

        if num_loops > 1:
            combined_states = original_hidden_states + alpha * hidden_states
        else:
            combined_states = original_hidden_states

        logits = model.lm_head(combined_states)

        next_tokens = torch.argmax(logits[:, -1, :], dim=-1)

        current_idx = 0
        next_tokens_ = torch.zeros((input_ids.size(0), 1), dtype=torch.long).to(device)
        for j in range(len(is_finished)):
            if not is_finished[j]:
                next_tokens_[j][0] = next_tokens[current_idx]
                if next_tokens[current_idx] == tokenizer.eos_token_id:
                    is_finished[j] = True
                current_idx += 1
        batch_generated_ids = torch.cat([batch_generated_ids, next_tokens_], dim=-1)

        attention_mask = torch.cat(
            [attention_mask, torch.ones((input_ids.size(0), 1)).to(device)], dim=-1
        )

    batch_answers = [
        tokenizer.decode(generated_id, skip_special_tokens=True)
        for generated_id in batch_generated_ids
    ]

    return batch_answers


def extract_answer(text):
    # 去除 LaTeX 中的空格控制命令（如 \!, \,, \; 等）
    cleaned = re.sub(r'\\[,\!\;\:\s]', '', text)

    # 匹配 \boxed{...} 中的内容（允许负号、逗号、数字）
    match = re.search(r'\\boxed\{(-?[\d,\.]+)\}', cleaned)
    if not match:
        # 否则提取最后一个数字（允许负号、逗号、小数）
        match = re.search(r'(-?[\d,\.]+)(?=\D*$)', cleaned)
    if match:
        return match.group(1).replace(',', '')  # 去除千位分隔逗号
    return None


def main(dp_rank, result_queue):
    args = parse_args()
    dp_size = args.dp_size
    dataset_name = args.dataset_name
    device = torch.device(f"cuda:{dp_rank}" if torch.cuda.is_available() else "cpu")

    cache_dir = '/mnt/bn/qifan-nas/cache'
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", padding_side='left', cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    ).eval().to(device)

    alpha = 0.001

    # Load dataset
    if dataset_name == 'gsm8k':
        full_dataset = load_dataset("gsm8k", "main")["test"]
        full_dataset = full_dataset.select(range(min(args.test_size, len(full_dataset))))
    elif dataset_name == 'primary':
        with open(args.dataset_path, 'r') as f:
            data = json.load(f)
        full_data = Dataset.from_list(data)
        full_dataset = full_data.select(range(min(args.test_size, len(full_data))))
    else:
        raise NotImplementedError

    dataset = full_dataset.shard(num_shards=dp_size, index=dp_rank)

    correct = 0
    total = len(dataset)
    batch_size = args.batch_size

    for i in tqdm(range(0, total, batch_size), desc=f"Rank {dp_rank} evaluating"):
        batch = dataset.select(range(i, min(i+batch_size, total)))
        responses = generate_answer(model, tokenizer, batch, device, max_length=1024, num_loops=args.num_loops, alpha=alpha, dataset_name=dataset_name, add_cot_prompt=args.add_cot_prompt, cot=args.cot_mode)

        for idx, response in enumerate(responses):
            if dataset_name == 'gsm8k':
                gt_answer = extract_answer(batch["answer"][idx])
            elif dataset_name == 'primary':
                gt_answer = batch['answer'][idx]
            else:
                raise NotImplementedError
            pred_answer = extract_answer(response)
            
            is_correct = False
            if gt_answer and pred_answer:
                try:
                    if abs(float(pred_answer) - float(gt_answer)) < 1e-5:
                        correct += 1
                        is_correct = True
                except:
                    pass
        
    print(f"[Rank {dp_rank}] Correct: {correct} / {total}")
    result_queue.put((correct, total))  # 发送结果给主进程

if __name__ == "__main__":
    args = parse_args()
    dp_size = args.dp_size

    result_queue = Queue()
    if dp_size > 1:
        procs = [
            Process(target=main, args=(dp_rank, result_queue))
            for dp_rank in range(dp_size)
        ]

        for p in procs:
            p.start()
        for p in procs:
            p.join()
    else:
        main(0, result_queue)

    total_correct = 0
    total_count = 0
    for _ in range(dp_size):
        correct, total = result_queue.get()
        total_correct += correct
        total_count += total

    acc = total_correct / total_count * 100
    print(f"[Global] Total Accuracy: {acc:.2f}% ({total_correct} / {total_count})")