import configargparse

def parse_args():
    parser = configargparse.ArgumentParser(description='Finetune looped model on deepseek backbone')

    # Model Configuration
    parser.add_argument('--model_name', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', help='Model name or path.')
    parser.add_argument('--num_loops', type=int, default=3, help='Number of loops.')

    # Data Configuration
    parser.add_argument('--dataset_name', type=str, default='gsm8k', help='Dataset name or path.')
    parser.add_argument('--dataset_path', type=str, default='data/primary/arithmetic_dataset_test.json', help='Dataset name or path.')
    parser.add_argument('--test_size', type=int, default=1000, help='Number of test samples.')
    parser.add_argument('--add_cot_prompt', action='store_true', help='Add COT prompt.')
    parser.add_argument('--cot_mode', action='store_true', help='Add COT prompt.')

    # Running Configuration
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--dp_size', type=int, default=1, help='Data parallel size.')

    args = parser.parse_args()
    return args