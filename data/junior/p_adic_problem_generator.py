import pandas as pd
import random
import argparse

def to_base_p_string(n: int, base: int) -> str:
    """
    将一个十进制整数转换为P进制的字符串表示形式。
    支持负数和0。

    Args:
        n (int): 需要转换的十进制数。
        base (int): 目标进制 (2-16)。

    Returns:
        str: P进制表示的字符串。
    """
    if n == 0:
        return '0'
    if n < 0:
        return '-' + to_base_p_string(-n, base)

    digits = "0123456789ABCDEF"
    result = ""
    while n > 0:
        remainder = n % base
        result = digits[remainder] + result
        n //= base
    return result

def generate_p_adic_problems(num_problems: int, bases_to_use: list[int], max_operands: int) -> pd.DataFrame:
    """
    根据指定参数生成P进制加减法题目和答案。

    Args:
        num_problems (int): 要生成的题目数量。
        bases_to_use (list[int]): 用于生成题目的进制列表。
        max_operands (int): 每道题目的最大操作数。

    Returns:
        pd.DataFrame: 包含'base', 'problem', 'answer'三列的DataFrame。
    """
    problem_list = []

    for _ in range(num_problems):
        base = random.choice(bases_to_use)
        num_operands = random.randint(2, max_operands)

        operands_in_p = []
        operands_in_10 = []

        for i in range(num_operands):
            if base == 16:
                max_val = 4095
            elif base == 10:
                max_val = 1000
            elif base == 8:
                max_val = 511
            else: # base == 2
                max_val = 255
            
            if i == 0:
                val_10 = random.randint(max_val // 2, max_val)
            else:
                val_10 = random.randint(1, max_val)
            
            operands_in_10.append(val_10)
            operands_in_p.append(to_base_p_string(val_10, base))

        # --- MODIFIED SECTION ---
        # 4. 生成简化的运算表达式并计算结果
        # 直接使用操作数本身，不加括号和角标
        problem_statement = operands_in_p[0]
        result_in_10 = operands_in_10[0]

        for i in range(1, num_operands):
            op = random.choice(['+', '-'])
            # 拼接 "运算符" 和 "操作数"
            problem_statement += f" {op} {operands_in_p[i]}"

            # 实时计算十进制结果
            if op == '+':
                result_in_10 += operands_in_10[i]
            else:
                result_in_10 -= operands_in_10[i]
        
        final_answer = to_base_p_string(result_in_10, base)
        # --- END OF MODIFIED SECTION ---

        problem_list.append({
            'problem': f"求解{base}进制运算：" + problem_statement,
            'answer': final_answer
        })

    return pd.DataFrame(problem_list)

def main():
    """主函数，用于解析命令行参数并执行程序。"""
    parser = argparse.ArgumentParser(
        description="生成P进制 (base-P) 加减法题目，并保存为 Parquet 文件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '-n', '--num-problems',
        type=int,
        default=10000,
        help="要生成的题目总数量。\n(默认: 10000)"
    )
    
    parser.add_argument(
        '-o', '--output-file',
        type=str,
        default='p_adic_math_problems.parquet',
        help="输出 Parquet 文件的路径和名称。\n(默认: p_adic_math_problems.parquet)"
    )
    
    parser.add_argument(
        '-b', '--bases',
        type=int,
        nargs='+',
        default=[2, 8, 10, 16],
        help="用于生成题目的进制列表 (例如, --bases 2 8 16)。\n(默认: 2 8 10 16)"
    )

    parser.add_argument(
        '-m', '--max-operands',
        type=int,
        default=15,
        help="每道题目中允许的最大操作数数量 (必须>=2)。\n(默认: 15)"
    )
    
    args = parser.parse_args()

    if args.max_operands < 2:
        parser.error("--max-operands 的值必须大于或等于 2。")
    for base in args.bases:
        if not 2 <= base <= 16:
            parser.error(f"无效的进制: {base}。进制必须在 2 到 16 之间。")

    print(f"▶️  开始生成 {args.num_problems} 条题目...")
    print(f"    - 使用进制: {args.bases}")
    print(f"    - 每题最大操作数: {args.max_operands}")
    
    df_problems = generate_p_adic_problems(
        num_problems=args.num_problems,
        bases_to_use=args.bases,
        max_operands=args.max_operands
    )

    try:
        df_problems.to_parquet(args.output_file, index=False, engine='pyarrow')
        print(f"\n✅ 成功！已将 {len(df_problems)} 条题目和答案保存到 '{args.output_file}'。")
        
        print("\n📄 文件内容预览 (已采用新的简化格式):")
        print(df_problems.head())
        
    except Exception as e:
        print(f"\n❌ 保存文件时出错: {e}")
        print("   请确保您已通过 'pip install pandas pyarrow' 安装了必要的库。")

if __name__ == '__main__':
    main()