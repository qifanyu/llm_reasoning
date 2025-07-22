# polynomial_problem_generator.py
import pandas as pd
import numpy as np
import sympy as sp
import argparse
import random

def generate_polynomial(max_degree, num_terms):
    """
    根据给定的最大次数和项数，生成一个随机多项式。
    """
    x = sp.Symbol('x')
    polynomial = 0
    num_terms = min(num_terms, max_degree)
    degrees = random.sample(range(1, max_degree + 1), num_terms)
    
    for degree in degrees:
        coeff = random.choice([i for i in range(-10, 11) if i != 0])
        polynomial += coeff * (x**degree)
    
    constant = random.randint(-10, 11)
    polynomial += constant
    return polynomial

def differentiate_polynomial(poly_expr, order):
    """
    对给定的 SymPy 表达式进行求导。
    """
    x = sp.Symbol('x')
    return sp.diff(poly_expr, x, order)

def format_expression(expr):
    """将 SymPy 表达式格式化为更易读的字符串。"""
    return str(expr).replace('**', '^').replace('*', ' * ')

def main(args):
    """
    主函数，用于生成问题并保存文件。
    """
    problems = []
    answers = []
    
    print(f"开始生成 {args.num_problems} 条题目...")
    print(f"参数设置: max_degree={args.max_degree}, max_terms={args.max_terms}, max_derivative_order={args.max_derivative_order}")

    for i in range(args.num_problems):
        num_terms = random.randint(2, args.max_terms)
        derivative_order = random.randint(1, args.max_derivative_order)
        
        min_degree_for_poly = max(num_terms, derivative_order)
        max_degree_for_poly = max(min_degree_for_poly, args.max_degree)
        
        polynomial_expr = generate_polynomial(max_degree=max_degree_for_poly, num_terms=num_terms)
        
        poly_str = format_expression(polynomial_expr)
        if derivative_order == 1:
            problem_str = f"求函数 f(x) = {poly_str} 的导数 f'(x)."
        else:
            problem_str = f"求函数 f(x) = {poly_str} 的 {derivative_order} 阶导数."

        answer_expr = differentiate_polynomial(polynomial_expr, order=derivative_order)
        answer_str = format_expression(answer_expr)
        
        problems.append(problem_str)
        answers.append(answer_str)
        
        if (i + 1) % 1000 == 0:
            print(f"已生成 {i + 1}/{args.num_problems} 条...")
            
    df = pd.DataFrame({'problem': problems, 'answer': answers})
    
    try:
        df.to_parquet(args.output_file, engine='pyarrow')
        print(f"\n成功生成 {args.num_problems} 条题目和答案！")
        print(f"数据已保存到文件: '{args.output_file}'")
        print("\n前5条题目示例:")
        print(df.head().to_string())
    except Exception as e:
        print(f"\n保存文件时发生错误: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="生成高中多项式求导题目和答案，并保存为 Parquet 文件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '-n', '--num_problems', type=int, default=10000,
        help="要生成的题目总数。\n默认值: 10000"
    )
    
    parser.add_argument(
        '-o', '--output_file', type=str, default='polynomial_derivatives.parquet',
        help="输出的 Parquet 文件名。\n默认值: 'polynomial_derivatives.parquet'"
    )
    
    parser.add_argument(
        '--max_degree', type=int, default=8,
        help="多项式中可能出现的最高次数。\n默认值: 8"
    )

    parser.add_argument(
        '--max_terms', type=int, default=15,
        help="每个多项式中最多包含的项数（不含常数项）。\n默认值: 15"
    )

    parser.add_argument(
        '--max_derivative_order', type=int, default=3,
        help="求导的最高阶数。\n默认值: 3"
    )

    args = parser.parse_args()
    
    try:
        import pandas
        import sympy
        import pyarrow
    except ImportError as e:
        print(f"错误: 缺少必要的库 -> {e.name}")
        print("请运行: pip install pandas sympy pyarrow numpy")
    else:
        main(args)