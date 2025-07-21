# -*- coding: utf-8 -*-

import json
import random

class ArithmeticExpressionGenerator:
    """
    一个用于生成随机四则运算表达式的类。

    该类可以生成包含加、减、乘和括号的数学表达式。
    可以控制表达式的复杂程度（通过操作数的数量），并确保生成的表达式语法正确。
    """

    def __init__(self, min_operands=2, max_operands=5):
        """
        初始化生成器。

        Args:
            min_operands (int): 表达式中最少包含的操作数（数字）数量。
            max_operands (int): 表达式中最多包含的操作数（数字）数量。
        """
        if min_operands < 2 or max_operands < min_operands:
            raise ValueError("操作数范围设置不合法，必须满足 min_operands >= 2 且 max_operands >= min_operands")
        
        self.min_operands = min_operands
        self.max_operands = max_operands
        self.operators = ['+', '-', '*']

    def _generate_number(self):
        """
        生成一个随机整数作为操作数。
        范围是1到100。
        """
        return random.randint(1, 100)

    def _generate_recursive(self, num_operands):
        """
        使用递归方法生成表达式。

        这个方法是生成器的核心。它通过将操作数数量一分为二，
        并为左右两边递归生成子表达式，然后用一个随机运算符连接它们，
        从而构建出整个表达式。

        Args:
            num_operands (int): 当前表达式需要包含的操作数数量。

        Returns:
            str: 生成的子表达式字符串。
        """
        # 基本情况：如果只有一个操作数，直接返回一个数字
        if num_operands == 1:
            return str(self._generate_number())

        # 递归步骤：将操作数拆分为两部分
        # 确保左右两边至少各有一个操作数
        left_operands = random.randint(1, num_operands - 1)
        right_operands = num_operands - left_operands

        # 递归生成左右子表达式
        left_expr = self._generate_recursive(left_operands)
        right_expr = self._generate_recursive(right_operands)

        # 为子表达式随机添加括号，使其结构更多样化
        # 如果子表达式本身包含多个操作数，它就有可能被加上括号
        if left_operands > 1 and random.random() < 0.5:
            left_expr = f"({left_expr})"
        
        if right_operands > 1 and random.random() < 0.5:
            right_expr = f"({right_expr})"

        # 选择一个随机运算符
        operator = random.choice(self.operators)
        
        # 组合成最终的表达式
        return f"{left_expr} {operator} {right_expr}"

    def generate_expression(self):
        """
        生成一个完整的、随机长度的表达式及其计算结果。

        Returns:
            tuple: 一个元组，包含两个元素：
                   - question (str): 生成的四则运算表达式。
                   - answer (int): 表达式的计算结果。
        """
        # 随机确定当前表达式的操作数数量
        num_operands = random.randint(self.min_operands, self.max_operands)
        
        # 生成表达式字符串
        expression_body = self._generate_recursive(num_operands)
        
        try:
            # 使用 eval() 计算表达式的结果。
            # 因为表达式是我们自己生成的，所以使用 eval 是安全的。
            # 为了美观，将表达式中的空格去掉后再计算
            answer = eval(expression_body.replace(" ", ""))
            question = f"{expression_body} ="
            return question, answer
        except (SyntaxError, ZeroDivisionError) as e:
            # 正常情况下，我们的生成逻辑不会产生非法表达式，但作为保险措施
            print(f"生成了错误的表达式: {expression_body}, 错误: {e}")
            return None, None


def create_dataset(generator, num_samples, output_path):
    """
    创建数据集并将其保存为JSON文件。

    Args:
        generator (ArithmeticExpressionGenerator): 表达式生成器实例。
        num_samples (int): 要生成的数据样本数量。
        output_path (str): 输出的JSON文件路径。
    """
    dataset = []
    print(f"开始生成 {num_samples} 条数据...")
    
    while len(dataset) < num_samples:
        question, answer = generator.generate_expression()
        if question is not None:
            dataset.append({
                "question": question,
                "answer": answer
            })
            if (len(dataset) % 100) == 0:
                print(f"已生成 {len(dataset)} / {num_samples} 条")

    print(f"数据生成完毕，正在写入文件: {output_path}")
    
    # 将数据集写入JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
        
    print("文件写入成功！")


if __name__ == '__main__':
    # --- 配置参数 ---

    # 1. 设置表达式的复杂度（通过操作数的数量）
    # min_operands: 算式中最少有几个数字
    # max_operands: 算式中最多有几个数字
    # 例如，min=2, max=4 会生成像 "a+b", "a*(b-c)", "(a+b)*(c-d)" 等复杂度的算式
    MIN_OPERANDS = 6
    MAX_OPERANDS = 10

    # 2. 要生成的数据集大小
    NUM_SAMPLES = 2000

    # 3. 输出文件名
    OUTPUT_FILE = "arithmetic_dataset.json"

    # --- 执行 ---
    
    # 创建生成器实例
    expression_generator = ArithmeticExpressionGenerator(
        min_operands=MIN_OPERANDS,
        max_operands=MAX_OPERANDS
    )

    # 创建并保存数据集
    create_dataset(expression_generator, NUM_SAMPLES, OUTPUT_FILE)

    # --- 读取并验证一个样本 ---
    print("\n--- 读取生成的文件并展示一个样本 ---")
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if data:
            sample = random.choice(data)
            print(f"问题: {sample['question']}")
            print(f"答案: {sample['answer']}")
