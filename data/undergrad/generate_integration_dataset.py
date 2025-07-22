import json
import random
from fractions import Fraction

def format_polynomial(poly_dict):
    """
    将表示多项式的字典格式化为人类可读的字符串。
    例如 {2: 3, 1: -2, 0: 5} -> "3x^2 - 2x + 5"

    Args:
        poly_dict (dict): 一个字典，键是指数，值是系数。

    Returns:
        str: 格式化后的多项式字符串。
    """
    if not poly_dict:
        return "0"

    terms = []
    # 按指数从大到小排序，以构建标准形式的多项式
    sorted_powers = sorted(poly_dict.keys(), reverse=True)

    for i, power in enumerate(sorted_powers):
        coeff = poly_dict[power]
        
        # --- 系数处理 ---
        # 如果系数是分数，则格式化为 a/b 的形式
        if isinstance(coeff, Fraction):
            if coeff.denominator == 1:
                coeff_str = str(coeff.numerator)
            else:
                # 如果分子是负数，将负号提到分数前面
                if coeff.numerator < 0:
                    coeff_str = f"-{abs(coeff.numerator)}/{coeff.denominator}"
                else:
                    coeff_str = f"{coeff.numerator}/{coeff.denominator}"
        else:
            coeff_str = str(coeff)

        # 隐藏系数为 1 或 -1 的情况 (除非是常数项)
        if abs(coeff) == 1 and power != 0:
            coeff_str = '-' if coeff == -1 else ''
        
        # --- 变量和指数处理 ---
        if power > 1:
            term_str = f"{coeff_str}x^{power}"
        elif power == 1:
            term_str = f"{coeff_str}x"
        else: # power == 0 (常数项)
            term_str = coeff_str
        
        # --- 拼接项 ---
        if i == 0:
            # 第一个项直接添加
            terms.append(term_str)
        else:
            # 后续项根据系数正负添加 "+" 或 "-"
            # 如果系数是分数，检查其分子
            if (isinstance(coeff, Fraction) and coeff.numerator > 0) or (isinstance(coeff, int) and coeff > 0):
                terms.append(f"+ {term_str}")
            else:
                # 对于负数，format_polynomial 会处理负号，所以我们只需要一个空格
                # 例如 "-2x" 而不是 "- 2x"
                # 我们通过移除自动生成的负号再重新拼接来标准化格式
                terms.append(f"- {term_str.replace('-', '')}")

    return ' '.join(terms)

def generate_polynomial(max_degree):
    """
    生成一个随机多项式。

    Args:
        max_degree (int): 多项式的最大次数。

    Returns:
        dict: 表示多项式的字典。
    """
    # 随机决定多项式的实际次数，至少为0次
    degree = random.randint(0, max_degree)
    
    polynomial = {}
    for power in range(degree, -1, -1):
        # 生成一个非零的随机整数系数
        # 确保最高次项的系数不为零
        if power == degree:
            coeff = random.choice([i for i in range(-10, 11) if i != 0])
        else:
            # 其他项的系数可以为0，但我们在这里选择非零系数来创建更“密集”的多项式
            # 如果想允许有缺项，可以包含0
            coeff = random.choice([i for i in range(-10, 11) if i != 0])
        
        if coeff != 0:
            polynomial[power] = coeff
            
    # 如果随机过程最终没有生成任何项（例如所有系数都碰巧为0），则返回一个常数项
    if not polynomial:
        polynomial[0] = random.choice([i for i in range(1, 11)])

    return polynomial

def integrate_polynomial(poly_dict):
    """
    计算多项式的不定积分。

    Args:
        poly_dict (dict): 表示多项式的字典。

    Returns:
        dict: 表示积分后多项式的字典，使用 Fraction 对象以保持精度。
    """
    integrated_poly = {}
    if not poly_dict:
        return {}

    for power, coeff in poly_dict.items():
        new_power = power + 1
        # 使用 Fraction 来进行精确的除法运算
        new_coeff = Fraction(coeff, new_power)
        integrated_poly[new_power] = new_coeff
        
    return integrated_poly

def generate_dataset(num_samples, max_degree):
    """
    生成整个问答数据集。

    Args:
        num_samples (int): 要生成的数据样本数量。
        max_degree (int): 多项式的最大次数。

    Returns:
        list: 一个包含问答对字典的列表。
    """
    dataset = []
    for _ in range(num_samples):
        # 1. 生成原始多项式
        poly = generate_polynomial(max_degree)
        
        # 2. 将其格式化为问题字符串
        question = format_polynomial(poly)
        
        # 3. 计算其不定积分
        integral_poly = integrate_polynomial(poly)
        
        # 4. 将积分结果格式化为答案字符串，并添加积分常数 "C"
        answer = format_polynomial(integral_poly) + " + C"
        
        dataset.append({"question": question, "answer": answer})
        
    return dataset

if __name__ == "__main__":
    # --- 参数配置 ---
    MAX_DEGREE = 10  # 生成多项式的最大次数
    NUM_SAMPLES = 10000 # 要生成的数据集大小
    OUTPUT_FILE = "polynomial_integration_dataset.json" # 输出文件名

    print(f"正在生成 {NUM_SAMPLES} 个样本...")
    
    # 生成数据集
    data = generate_dataset(NUM_SAMPLES, MAX_DEGREE)
    
    # 将数据集写入 JSON 文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # indent=4 使 JSON 文件格式优美，易于阅读
        # ensure_ascii=False 确保中文字符能正确显示
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    print(f"数据集已成功生成并保存到文件: {OUTPUT_FILE}")