# test.py (放置在项目根目录 MissMecha/ 下)
import sys
import os
import numpy as np
import pandas as pd

# 临时添加项目路径到Python路径（无需安装包）
sys.path.append(os.path.abspath("."))      # 添加项目根目录
sys.path.append(os.path.abspath("./src"))  # 添加src目录

try:
    from mechamiss.generators import MCARGenerator, MNARGenerator
    from mechamiss.mistypes import MistypeInjector
    print("✅ 模块导入成功!")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请检查：")
    print("1. 项目结构是否为 MissMecha/src/mechamiss/...")
    print("2. 是否在项目根目录运行测试（例如 F:/Deakin/MissMecha/）")
    sys.exit(1)

def main():
    # 生成测试数据
    data = pd.DataFrame({
        'age': np.random.randint(18, 65, 100),
        'income': np.random.normal(50000, 15000, 100).astype(int),
        'department': np.random.choice(['HR', 'IT', 'Finance'], 100)
    })
    
    print("\n=== 测试1: MCAR生成器 ===")
    mcar_gen = MCARGenerator(missing_col='income', missing_rate=0.3)
    mcar_data = mcar_gen.generate(data)
    missing_percent = mcar_data['income'].isna().mean()
    print(f"理论缺失率: 30% | 实际缺失率: {missing_percent:.1%}")
    assert abs(missing_percent - 0.3) < 0.05, "MCAR缺失率偏差超过5%"
    
    print("\n=== 测试2: MNAR生成器 ===")
    mnar_gen = MNARGenerator(missing_col='income', threshold=70000, direction='above')
    mnar_data = mnar_gen.generate(data)
    high_income_missing = mnar_data.loc[data['income'] > 70000, 'income'].isna().mean()
    print(f"高收入组缺失率: {high_income_missing:.0%} (应接近100%)")
    assert high_income_missing > 0.95, "MNAR机制未生效"
    
    print("\n=== 测试3: 错误注入器 ===")
    error_gen = MistypeInjector(columns=['age'], error_rate=0.2)
    error_data = error_gen.inject(data)
    error_samples = error_data.sample(5)
    print("注入错误示例:")
    print(error_samples[['age']].to_string())
    
    # 验证错误类型
    num_errors = error_data['age'].apply(lambda x: isinstance(x, str)).sum()
    print(f"检测到 {num_errors} 处类型错误注入")
    assert num_errors > 10, "错误注入数量不足"

if __name__ == "__main__":
    main()
    print("\n🔥 所有基础测试通过！")