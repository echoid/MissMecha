
import missmecha.analysis.visual as mmv
import missmecha.generate as mmg

import pandas as pd
import numpy as np

collisions = pd.read_csv("https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv")

data = collisions.sample(250, random_state=42)  # ✅ Keeps the random seed

df = pd.read_csv("_data.txt", delimiter=",", header=None)

# #mmv.matrix(df,color=False)
# mmv.matrix(data,color=True)
# mmv.matrix(data,color=False)
# mmv.heatmap(data)



# 使用示例
if __name__ == "__main__":
    # 生成测试数据
    data = pd.DataFrame({
        'age': np.random.randint(18, 65, 100),
        'income': np.random.normal(50000, 15000, 100),
        #'department': np.random.choice(['HR', 'IT', 'Finance'], 100)
    })
    
    # 定义混合机制
    mechanisms = [
        mmg.MCARMechanism(target_cols=['age'], missing_rate=0.2),
        mmg.MARMechanism(
            target_cols=['income'],
            depend_cols=['department'],
            model_type='logistic',
            data_type='categorical',
            missing_rate=0.3
        ),
        mmg.MNARMechanism(
            target_cols=['department'],
            threshold=0.5,  # 假设department被编码后阈值
            direction='above',
            data_type='categorical'
        )
    ]
    

    mix_generator = mmg.MixedMechanism(mechanisms)
    missing_data = mix_generator.apply(data)
    
    print("Generated missing data:")
    print(missing_data.head())


        # 生成测试数据
    # cat_data = pd.DataFrame({
    #     'department': np.random.choice(['HR', 'IT', 'Finance'], 1000),
    #     'job_level': np.random.choice(['Junior', 'Senior'], 1000)
    # })

    # 初始化生成器
    mar_gen = mmg.MissingGenerator(
        mech_type="MAR",
        target_cols=['age'],
        #data_type="categorical",
        depend_cols=['income'],  # MAR特有参数
        model_type="logistic",
        missing_rate=0.4
    )

    # 应用缺失
    mar_data = mar_gen.apply(data)
