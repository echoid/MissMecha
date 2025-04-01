import pandas as pd
import numpy as np
# from missmecha.generator import generate_missing
from missmecha.analysis import compute_missing_rate
df_cate = pd.DataFrame({
    "age": [25, 30, np.nan, 40],
    "income": [3000, np.nan, 2800, 5200],
    "gender": ["M", "F", np.nan, "F"]
})


import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 1000

# Generate synthetic data
ages = np.random.choice([25, 30, 35, 40, 45], size=n_samples, p=[0.2, 0.3, 0.2, 0.2, 0.1])
incomes = np.random.normal(loc=4000, scale=1000, size=n_samples).astype(int)
incomes = np.clip(incomes, 2000, 8000)  # Clamp to a realistic range
genders = np.random.choice([0, 1], size=n_samples)

# Create DataFrame
df_cate = pd.DataFrame({
    "age": ages,
    "income": incomes,
    "gender": genders
})




# data_num = np.random.default_rng(1).normal(loc=0.0, scale=1.0, size=(1000, 10))






import numpy as np
from sklearn.model_selection import train_test_split
from missmecha import MissMechaGenerator
from missmecha.analysis import compute_missing_rate,MCARTest

# 生成完整数值型数据
data_num = np.random.default_rng(1).normal(loc=0.0, scale=1.0, size=(1000, 10))


# 拆分训练集和测试集
X_train, X_test = train_test_split(data_num, test_size=0.3, random_state=42)




from missmecha.generator import MissMechaGeneratorMultiple


info = {
    ("age", "income"): {"mechanism": "mcar", "type": 1, "rate": 0.5},
    "gender": {"mechanism": "mnar", "type": 5, "rate": 0.1}
}

generator = MissMechaGeneratorMultiple(info=info)
generator.fit(df_cate)
df_missing = generator.transform(df_cate)

compute_missing_rate(df_missing)
MCARTest(method="little")(df_missing)














# missing_type_list = ["mcar","mar","mnar"]
# missing_rate_list = [0.1,0.3,0.5]
# mechanism_type_list = [1,2,3,4,5,6,7,8]

# missing_type_list = ["mcar"]
# mechanism_type_list = [1,2,3]

# # missing_type_list = ["mnar"]
# # mechanism_type_list = [1,2,3,4,5,6]

# # missing_type_list = ["mar"]
# # mechanism_type_list = [1,2,3,4,5,6,7,8]



# for missing_type in missing_type_list:
#     for missing_rate in missing_rate_list:
#         for mechanism_type in mechanism_type_list:
#             print(" | mechanism_type: ",mechanism_type, " | missing_type: ",missing_type," | missing_rate: ",missing_rate,)
#             mecha = MissMechaGenerator(
#                 mechanism=missing_type,            # or "mnar", "mcar"
#                 mechanism_type=mechanism_type,           # type id
#                 missing_rate=missing_rate,           # 缺失比例
#             )

#             X_train_missing = mecha.fit_transform(data_num)
#             # 输出缺失比例
#             print("=========================>Train Fit_transform=========================>")
#             compute_missing_rate(X_train_missing)
            
#             # Little's test
#             pval = MCARTest(method="little")(X_train_missing)
#             print(f"Little’s MCAR test p-value: {pval}")
#             print("=========================END=========================>")
#             # # T-test matrix
#             # ttest_df = MCARTest(method="ttest")(X_train_missing)
#             # print(ttest_df)
#             # print("=========================>")
#             # print("=========================>")
#             # print("=========================>")
#             # print("=========================>")
#             # print("=========================>")

#             # print("=========================>Train Missing Rate=========================>")
#             # X_train_missing = mecha.fit(X_train)
#             # X_train_missing = mecha.transform(X_train)
#             # print(mecha.get_mask())
#             # compute_missing_rate(X_train_missing)


#             # print("=========================>Test Missing Rate=========================>")
#             # X_test_missing = mecha.transform(X_test)
#             # print(mecha.get_mask())
#             # compute_missing_rate(X_test_missing)




