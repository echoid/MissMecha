import pandas as pd
import numpy as np
# from missmecha.generator import generate_missing
from missmecha.analysis import compute_missing_rate
# df_cate = pd.DataFrame({
#     "age": [25, 30, np.nan, 40],
#     "income": [3000, np.nan, 2800, 5200],
#     "gender": ["M", "F", np.nan, "F"]
# })


# data_num = np.random.default_rng(1).normal(loc=0.0, scale=1.0, size=(1000, 10))


# # Define info dictionary for generate()
# info = {
#     "rate": 0.2,  # 20% missingness
#     "type": 1     # Use type_two generator
# }

# type_list = [5,6]
# missing_rate_list = [0,0.1,0.5,0.9]

# for type_val in type_list:
#     for rate in missing_rate_list:
#         print("======================")
#         print("Missing type", type_val,"Missing Rate", rate)
#         result = generate_missing(data_num, missing_type="mnar", type=type_val, missing_rate=rate)
#         compute_missing_rate(result["data"])
#         print(result["mask_int"])
#         print("======================")

# print("Try MissMechaGenerator")



import numpy as np
from sklearn.model_selection import train_test_split
from missmecha import MissMechaGenerator
from missmecha.analysis import compute_missing_rate,MCARTest

# 生成完整数值型数据
data_num = np.random.default_rng(1).normal(loc=0.0, scale=1.0, size=(1000, 10))

# 拆分训练集和测试集
X_train, X_test = train_test_split(data_num, test_size=0.3, random_state=42)

missing_type_list = ["mcar","mar","mnar"]
missing_rate_list = [0.1,0.3,0.5]
mechanism_type_list = [1,2,3,4,5,6,7,8]

missing_type_list = ["mcar"]
mechanism_type_list = [1,2,3]

# missing_type_list = ["mnar"]
# mechanism_type_list = [1,2,3,4,5,6]

# missing_type_list = ["mar"]
# mechanism_type_list = [1,2,3,4,5,6,7,8]



for missing_type in missing_type_list:
    for missing_rate in missing_rate_list:
        for mechanism_type in mechanism_type_list:
            print(" | mechanism_type: ",mechanism_type, " | missing_type: ",missing_type," | missing_rate: ",missing_rate,)
            mecha = MissMechaGenerator(
                mechanism=missing_type,            # or "mnar", "mcar"
                mechanism_type=mechanism_type,           # type id
                missing_rate=missing_rate,           # 缺失比例
            )

            X_train_missing = mecha.fit_transform(data_num)
            # 输出缺失比例
            print("=========================>Train Fit_transform=========================>")
            compute_missing_rate(X_train_missing)
            
            # Little's test
            pval = MCARTest(method="little")(X_train_missing)
            print(f"Little’s MCAR test p-value: {pval}")
            print("=========================END=========================>")
            # # T-test matrix
            # ttest_df = MCARTest(method="ttest")(X_train_missing)
            # print(ttest_df)
            # print("=========================>")
            # print("=========================>")
            # print("=========================>")
            # print("=========================>")
            # print("=========================>")

            # print("=========================>Train Missing Rate=========================>")
            # X_train_missing = mecha.fit(X_train)
            # X_train_missing = mecha.transform(X_train)
            # print(mecha.get_mask())
            # compute_missing_rate(X_train_missing)


            # print("=========================>Test Missing Rate=========================>")
            # X_test_missing = mecha.transform(X_test)
            # print(mecha.get_mask())
            # compute_missing_rate(X_test_missing)




