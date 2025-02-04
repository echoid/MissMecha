from scipy import stats
import pandas as pd

class MissingMechanismValidator:
    """缺失机制统计验证"""
    
    @staticmethod
    def check_mcar(data: pd.DataFrame, missing_col: str) -> float:
        """
        Little's MCAR检验
        返回p-value (>0.05表示符合MCAR)
        """
        # 实现参考: https://www.statsmodels.org/stable/generated/statsmodels.imputation.mice.MICEData.html
        group1 = data[~data[missing_col].isna()].mean()
        group2 = data[data[missing_col].isna()].mean()
        _, p_value = stats.ttest_ind(group1, group2)
        return p_value