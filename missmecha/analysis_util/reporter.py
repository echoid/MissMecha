import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ImpactReporter:
    """缺失对下游任务影响分析"""
    
    def generate_report(self, original_df: pd.DataFrame, missing_df: pd.DataFrame, target_col: str):
        # 数据漂移分析
        drift_score = self._calc_drift(original_df, missing_df)
        
        # 可视化
        self._plot_distributions(original_df, missing_df, target_col)
        
        return {
            "data_drift": drift_score,
            "plots": ["distplot.png"]
        }
    
    def _calc_drift(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        # 计算Wasserstein距离
        from scipy.stats import wasserstein_distance
        return wasserstein_distance(df1.values.flatten(), df2.values.flatten())
    
    def _plot_distributions(self, df1: pd.DataFrame, df2: pd.DataFrame, col: str):
        plt.figure()
        sns.kdeplot(df1[col], label="Original")
        sns.kdeplot(df2[col].dropna(), label="With Missing")
        plt.savefig("distplot.png")