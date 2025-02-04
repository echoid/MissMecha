
# generate.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

class MissingMechanism(ABC):
    """缺失机制抽象基类"""
    
    def __init__(self, 
                 target_cols: List[str],
                 missing_rate: float = 0.1,
                 data_type: str = "numeric"):
        """
        :param target_cols: 应用缺失的目标列
        :param missing_rate: 缺失率 (0-1)
        :param data_type: 数据类型 (numeric/categorical/datetime)
        """
        self.target_cols = target_cols
        self.missing_rate = missing_rate
        self.data_type = data_type
        
    @abstractmethod
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """应用缺失机制"""
        pass
    
    def _validate_inputs(self, data: pd.DataFrame):
        """输入校验"""
        missing_cols = [col for col in self.target_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
            
        if not 0 <= self.missing_rate <= 1:
            raise ValueError("Missing rate must be between 0 and 1")

class MCARMechanism(MissingMechanism):
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_inputs(data)
        data_missing = data.copy()
        
        for col in self.target_cols:
            mask = np.random.rand(len(data)) < self.missing_rate
            data_missing.loc[mask, col] = np.nan
            
        return data_missing

class MARMechanism(MissingMechanism):
    """随机缺失（依赖其他观测列）"""
    
    def __init__(self,
                 target_cols: List[str],
                 depend_cols: List[str],
                 model_type: str = "logistic",
                 **kwargs):
        """
        :param depend_cols: 决定缺失模式的依赖列
        :param model_type: 生成机制模型 (logistic/random/quantile)
        """
        super().__init__(target_cols, **kwargs)
        self.depend_cols = depend_cols
        self.model_type = model_type
        
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_inputs(data)
        data_missing = data.copy()
        
        X = data[self.depend_cols]
        if self.data_type == "categorical":
            X = self._encode_categorical(X)
            
        # 根据模型类型生成缺失概率
        if self.model_type == "logistic":
            prob = self._logistic_model(X)
        elif self.model_type == "quantile":
            prob = self._quantile_based(X)
        else:  # random
            prob = np.random.rand(len(X))
            
        # 应用缺失
        for col in self.target_cols:
            mask = prob > (1 - self.missing_rate)
            data_missing.loc[mask, col] = np.nan
            
        return data_missing
    
    def _logistic_model(self, X: pd.DataFrame) -> np.ndarray:
        """基于逻辑回归生成缺失概率"""
        model = LogisticRegression()
        y = np.random.randint(0, 2, len(X))  # 模拟依赖关系
        model.fit(X, y)
        return model.predict_proba(X)[:, 1]
    
    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """分类变量编码"""
        return X.apply(lambda x: LabelEncoder().fit_transform(x) if x.dtype == 'object' else x)

class MNARMechanism(MissingMechanism):
    """非随机缺失（依赖自身未观测值）"""
    
    def __init__(self,
                 target_cols: List[str],
                 threshold: Optional[float] = None,
                 direction: str = "above",
                 **kwargs):
        """
        :param threshold: 缺失阈值
        :param direction: 缺失方向 (above/below)
        """
        super().__init__(target_cols, **kwargs)
        self.threshold = threshold
        self.direction = direction
        
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_inputs(data)
        data_missing = data.copy()
        
        for col in self.target_cols:
            col_data = data[col]
            if self.threshold is None:
                self.threshold = np.percentile(col_data.dropna(), 75)  # 默认上四分位数
                
            if self.direction == "above":
                mask = col_data > self.threshold
            else:
                mask = col_data < self.threshold
                
            data_missing.loc[mask, col] = np.nan
            
        return data_missing

class MixedMechanism(MissingMechanism):
    """混合缺失模式"""
    
    def __init__(self, mechanisms: List[MissingMechanism]):
        self.mechanisms = mechanisms
        
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        data_missing = data.copy()
        for mechanism in self.mechanisms:
            data_missing = mechanism.apply(data_missing)
        return data_missing
    

class MissingGenerator:
    """缺失机制生成工厂类"""
    
    def __init__(self, 
                 mech_type: str, 
                 target_cols: List[str],
                 data_type: str = "numeric",
                 **kwargs):

        self.mech_type = mech_type.upper()
        self.data_type = data_type
        
        # 根据机制类型初始化对应类
        mech_map = {
            "MCAR": MCARMechanism,
            "MAR": MARMechanism,
            "MNAR": MNARMechanism
        }
        if self.mech_type not in mech_map:
            raise ValueError(f"Invalid mechanism type: {mech_type}")
            
        self.mechanism = mech_map[self.mech_type](
            target_cols=target_cols,
            data_type=data_type,
            **kwargs
        )
        
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.mechanism.apply(data)

