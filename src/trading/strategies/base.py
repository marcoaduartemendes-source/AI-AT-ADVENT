from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    strategy_name: str
    product_id: str
    signal: SignalType
    confidence: float  # 0.0 – 1.0
    price: float
    reason: str
    metadata: dict = field(default_factory=dict)


class BaseStrategy:
    def __init__(
        self,
        name: str,
        products: list,
        granularity: str = "ONE_HOUR",
        lookback: int = 100,
    ):
        self.name = name
        self.products = products
        self.granularity = granularity
        self.lookback = lookback

    def analyze(self, product_id: str, candles: np.ndarray) -> Signal:
        raise NotImplementedError(f"{self.__class__.__name__}.analyze() not implemented")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
