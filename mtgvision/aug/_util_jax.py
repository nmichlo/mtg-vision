from enum import Enum


class ResizeMethod(str, Enum):
    NEAREST = "nearest"
    LINEAR = "linear"
    LANCZOS3 = "lanczos3"
    LANCZOS5 = "lanczos5"
    CUBIC = "cubic"
