"""For miscellaneous utilities"""
from typing import List


def round_list(nums: List[float], precision=2):
    return [round(n, precision) for n in nums]
