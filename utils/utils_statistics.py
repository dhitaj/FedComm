"""
    This file will contain various statistical or related methods for weight observations etc.
"""


def difference(s1, s2):
    """
    :param s1: str
    :param s2: str
    :return: List of indexes where the strings differ
    """
    return [i for i in range(len(s1)) if s1[i] != s2[i]]
