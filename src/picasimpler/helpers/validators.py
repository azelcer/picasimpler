from enum import Enum, auto

from PyQt6.QtCore import QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator

from picasimpler.helpers.utils import cust_auto

RE_PATT_SCIENCE_NO_WO_INF = r"((\+|-)?\d+)(\.((\d+)?))?((e|E)((\+|-)?)\d+)?"
RE_PATT_SCIENCE_NO_W_INF = rf"({RE_PATT_SCIENCE_NO_WO_INF})|\+?inf"
RE_PATT_SCIENCE_ONLY_POS_WO_INF = r"^(\+?\d+)(\.((\d+)?))?((e|E)((\+|-)?)\d+)?"
RE_PATT_SCIENCE_ONLY_POS_W_INF = rf"({RE_PATT_SCIENCE_ONLY_POS_WO_INF})|\+?inf"
RE_PATT_NO_SCIENCE_WO_INF = r"^((\+|-)?\d+)(\.((\d+)?))?"
RE_PATT_NO_SCIENCE_ONLY_POS_WO_INF = r"^(\+?\d+)(\.((\d+)?))?"
RE_PATT_INT_NO_SCIENCE_ONLY_POS_WO_INF = r"^(\+?\d+)"

class NumberValidators(Enum):
    """
    Enum class to define the different types of number validators

    Style 1:
        SCI_NOTAT_WO_INF: Scientific notation without infinity
        acceptable: 1, 1.0, 1.0e1, 1.0e+1, 1.0e-1, -1, -1.0, -1.0e1, -1.0e+1, -1.0e-1

    Style 2:
        SCI_NOTAT_W_INF: Scientific notation with infinity
        acceptable: 1, 1.0, 1.0e1, 1.0e+1, 1.0e-1, -1, -1.0, -1.0e1, -1.0e+1, -1.0e-1, inf, +inf

    Style 3:
        SCI_NOTAT_ONLY_POS_WO_INF: Scientific notation only positive numbers, WIHTOUT infinity
        acceptable: 1, 1.0, 1.0e1, 1.0e+1, 1.0e-1

    Style 4:
        SCI_NOTAT_ONLY_POS_W_INF: Scientific notation only positive numbers, WITH infinity
        acceptable: 1, 1.0, 1.0e1, 1.0e+1, 1.0e-1, inf, +inf
        
    Style 5:
        NO_SCI_NOTAT_WO_INF: No scientific notation, WITHOUT infinity
        acceptable: 1, 1.0, -1, -1.0
        
    Style 6:
        NO_SCI_NOTAT_ONLY_POS_WO_INF: No scientific notation only positive numbers, WITHOUT infinity
        acceptable: 1, 1.0   

    Style 7:
        INT_NO_SCI_NOTAT_ONLY_POS_WO_INF: Integers, no scientific notation only positive numbers, WITHOUT infinity
        acceptable: 1 
    """

    reg_exp_str: str
    reg_exp_val: QRegularExpressionValidator
    
    def __new__(cls, prog_order, reg_exp_str):
        obj = object.__new__(cls)
        obj._value_ = cust_auto(cls, prog_order)
        obj.reg_exp_str = reg_exp_str
        obj.reg_exp_val = QRegularExpressionValidator(QRegularExpression(obj.reg_exp_str))
        return obj

    SCI_NOTAT_WO_INF = (auto(), RE_PATT_SCIENCE_NO_WO_INF)
    SCI_NOTAT_W_INF = (auto(), RE_PATT_SCIENCE_NO_W_INF)
    SCI_NOTAT_ONLY_POS_WO_INF = (auto(), RE_PATT_SCIENCE_ONLY_POS_WO_INF)
    SCI_NOTAT_ONLY_POS_W_INF = (auto(), RE_PATT_SCIENCE_ONLY_POS_W_INF)
    NO_SCI_NOTAT_WO_INF = (auto(), RE_PATT_NO_SCIENCE_WO_INF)
    NO_SCI_NOTAT_ONLY_POS_WO_INF = (auto(), RE_PATT_NO_SCIENCE_ONLY_POS_WO_INF)
    INT_NO_SCI_NOTAT_ONLY_POS_WO_INF = (auto(), RE_PATT_INT_NO_SCIENCE_ONLY_POS_WO_INF)