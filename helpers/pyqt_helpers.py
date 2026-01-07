#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 16:47:27 2026

@author: azelcer
"""
from PyQt5.QtWidgets import (
    QLabel as _QLabel,
    QHBoxLayout as _QHBoxLayout,
    QSpinBox as _QSpinBox,
    QDoubleSpinBox as _QDoubleSpinBox,
    QBoxLayout,
)


def create_labeled_float(name: str, external_layout: QBoxLayout,
                         value: float, decimals: int, step: float,
                         minimum: float = 0., maximum: float = None
                         ) -> _QDoubleSpinBox:
    """Creates a labeled float spinbox."""
    hlayout = _QHBoxLayout()
    sb = _QDoubleSpinBox()
    sb.setDecimals(decimals)
    sb.setMinimum(minimum)
    if maximum is not None:
        sb.setMaximum(maximum)
    sb.setSingleStep(step)
    sb.setValue(value)
    hlayout.addWidget(_QLabel(name))
    hlayout.addWidget(sb)
    external_layout.addLayout(hlayout)
    return sb


def create_labeled_int(name: str, external_layout: QBoxLayout,
                       value: int, step: int = 1,
                       minimum: int | None = None, maximum: int | None = None,
                       ) -> _QSpinBox:
    """Creates a labeled float spinbox."""
    hlayout = _QHBoxLayout()
    sb = _QSpinBox()
    if minimum is not None:
        sb.setMinimum(minimum)
    if maximum is None:
        maximum = (1 << 31) - 1  # signed 32 bit
    sb.setMaximum(maximum)

    sb.setValue(value)
    hlayout.addWidget(_QLabel(name))
    hlayout.addWidget(sb)
    external_layout.addLayout(hlayout)
    return sb
