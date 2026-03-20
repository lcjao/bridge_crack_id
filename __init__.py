"""
桥梁裂纹智能检测系统
Bridge Crack Identification System

基于车-桥耦合动力学和神经网络的桥梁裂缝识别系统
"""

__version__ = "1.0.0"
__author__ = "Bridge Health Monitoring Research Team"

from .system import BridgeVehicleSystem
from .nn import NeuralNetwork
from .data_preparation import data_preparation

__all__ = [
    "BridgeVehicleSystem",
    "NeuralNetwork", 
    "data_preparation",
]
