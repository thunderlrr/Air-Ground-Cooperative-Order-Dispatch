"""
统一的距离计算工具模块
用于road_network_vehicles.py和环境文件
"""
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import networkx as nx
from typing import Tuple, Optional, Dict

class DistanceCalculator:
    """统一的距离计算器 - 支持真实经纬度坐标"""
    
    def __init__(self):
        self.node_coordinates: Dict[int, Tuple[float, float]] = {}
        self.road_graph = None
        self._load_node_coordinates()
    
    def _load_node_coordinates(self):
        """加载节点的真实经纬度坐标"""
        try:
            # 方法1：从nodes_processed.csv加载（最直接）
            df = pd.read_csv('./shanghai_road_dataset/road_network/nodes_processed.csv')
            for _, row in df.iterrows():
                node_id = int(row.iloc[0])
                longitude = float(row.iloc[1])
                latitude = float(row.iloc[2])
                self.node_coordinates[node_id] = (longitude, latitude)
            
        except Exception as e:
            try:
                # 方法2：从node_features_fixed.npy加载（备用）
                node_features = np.load('./shanghai_road_dataset/road_network/node_features_fixed.npy')
                self.node_coordinates = {
                    i: (float(node_features[i][0]), float(node_features[i][1])) 
                    for i in range(len(node_features))
                }
                print(f"✅ 从NPY加载节点坐标: {len(self.node_coordinates)} 个节点")
                
            except Exception as e2:
                print(f"❌ 节点坐标加载失败: CSV错误={e}, NPY错误={e2}")
                self.node_coordinates = {}
    
    def set_road_graph(self, road_graph):
        """设置路网图"""
        self.road_graph = road_graph
    
    def haversine_distance(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """
        使用Haversine公式计算两个经纬度点间的直线距离
        返回：距离(km)
        """
        # 转换为弧度
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine公式
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # 地球半径 6371 km
        return c * 6371
    
    def get_straight_line_distance(self, node1: int, node2: int) -> float:
        """
        获取两个节点间的直线距离(km) - 用于UAV
        """
        if node1 == node2:
            return 0.0
            
        # 优先使用真实坐标
        if node1 in self.node_coordinates and node2 in self.node_coordinates:
            lon1, lat1 = self.node_coordinates[node1]
            lon2, lat2 = self.node_coordinates[node2]
            return self.haversine_distance(lon1, lat1, lon2, lat2)
        
        # 回退：基于节点ID差值估算（极不推荐）
        node_dist = abs(node1 - node2)
        estimated_km = node_dist * 0.0001  # 保守估算
        print(f"⚠️ 节点{node1}->{node2}使用回退距离估算: {estimated_km:.3f}km")
        return estimated_km
    
    def get_road_network_distance(self, node1: int, node2: int) -> float:
        """
        获取路网中两点间的实际路径距离(km) - 用于Carrier
        """
        if node1 == node2:
            return 0.0
            
        if self.road_graph is None:
            print("⚠️ 路网图未设置，使用直线距离")
            return self.get_straight_line_distance(node1, node2)
        
        try:
            # 使用路网最短路径
            path_weight = nx.shortest_path_length(self.road_graph, node1, node2, weight='weight')
            
            # 🔧 修正：路网weight已经是距离的0.01倍，需要*102还原为km
            actual_distance_km = path_weight * 102
            return actual_distance_km
            
        except nx.NetworkXNoPath:
            print(f"⚠️ 路网中无路径: {node1} -> {node2}，使用直线距离")
            return self.get_straight_line_distance(node1, node2)
        except Exception as e:
            print(f"⚠️ 路网距离计算错误: {e}，使用直线距离")
            return self.get_straight_line_distance(node1, node2)
    
    def calculate_travel_time(self, distance_km: float, speed_kmh: float) -> float:
        """
        根据距离和速度计算旅行时间
        参数：
            distance_km: 距离(km)
            speed_kmh: 速度(km/h)
        返回：时间(h)
        """
        if speed_kmh <= 0:
            return float('inf')
        return distance_km / speed_kmh

# 全局距离计算器实例
distance_calculator = DistanceCalculator()
