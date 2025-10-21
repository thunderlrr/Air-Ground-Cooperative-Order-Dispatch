import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import networkx as nx
from typing import Tuple, Optional, Dict

class DistanceCalculator:
    
    def __init__(self):
        self.node_coordinates: Dict[int, Tuple[float, float]] = {}
        self.road_graph = None
        self._load_node_coordinates()
    
    def _load_node_coordinates(self):
        try:
            df = pd.read_csv('./shanghai_road_dataset/road_network/nodes_processed.csv')
            for _, row in df.iterrows():
                node_id = int(row.iloc[0])
                longitude = float(row.iloc[1])
                latitude = float(row.iloc[2])
                self.node_coordinates[node_id] = (longitude, latitude)
            
        except Exception as e:
            try:
                node_features = np.load('./shanghai_road_dataset/road_network/node_features_fixed.npy')
                self.node_coordinates = {
                    i: (float(node_features[i][0]), float(node_features[i][1])) 
                    for i in range(len(node_features))
                }
                print(f"Loaded node coordinates from NPY: {len(self.node_coordinates)} nodes")
                
            except Exception as e2:
                print(f"Node coordinates loading failed: CSV error={e}, NPY error={e2}")
                self.node_coordinates = {}
    
    def set_road_graph(self, road_graph):
        self.road_graph = road_graph
    
    def haversine_distance(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        return c * 6371
    
    def get_straight_line_distance(self, node1: int, node2: int) -> float:
        if node1 == node2:
            return 0.0
            
        if node1 in self.node_coordinates and node2 in self.node_coordinates:
            lon1, lat1 = self.node_coordinates[node1]
            lon2, lat2 = self.node_coordinates[node2]
            return self.haversine_distance(lon1, lat1, lon2, lat2)
        
        node_dist = abs(node1 - node2)
        estimated_km = node_dist * 0.0001
        print(f"Fallback distance estimation for nodes {node1}->{node2}: {estimated_km:.3f}km")
        return estimated_km
    
    def get_road_network_distance(self, node1: int, node2: int) -> float:
        if node1 == node2:
            return 0.0
            
        if self.road_graph is None:
            print("Road graph not set, using straight line distance")
            return self.get_straight_line_distance(node1, node2)
        
        try:
            path_weight = nx.shortest_path_length(self.road_graph, node1, node2, weight='weight')
            actual_distance_km = path_weight * 102
            return actual_distance_km
            
        except nx.NetworkXNoPath:
            print(f"No path in road network: {node1} -> {node2}, using straight line distance")
            return self.get_straight_line_distance(node1, node2)
        except Exception as e:
            print(f"Road network distance calculation error: {e}, using straight line distance")
            return self.get_straight_line_distance(node1, node2)
    
    def calculate_travel_time(self, distance_km: float, speed_kmh: float) -> float:
        if speed_kmh <= 0:
            return float('inf')
        return distance_km / speed_kmh

distance_calculator = DistanceCalculator()