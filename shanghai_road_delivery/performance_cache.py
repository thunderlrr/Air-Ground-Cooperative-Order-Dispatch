import numpy as np
from typing import Dict, Tuple, Any
import time

class PathDistanceCache:
    
    def __init__(self, max_cache_size=10000):
        self.distance_cache = {}
        self.time_cache = {}
        self.max_cache_size = max_cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get_distance_and_time(self, graph, start_node, end_node):
        key = (start_node, end_node)
        
        if key in self.distance_cache:
            self.cache_hits += 1
            return self.distance_cache[key], self.time_cache[key]
        
        self.cache_misses += 1
        
        try:
            import networkx as nx
            path_length = nx.shortest_path_length(graph, start_node, end_node, weight='length')
            travel_time = path_length / 60
            
            if len(self.distance_cache) < self.max_cache_size:
                self.distance_cache[key] = path_length
                self.time_cache[key] = travel_time
                
        except:
            path_length = float('inf')
            travel_time = float('inf')
            
        return path_length, travel_time
    
    def get_cache_stats(self):
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.distance_cache)
        }

class EnergyCalculationCache:
    
    def __init__(self, max_cache_size=5000):
        self.energy_cache = {}
        self.max_cache_size = max_cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_energy_consumption(self, distance_km, flight_time_h, weight_kg, base_power_kw=0.32, weight_factor=0.08):
        key = (
            round(distance_km, 2),
            round(flight_time_h, 3),
            round(weight_kg, 2)
        )
        
        if key in self.energy_cache:
            self.cache_hits += 1
            return self.energy_cache[key]
        
        self.cache_misses += 1
        energy_consumption = flight_time_h * (base_power_kw + weight_factor * weight_kg)
        
        if len(self.energy_cache) < self.max_cache_size:
            self.energy_cache[key] = energy_consumption
            
        return energy_consumption
    
    def simplified_energy_check(self, uav_battery, uav_location, order_pickup, order_delivery, order_weight, 
                               path_cache, graph, safety_margin=0.25):
        pickup_dist, pickup_time = path_cache.get_distance_and_time(graph, uav_location, order_pickup)
        delivery_dist, delivery_time = path_cache.get_distance_and_time(graph, order_pickup, order_delivery)
        return_dist, return_time = path_cache.get_distance_and_time(graph, order_delivery, uav_location)
        
        if any(d == float('inf') for d in [pickup_dist, delivery_dist, return_dist]):
            return False
        
        total_distance = pickup_dist + delivery_dist + return_dist
        
        empty_energy = self.get_energy_consumption(pickup_dist + return_dist, pickup_time + return_time, 0)
        loaded_energy = self.get_energy_consumption(delivery_dist, delivery_time, order_weight)
        total_energy = empty_energy + loaded_energy
        
        required_energy = total_energy * (1 + safety_margin)
        
        return required_energy <= uav_battery

class PerformanceOptimizer:
    
    def __init__(self):
        self.path_cache = PathDistanceCache()
        self.energy_cache = EnergyCalculationCache()
        self.start_time = time.time()
        
    def get_optimization_stats(self):
        return {
            'path_cache': self.path_cache.get_cache_stats(),
            'energy_cache': self.energy_cache.get_cache_stats(),
            'runtime_seconds': time.time() - self.start_time
        }
    
    def print_optimization_report(self):
        stats = self.get_optimization_stats()
        print("\n" + "="*60)
        print("Performance Optimization Report")
        print("="*60)
        
        print(f"Path Cache: hit rate {stats['path_cache']['hit_rate']:.2%}, "
              f"cache size {stats['path_cache']['cache_size']}")
        
        print(f"Energy Cache: hit rate {stats['energy_cache']['hit_rate']:.2%}, "
              f"cache size {stats['energy_cache']['cache_size']}")
        
        print(f"Runtime: {stats['runtime_seconds']:.2f} seconds")
        print("="*60)