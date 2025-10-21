"""
性能优化缓存模块
用于缓存路径距离、能量计算等高频操作的结果
"""

import numpy as np
from typing import Dict, Tuple, Any
import time

class PathDistanceCache:
    """路径距离缓存类，避免重复计算相同的路径"""
    
    def __init__(self, max_cache_size=10000):
        self.distance_cache = {}
        self.time_cache = {}
        self.max_cache_size = max_cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get_distance_and_time(self, graph, start_node, end_node):
        """获取缓存的距离和时间，如果不存在则计算并缓存"""
        key = (start_node, end_node)
        
        # 检查缓存
        if key in self.distance_cache:
            self.cache_hits += 1
            return self.distance_cache[key], self.time_cache[key]
        
        # 缓存未命中，计算路径
        self.cache_misses += 1
        
        # 使用现有的路径计算逻辑
        try:
            import networkx as nx
            path_length = nx.shortest_path_length(graph, start_node, end_node, weight='length')
            travel_time = path_length / 60  # 假设速度为60km/h
            
            # 缓存结果（如果缓存未满）
            if len(self.distance_cache) < self.max_cache_size:
                self.distance_cache[key] = path_length
                self.time_cache[key] = travel_time
                
        except:
            # 无路径可达，返回无穷大
            path_length = float('inf')
            travel_time = float('inf')
            
        return path_length, travel_time
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.distance_cache)
        }

class EnergyCalculationCache:
    """能量计算缓存类，用于缓存UAV能量计算结果"""
    
    def __init__(self, max_cache_size=5000):
        self.energy_cache = {}
        self.max_cache_size = max_cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_energy_consumption(self, distance_km, flight_time_h, weight_kg, base_power_kw=0.32, weight_factor=0.08):
        """获取缓存的能量消耗，如果不存在则计算并缓存"""
        # 创建缓存键（四舍五入到合理精度）
        key = (
            round(distance_km, 2),
            round(flight_time_h, 3),
            round(weight_kg, 2)
        )
        
        # 检查缓存
        if key in self.energy_cache:
            self.cache_hits += 1
            return self.energy_cache[key]
        
        # 缓存未命中，计算能量消耗
        self.cache_misses += 1
        energy_consumption = flight_time_h * (base_power_kw + weight_factor * weight_kg)
        
        # 缓存结果（如果缓存未满）
        if len(self.energy_cache) < self.max_cache_size:
            self.energy_cache[key] = energy_consumption
            
        return energy_consumption
    
    def simplified_energy_check(self, uav_battery, uav_location, order_pickup, order_delivery, order_weight, 
                               path_cache, graph, safety_margin=0.25):
        """简化的能量可行性检查，避免详细的三阶段计算"""
        # 获取总距离（使用缓存）
        pickup_dist, pickup_time = path_cache.get_distance_and_time(graph, uav_location, order_pickup)
        delivery_dist, delivery_time = path_cache.get_distance_and_time(graph, order_pickup, order_delivery)
        return_dist, return_time = path_cache.get_distance_and_time(graph, order_delivery, uav_location)
        
        # 如果任何路径不可达，返回False
        if any(d == float('inf') for d in [pickup_dist, delivery_dist, return_dist]):
            return False
        
        # 简化能量估算
        total_distance = pickup_dist + delivery_dist + return_dist
        
        # 分阶段估算（简化版）
        empty_energy = self.get_energy_consumption(pickup_dist + return_dist, pickup_time + return_time, 0)
        loaded_energy = self.get_energy_consumption(delivery_dist, delivery_time, order_weight)
        total_energy = empty_energy + loaded_energy
        
        # 加上安全余量
        required_energy = total_energy * (1 + safety_margin)
        
        return required_energy <= uav_battery

class PerformanceOptimizer:
    """性能优化管理器，整合所有缓存和优化策略"""
    
    def __init__(self):
        self.path_cache = PathDistanceCache()
        self.energy_cache = EnergyCalculationCache()
        self.start_time = time.time()
        
    def get_optimization_stats(self):
        """获取所有优化统计信息"""
        return {
            'path_cache': self.path_cache.get_cache_stats(),
            'energy_cache': self.energy_cache.get_cache_stats(),
            'runtime_seconds': time.time() - self.start_time
        }
    
    def print_optimization_report(self):
        """打印优化效果报告"""
        stats = self.get_optimization_stats()
        print("\n" + "="*60)
        print("🚀 性能优化效果报告")
        print("="*60)
        
        print(f"📍 路径缓存: 命中率 {stats['path_cache']['hit_rate']:.2%}, "
              f"缓存大小 {stats['path_cache']['cache_size']}")
        
        print(f"⚡ 能量缓存: 命中率 {stats['energy_cache']['hit_rate']:.2%}, "
              f"缓存大小 {stats['energy_cache']['cache_size']}")
        
        print(f"⏱️ 运行时间: {stats['runtime_seconds']:.2f} 秒")
        print("="*60)
