#!/usr/bin/env python3
"""
带能量约束的纯路网环境 - 集成UAV电池限制和Carrier工作时间限制
基于验证成功的真实路网结构，使用shared_components中的基础类
"""

import numpy as np
import networkx as nx
import torch
import pickle
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

# 导入路网车辆适配器
from road_network_vehicles import RoadNetworkUAV, RoadNetworkCarrier, RoadNetworkOrder

# 🚀 导入性能优化缓存
from performance_cache import PerformanceOptimizer
# 🆕 导入统一距离计算器
from distance_utils import distance_calculator
from enum import Enum

class VehicleState(Enum):
    """车辆工作状态枚举"""
    IDLE = "idle"                    # 空闲状态，可以接受新订单
    ASSIGNED = "assigned"            # 已分配订单，前往取货
    DELIVERING = "delivering"        # 配送中，不能接受新订单

class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"              # 等待分配
    ASSIGNED = "assigned"            # 已分配给车辆
    PICKED_UP = "picked_up"          # 已取货
    COMPLETED = "completed"          # 配送完成
    EXPIRED = "expired"              # 已过期

@dataclass
class RealRoadOrder:
    """基于真实路网的订单"""
    order_id: int
    start_node: int  # 真实路网节点ID
    end_node: int    # 真实路网节点ID
    start_time: int
    deadline: int
    weight: float
    priority: int
    created_time: int
    original_road_order_id: int  # 原始路网订单ID

class PureRealRoadNetworkEnvironmentWithConstraints:
    """带约束的真实路网配送环境 - 集成能量和工作时间限制"""
    
    def _get_global_order_distribution(self):
        """获取全局订单分布信息"""
        if not self.active_orders:
            return np.zeros(8, dtype=np.float32)
        
        # 计算全局订单统计
        total_orders = len(self.active_orders)
        urgent_orders = sum(1 for order in self.active_orders.values() if self._is_order_urgent(order))
        
        # 订单地理分布（简化为区域分布）
        order_nodes = [order.start_node for order in self.active_orders.values()]
        avg_order_node = np.mean(order_nodes) if order_nodes else 0
        
        # 车辆-订单匹配度
        matched_orders = sum(1 for vehicle in self.vehicles if vehicle.vehicle_id in self.vehicle_assigned_orders)
        
        global_info = np.array([
            total_orders / 50.0,              # 总订单数 (归一化)
            urgent_orders / max(total_orders, 1),  # 紧急订单比例
            matched_orders / self.num_vehicles,    # 车辆匹配率
            self.time_step / self.max_time_steps,  # 时间进度
            len(self.completed_orders) / max(self.total_orders_generated, 1),  # 完成率
            self.total_orders_matched / max(self.total_orders_generated, 1),   # 匹配率
            avg_order_node / 600000.0,        # 平均订单位置(归一化)
            min(len(self.active_orders) / 20.0, 1.0)  # 订单密度
        ], dtype=np.float32)
        
        return global_info

    def __init__(self, num_ground_vehicles=6, num_uavs=3, max_time_steps=120, max_concurrent_orders=15):
        self.num_ground_vehicles = num_ground_vehicles
        self.num_uavs = num_uavs
        self.num_vehicles = num_ground_vehicles + num_uavs
        self.max_concurrent_orders = max_concurrent_orders  # <-- Add this line
        
        # 真实路网数据
        self.road_graph = None
        self.node_features = None
        self.node_embeddings = {}
        self.largest_component_nodes = None  # 最大连通分量节点
        
        # 车辆和订单
        self.vehicles: List = []  # 混合类型：RoadNetworkUAV 和 RoadNetworkCarrier

        # 🆕 状态跟踪
        self.vehicle_states = {}  # vehicle_id -> VehicleState
        self.vehicle_assigned_orders = {}  # vehicle_id -> order
        self.order_statuses = {}  # order_id -> OrderStatus
        self.active_orders: Dict[int, RealRoadOrder] = {}
        self.completed_orders: List[RealRoadOrder] = []
        
        # 环境状态
        self.time_step = 0
        self.max_time_steps = max_time_steps  # 匹配订单数据的时间步数
        self.orders_data = None
        
        # 统计信息
        self.total_orders_generated = 0
        self.total_orders_completed = 0
        self.total_orders_matched = 0
        self.path_planning_failures = 0

        # 🚀 性能优化器
        self.performance_optimizer = PerformanceOptimizer()
        self.uav_energy_failures = 0      # 新增：UAV能量不足次数
        #self.carrier_time_failures = 0    # 新增：Carrier工作时间不足次数
        self.charging_events = 0           # 新增：充电事件计数
        self.episode_count = 0             # 新增：episode计数器

        # 📊 Episode统计计数器
        self.episode_pickups = 0           # 本episode取货次数
        self.episode_deliveries = 0        # 本episode送货次数
        # max_concurrent_orders already set in __init__

        # 🆕 重置状态跟踪
        self.vehicle_states = {}
        self.vehicle_assigned_orders = {}
        self.order_statuses = {}
        
        # 状态维度
        self.state_dim = 52  # 实际维度：8+4+8+8+12+8+4=52
        
        self._load_real_road_network()
        self._initialize_vehicles_with_constraints()
        
    def _load_real_road_network(self):
        """加载真实上海路网数据"""
        print("⚙️ 加载上海真实路网图结构...")
        
        try:
            # 加载路网图
            with open('./shanghai_road_dataset/road_network/road_graph_fixed.pkl', 'rb') as f:
                self.road_graph = pickle.load(f)
            print(f"✅ 路网图加载成功: {len(self.road_graph.nodes)} 节点, {len(self.road_graph.edges)} 边")
            # 🆕 设置距离计算器
            distance_calculator.set_road_graph(self.road_graph)
            
            # 加载节点特征
            self.node_features = np.load('./shanghai_road_dataset/road_network/node_features_fixed.npy')
            print(f"✅ 节点特征加载成功: {self.node_features.shape}")
            
            # 预计算节点嵌入 (使用节点特征的前8维)
            self.node_embeddings = {}
            for node_id in self.road_graph.nodes:
                if node_id < len(self.node_features):
                    raw_features = self.node_features[node_id][:8]
                    normalized_features = raw_features.copy()
                    normalized_features[0] = (raw_features[0] - 121.0) / 1.0  # 经度归一化
                    normalized_features[1] = (raw_features[1] - 31.0) / 1.0   # 纬度归一化
                    normalized_features[2:] = raw_features[2:] / (np.abs(raw_features[2:]) + 1e-6)  # 其他特征归一化
                    
                    self.node_embeddings[node_id] = normalized_features.astype(np.float32)
                else:
                    self.node_embeddings[node_id] = np.zeros(8, dtype=np.float32)
            
            print(f"✅ 节点嵌入预计算完成: {len(self.node_embeddings)} 个节点")
            
            # 找到最大连通分量
            connected_components = list(nx.connected_components(self.road_graph))
            self.largest_component_nodes = list(max(connected_components, key=len))
            print(f"✅ 最大连通分量: {len(self.largest_component_nodes)} 节点 ({len(self.largest_component_nodes)/len(self.road_graph.nodes)*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ 真实路网加载失败: {e}")
            raise RuntimeError("必须使用真实路网数据！")
    
    def _initialize_vehicles_with_constraints(self):
        """初始化带约束的车辆"""
        self.vehicles = []
        
        # 选择起始节点
        if self.largest_component_nodes is None or len(self.largest_component_nodes) < self.num_vehicles:
            if self.largest_component_nodes is None:
                print(f"⚠️ 最大连通分量未初始化")
                available_nodes = list(self.road_graph.nodes())[:self.num_vehicles] if self.road_graph else [0] * self.num_vehicles
            else:
                print(f"⚠️ 连通分量节点数({len(self.largest_component_nodes)}) < 车辆数({self.num_vehicles})")
                available_nodes = self.largest_component_nodes
        else:
            available_nodes = np.random.choice(self.largest_component_nodes, self.num_vehicles, replace=False)
        
        # 选择充电桩位置 (从连通分量中随机选择几个作为充电桩)
        num_charging_stations = 15  # 充电桩数量
        if self.largest_component_nodes is not None and len(self.largest_component_nodes) >= num_charging_stations:
            charging_stations = np.random.choice(self.largest_component_nodes, num_charging_stations, replace=False)
        else:
            charging_stations = self.largest_component_nodes[:num_charging_stations] if self.largest_component_nodes is not None else [0] * num_charging_stations
        
        self.charging_stations = charging_stations  # 保存充电桩列表
        print(f"🔋 充电桩位置: {charging_stations}")
        
        # 创建地面载体 (Carrier) - 工作时间限制
        for i in range(self.num_ground_vehicles):
            start_node = available_nodes[i % len(available_nodes)]
            # 工作时间限制：8小时，容量无限，速度30km/h
            carrier = RoadNetworkCarrier(
                carrier_id=i, 
                start_node=start_node, 
                road_graph=self.road_graph,
                capacity=float('inf'),  # 地面载体无载重限制
                speed=45,               # 45 km/h (电动车/摩托车)
                # range_limit=50,         # 50km范围限制
                # max_work_hours=8        # 8小时工作限制
            )
            self.vehicles.append(carrier)
            self.vehicle_states[carrier.vehicle_id] = VehicleState.IDLE
        
        # 创建无人机 (UAV) - 电池能量限制
        for i in range(self.num_uavs):
            vehicle_id = self.num_ground_vehicles + i
            start_node = available_nodes[vehicle_id % len(available_nodes)]
            charging_station = charging_stations[i % len(charging_stations)]  # 分配充电桩
            
            # 电池容量：100kWh，载重5kg，速度60km/h
            uav = RoadNetworkUAV(
                uav_id=vehicle_id,
                start_node=start_node,
                road_graph=self.road_graph,
                capacity=5,                    # 5kg载重限制
                battery_capacity= 1,          # 50kWh电池
                charging_station_node=charging_station,  # 指定充电桩
                charging_stations_list=charging_stations,  # 充电桩列表
                speed=60                        # 60 km/h (高性能配送无人机)
            )
            self.vehicles.append(uav)
            self.vehicle_states[uav.vehicle_id] = VehicleState.IDLE
        
        print(f"✅ 初始化 {len(self.vehicles)} 辆带约束车辆:")
        print(f"   地面载体: {self.num_ground_vehicles} 辆 (工作时间限制: 8h)")
        print(f"   无人机: {self.num_uavs} 辆 (电池限制: 50kWh)")
        print(f"   车辆位置: {[v.current_node for v in self.vehicles]}")
    
    def load_road_orders(self, day: int) -> bool:
        """加载真实路网订单数据"""
        try:
            order_file = f'./shanghai_road_dataset/processed_orders_road/Orders_Dataset_shanghai_road_day_{day}'
            with open(order_file, 'rb') as f:
                self.orders_data = pickle.load(f)
            
            print(f"✅ 真实路网订单加载成功: 第{day}天, {len(self.orders_data)} 个时间步")
            return True
            
        except Exception as e:
            print(f"❌ 真实路网订单加载失败: {e}")
            return False
    
    def generate_orders(self) -> List[RealRoadOrder]:
        """生成当前时间步的真实路网订单"""
        new_orders = []
        
        if not hasattr(self, 'orders_data') or not self.orders_data or self.time_step not in self.orders_data:
            return new_orders
        
        if self.road_graph is None:
            print("⚠️ 路网图未加载，无法生成订单")
            return new_orders
        
        time_step_orders = self.orders_data[self.time_step]
        
        for order_key, order_data in time_step_orders.items():
            try:
                if isinstance(order_data, list) and len(order_data) > 0:
                    if isinstance(order_data[0], list):
                        actual_data = order_data[0]
                    else:
                        actual_data = order_data
                else:
                    continue
                
                if len(actual_data) >= 5:
                    road_order_id, start_time, end_time, start_node, end_node = actual_data[:5]
                    
                    if (self.road_graph is not None and 
                        start_node in self.road_graph.nodes and 
                        end_node in self.road_graph.nodes):
                        try:
                            path_length = nx.shortest_path_length(self.road_graph, start_node, end_node)
                        except nx.NetworkXNoPath:
                            continue
                        except Exception:
                            path_length = abs(start_node - end_node) // 1000
                        
                        weight = min(max(path_length / 100, 1.0), 10.0)  
                        priority = 1 if path_length < 50 else (2 if path_length < 200 else 3)  
                        
                        order = RealRoadOrder(
                            order_id=len(self.active_orders) + len(new_orders) + 10000,
                            start_node=int(start_node),
                            end_node=int(end_node),
                            start_time=int(start_time),
                            deadline=int(end_time),  # 直接使用end_time作为DDL，不添加额外缓冲
                            weight=float(weight),
                            priority=int(priority),
                            created_time=int(self.time_step),
                            original_road_order_id=int(road_order_id)
                        )
                        
                        new_orders.append(order)
                        self.active_orders[order.order_id] = order
            
            except Exception as e:
                continue
        
        self.total_orders_generated += len(new_orders)
        
        if new_orders:
            print(f"📦 时间步{self.time_step}: 生成 {len(new_orders)} 个真实路网订单")
        
        return new_orders
    

    # 🆕 状态管理方法
    def assign_order_to_vehicle(self, vehicle_id, order):
        """分配订单给车辆"""
        if self.vehicle_states.get(vehicle_id) != VehicleState.IDLE:
            return False
        
        self.vehicle_states[vehicle_id] = VehicleState.ASSIGNED
        self.vehicle_assigned_orders[vehicle_id] = order
        self.order_statuses[order.order_id] = OrderStatus.ASSIGNED
        
        # 标记订单已分配
        order.assigned_vehicle = vehicle_id
        
        print(f"🎯 车辆 {vehicle_id} 接受订单 {order.order_id}")
        return True
    
    def vehicle_start_pickup(self, vehicle_id):
        """车辆开始取货"""
        if (self.vehicle_states.get(vehicle_id) == VehicleState.ASSIGNED and 
            vehicle_id in self.vehicle_assigned_orders):
            order = self.vehicle_assigned_orders[vehicle_id]
            self.order_statuses[order.order_id] = OrderStatus.PICKED_UP
            
            # 🆕 根据车辆类型显示正确用语
            vehicle = next((v for v in self.vehicles if v.vehicle_id == vehicle_id), None)
            if vehicle and hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'uav':
                print(f"📦 UAV {vehicle_id} 开始取货订单 {order.order_id}")
            else:
                print(f"📦 Carrier {vehicle_id} 开始取货订单 {order.order_id}")

            # 📊 统计取货次数
            self.episode_pickups += 1
            return True
        return False
    
    def vehicle_start_delivery(self, vehicle_id):
        """车辆开始配送"""
        if (self.vehicle_states.get(vehicle_id) == VehicleState.ASSIGNED and 
            vehicle_id in self.vehicle_assigned_orders):
            self.vehicle_states[vehicle_id] = VehicleState.DELIVERING
            order = self.vehicle_assigned_orders[vehicle_id]
            
            # 🆕 根据车辆类型显示正确用语
            vehicle = next((v for v in self.vehicles if v.vehicle_id == vehicle_id), None)
            if vehicle and hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'uav':
                print(f"🚁 UAV {vehicle_id} 开始配送订单 {order.order_id}")
            else:
                print(f"�� Carrier {vehicle_id} 开始配送订单 {order.order_id}")
            return True
        return False
    
    def vehicle_complete_delivery(self, vehicle_id):
        """车辆完成配送"""
        if (self.vehicle_states.get(vehicle_id) == VehicleState.DELIVERING and 
            vehicle_id in self.vehicle_assigned_orders):
            order = self.vehicle_assigned_orders[vehicle_id]
            
            # 更新状态
            self.order_statuses[order.order_id] = OrderStatus.COMPLETED
            self.vehicle_states[vehicle_id] = VehicleState.IDLE
            
            # 移动订单到完成列表
            if order.order_id in self.active_orders:
                completed_order = self.active_orders.pop(order.order_id)
                self.completed_orders.append(completed_order)
                self.total_orders_completed += 1

                # 📊 统计配送次数
                self.episode_deliveries += 1
            
            # 清理分配
            del self.vehicle_assigned_orders[vehicle_id]
            
            # 🆕 获取车辆类型用于正确显示
            vehicle = next((v for v in self.vehicles if v.vehicle_id == vehicle_id), None)
            if vehicle and hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'uav':
                print(f"✅ UAV {vehicle_id} 完成配送订单 {order.order_id}，重新进入空闲状态")
            else:
                print(f"✅ Carrier {vehicle_id} 完成配送订单 {order.order_id}，重新进入空闲状态")
            
            return True
        return False
    
    # def is_vehicle_available(self, vehicle_id):
    #     """检查车辆是否可接受新订单"""
    #     return self.vehicle_states.get(vehicle_id) == VehicleState.IDLE
    
    # def is_vehicle_busy(self, vehicle_id):
    #     """检查车辆是否忙碌"""
    #     state = self.vehicle_states.get(vehicle_id)
    #     return state in [VehicleState.ASSIGNED, VehicleState.DELIVERING]
    
    # def get_vehicle_status_info(self, vehicle_id):
    #     """获取车辆状态信息"""
    #     vehicle = next((v for v in self.vehicles if v.vehicle_id == vehicle_id), None)
    #     if not vehicle:
    #         return None
            
    #     assigned_order = self.vehicle_assigned_orders.get(vehicle_id)

    #     return {
    #         "vehicle_id": vehicle_id,
    #         "vehicle_type": vehicle.vehicle_type,
    #         "state": self.vehicle_states.get(vehicle_id, VehicleState.IDLE).value,
    #         "current_node": vehicle.current_node,
    #         "assigned_order_id": assigned_order.order_id if assigned_order else None
    #     }
    

    def _get_distance(self, node1, node2):
        if node1 is None or node2 is None:
            return float('inf')
        if self.road_graph is not None:
            return distance_calculator.get_road_network_distance(node1, node2)
        else:
            return distance_calculator.get_straight_line_distance(node1, node2)
    
    # def can_complete_order_within_ddl(self, vehicle, order):
    #     """检查车辆是否能在DDL内完成订单"""
    #     try:
    #         # 计算到达取货点的时间
    #         if vehicle.vehicle_type == 'uav':
    #             pickup_time = vehicle.calculate_flight_time_to_node(order.start_node)
    #             delivery_time = vehicle.calculate_flight_time_to_node(order.end_node)
    #         else:  # carrier
    #             pickup_time = vehicle.calculate_road_travel_time(vehicle.current_node, order.start_node)
    #             delivery_time = vehicle.calculate_road_travel_time(order.start_node, order.end_node)
            
    #         # 总完成时间 = 当前时间 + 到达取货点时间 + 配送时间
    #         total_completion_time = self.time_step + pickup_time + delivery_time
            
    #         # 检查是否在DDL内
    #         ddl_margin = order.deadline - total_completion_time
            
    #         if ddl_margin >= 0:
    #             print(f"   ✅ 订单{order.order_id} DDL检查通过，剩余时间: {ddl_margin:.2f}")
    #             return True
    #         else:
    #             print(f"   ❌ 订单{order.order_id} DDL检查失败，超时: {-ddl_margin:.2f}")
    #             return False
                
    #     except Exception as e:
    #         print(f"   ⚠️ 订单{order.order_id} DDL检查出错: {e}")
    #         return False
    
    def _is_order_urgent(self, order):
        """判断订单是否紧急（接近DDL）"""
        time_remaining = order.deadline - self.time_step
        return time_remaining <= 10  # 剩余时间<=10步认为紧急


    def _perform_platform_order_assignment(self, action: np.ndarray) -> int:
        """🆕 平台DDPG订单分配 - 基于车辆-订单匹配权重"""
        if not self.active_orders:
            return 0
        
        # 获取可用车辆和活跃订单
        available_vehicles = [v for v in self.vehicles if self.vehicle_states.get(v.vehicle_id) == VehicleState.IDLE]
        active_orders_list = list(self.active_orders.values())
        
        if not available_vehicles or not active_orders_list:
            return 0
        
        # 限制订单数量到最大并发数
        active_orders_list = active_orders_list[:self.max_concurrent_orders]
        
        # 动作维度：available_vehicles × active_orders
        # 动作展开为匹配权重矩阵
        action_matrix_size = len(available_vehicles) * len(active_orders_list)
        
        # 如果动作维度不够，填充0.5（中性值）
        if len(action) < action_matrix_size:
            padded_action = np.pad(action, (0, action_matrix_size - len(action)), constant_values=0.5)
        else:
            padded_action = action[:action_matrix_size]
        
        # 重塑为权重矩阵 [vehicles, orders]
        weight_matrix = padded_action.reshape(len(available_vehicles), len(active_orders_list))
        
        assignment_count = 0
        used_orders = set()
        
        # 贪心匹配：选择权重最高的车辆-订单对
        for _ in range(min(len(available_vehicles), len(active_orders_list))):
            best_score = -1
            best_vehicle_idx = -1
            best_order_idx = -1
            
            for v_idx, vehicle in enumerate(available_vehicles):
                if self.vehicle_states.get(vehicle.vehicle_id) != VehicleState.IDLE:
                    continue
                    
                for o_idx, order in enumerate(active_orders_list):
                    if o_idx in used_orders:
                        continue
                        
                    # 约束检查
                    if not self._can_vehicle_handle_order(vehicle, order):
                        continue
                        
                    score = weight_matrix[v_idx, o_idx]
                    if score > best_score and score > 0.1:  # 分数门槛
                        best_score = score
                        best_vehicle_idx = v_idx
                        best_order_idx = o_idx
            
            # 执行最佳匹配
            if best_vehicle_idx >= 0 and best_order_idx >= 0:
                vehicle = available_vehicles[best_vehicle_idx]
                order = active_orders_list[best_order_idx]
                
                if self.assign_order_to_vehicle(vehicle.vehicle_id, order):
                    assignment_count += 1
                    used_orders.add(best_order_idx)
                    print(f"   🎯 平台匹配: 车辆{vehicle.vehicle_id} ← 订单{order.order_id}: 权重={best_score:.3f}")
        
        return assignment_count

    # def _perform_global_order_assignment(self, actions: List[np.ndarray]):
    #     """重构版全局订单分配 - 基于动作对订单评分"""

    #     if not self.active_orders:
    #         return 0
        
    #     assignment_count = 0
        
    #     # 获取未分配的订单列表 (按创建时间排序，确保一致性)
    #     available_orders = []

    #     for order in self.active_orders.values():
    #         if order.order_id not in [assigned_order.order_id for assigned_order in self.vehicle_assigned_orders.values()]:
    #             available_orders.append(order)
        
    #     # 按创建时间排序，确保订单顺序一致性
    #     available_orders.sort(key=lambda x: x.created_time)
        
    #     if not available_orders:
    #         return 0
        
    #     # 为每个车辆找最佳订单（Vehicle-Centric with Order Scoring）
    #     for vehicle in self.vehicles:
    #         if self.vehicle_states.get(vehicle.vehicle_id) != VehicleState.IDLE:
    #             continue
            
    #         if vehicle.vehicle_id >= len(actions):
    #             continue
                
    #         action = actions[vehicle.vehicle_id]
            
    #         # 🆕 新动作空间：纯订单评分（无移动动作）
            
    #         order_scores = action if len(action) > 0 else []
            
    #         best_order = None
    #         best_score = -float('inf')
            
    #         # 为每个可用订单计算最终分数
    #         for idx, order in enumerate(available_orders[:self.max_concurrent_orders]):
    #             # 1. 约束检查（硬约束）
    #             if not self._can_vehicle_handle_order(vehicle, order):
    #                 continue
                
    #             # 2. 从动作中获取该订单的评分
    #             if idx < len(order_scores):
    #                 action_score = (order_scores[idx] + 1) / 2  # 归一化到[0,1]
    #             else:
    #                 action_score = 0.5  # 超出动作维度的订单使用中性评分
                
    #             # 3. 使用动作评分作为最终分数（纯强化学习）
    #             final_score = action_score
                
    #             if final_score > best_score:
    #                 best_score = final_score
    #                 best_order = order
            
    #             # 分配最佳匹配
    #         if best_order is not None and best_score > 0.1:  # 分数门槛0.1
    #             if self.assign_order_to_vehicle(vehicle.vehicle_id, best_order):
    #                 assignment_count += 1
    #                 available_orders.remove(best_order)  # 从可用订单中移除
    #                 print(f"   🎯 车辆{vehicle.vehicle_id} ← 订单{best_order.order_id}: 评分={best_score:.3f}")
        
    #     return assignment_count
    
    def _can_vehicle_handle_order(self, vehicle, order):
        """🆕 增强版：检查车辆是否能处理订单（包含UAV能量预检查）"""
        # DDL检查
        if not self._optimized_can_complete_order_within_ddl(vehicle, order):
            return False
        
        # 车辆特定约束检查
        if vehicle.vehicle_type == "uav":
            # 🆕 使用增强版UAV能量检查
            if hasattr(vehicle, "can_complete_three_phase_mission"):
                # 使用现有的三阶段能量检查
                can_handle, reason = vehicle.can_complete_three_phase_mission(order)
                if not can_handle:
                    print(f"   ❌ UAV{vehicle.vehicle_id} 能量不足，无法完成订单{order.order_id}: {reason}")
                return can_handle
            elif hasattr(vehicle, "can_deliver_order_safely"):
                # 回退到原始检查
                road_order = RoadNetworkOrder(
                    order.order_id, order.start_node, order.end_node,
                    order.start_time, order.deadline, order.weight, order.priority
                )
                return vehicle.can_deliver_order_safely(road_order)
        else:
            # # Carrier工作时间检查
            # road_order = RoadNetworkOrder(
            #     order.order_id, order.start_node, order.end_node,
            #     order.start_time, order.deadline, order.weight, order.priority
            # )
            # if hasattr(vehicle, "can_deliver_order_within_work_hours"):
            #     return vehicle.can_deliver_order_within_work_hours(road_order)
            return True
        
        return True


    def _update_vehicle_behaviors(self):
        """更新车辆行为：处理真实移动和任务进度"""
        for vehicle in self.vehicles:
            vehicle_id = vehicle.vehicle_id
            current_state = self.vehicle_states.get(vehicle_id, VehicleState.IDLE)
            
            if current_state == VehicleState.ASSIGNED:
                # 车辆已分配订单，需要前往取货点
                assigned_order = self.vehicle_assigned_orders.get(vehicle_id)
                if assigned_order:
                    self._handle_pickup_movement(vehicle, assigned_order)
                    
            elif current_state == VehicleState.DELIVERING:
                # 车辆正在配送，需要前往配送点
                assigned_order = self.vehicle_assigned_orders.get(vehicle_id)
                if assigned_order:
                    self._handle_delivery_movement(vehicle, assigned_order)

    def _optimized_can_complete_order_within_ddl(self, vehicle, order):
        """优化的DDL检查 - 使用缓存避免重复计算"""
        try:
            # 使用缓存计算路径时间
            path_cache = self.performance_optimizer.path_cache
            
            if vehicle.vehicle_type == 'uav':
                # UAV直线飞行时间计算（简化）
                pickup_dist, pickup_time = path_cache.get_distance_and_time(
                    self.road_graph, vehicle.current_node, order.start_node)
                delivery_dist, delivery_time = path_cache.get_distance_and_time(
                    self.road_graph, order.start_node, order.end_node)
                
                # 对于UAV，使用简化的时间计算
                if pickup_dist != float('inf') and delivery_dist != float('inf'):
                    # 假设UAV平均速度为80km/h
                    pickup_time = pickup_dist / 80.0
                    delivery_time = delivery_dist / 80.0
                else:
                    return False
            else:  # carrier
                pickup_dist, pickup_time = path_cache.get_distance_and_time(
                    self.road_graph, vehicle.current_node, order.start_node)
                delivery_dist, delivery_time = path_cache.get_distance_and_time(
                    self.road_graph, order.start_node, order.end_node)
                
                if pickup_dist == float('inf') or delivery_dist == float('inf'):
                    return False
            
            # 总完成时间 = 当前时间 + 到达取货点时间 + 配送时间
            total_completion_time = self.time_step + pickup_time + delivery_time
            
            # 检查是否在DDL内
            ddl_margin = order.deadline - total_completion_time
            
            return ddl_margin >= 0
                
        except Exception as e:
            return False

    def _optimized_uav_energy_check(self, uav, order):
        """优化的UAV能量检查 - 使用缓存和简化计算"""
        try:
            energy_cache = self.performance_optimizer.energy_cache
            path_cache = self.performance_optimizer.path_cache
            
            # 使用简化的能量检查
            return energy_cache.simplified_energy_check(
                uav_battery=uav.current_battery,
                uav_location=uav.current_node,
                order_pickup=order.start_node,
                order_delivery=order.end_node,
                order_weight=order.weight,
                path_cache=path_cache,
                graph=self.road_graph,
                safety_margin=0.25
            )
        except Exception as e:
            return False

    def _check_ddl_timeouts(self):
        """检查DDL超时并自动处理"""
        expired_vehicles = []
        
        for vehicle_id, order in list(self.vehicle_assigned_orders.items()):
            if order.deadline <= self.time_step:
                # 订单已过期
                vehicle_state = self.vehicle_states.get(vehicle_id)
                
                if vehicle_state == VehicleState.DELIVERING:
                    # 如果正在配送，视为成功完成
                    print(f"⏰ 订单{order.order_id}在DDL时刻完成配送")
                    self.vehicle_complete_delivery(vehicle_id)
                else:
                    # 如果还在前往取货，视为失败
                    print(f"❌ 订单{order.order_id}DDL超时，配送失败")
                    self._handle_order_failure(vehicle_id, order)
                    
                expired_vehicles.append(vehicle_id)
        
        return len(expired_vehicles)
    
    def _handle_order_failure(self, vehicle_id, order):
        """处理订单失败"""
        # 更新订单状态
        self.order_statuses[order.order_id] = OrderStatus.EXPIRED
        
        # 重置车辆状态
        self.vehicle_states[vehicle_id] = VehicleState.IDLE
        
        # 清理分配
        if vehicle_id in self.vehicle_assigned_orders:
            del self.vehicle_assigned_orders[vehicle_id]
        
        # 从活跃订单中移除
        if order.order_id in self.active_orders:
            expired_order = self.active_orders.pop(order.order_id)
            # 可以添加到失败订单列表以便分析
            if not hasattr(self, "failed_orders"):
                self.failed_orders = []
            self.failed_orders.append(expired_order)
        """更新车辆行为：处理真实移动和任务进度"""
        for vehicle in self.vehicles:
            vehicle_id = vehicle.vehicle_id
            current_state = self.vehicle_states.get(vehicle_id, VehicleState.IDLE)
            
            if current_state == VehicleState.ASSIGNED:
                # 车辆已分配订单，需要前往取货点
                assigned_order = self.vehicle_assigned_orders.get(vehicle_id)
                if assigned_order:
                    self._handle_pickup_movement(vehicle, assigned_order)
                    
            elif current_state == VehicleState.DELIVERING:
                # 车辆正在配送，需要前往配送点
                assigned_order = self.vehicle_assigned_orders.get(vehicle_id)
                if assigned_order:
                    self._handle_delivery_movement(vehicle, assigned_order)
    
    def _handle_pickup_movement(self, vehicle, order):
        """处理取货阶段的移动"""
        target_node = order.start_node
        current_node = vehicle.current_node
        
        # 检查是否已到达取货点
        if self._is_at_pickup_location(vehicle, order):
            # 开始取货
            if self.vehicle_start_pickup(vehicle.vehicle_id):
                # 立即开始配送
                self.vehicle_start_delivery(vehicle.vehicle_id)
            return
        
        # 🆕 根据车辆类型选择不同的移动方式
        if hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'uav':
            # UAV使用直线飞行
            self._handle_uav_direct_flight(vehicle, target_node, "取货点")
        else:
            # Carrier使用路网移动
            self._handle_carrier_road_movement(vehicle, target_node, "取货点")
    
    def _handle_delivery_movement(self, vehicle, order):
        """处理配送阶段的移动"""
        target_node = order.end_node
        current_node = vehicle.current_node
        
        # 检查是否已到达配送点
        if self._is_at_delivery_location(vehicle, order):
            # 完成配送
            self.vehicle_complete_delivery(vehicle.vehicle_id)
            return
        
        # 🆕 根据车辆类型选择不同的移动方式
        if hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'uav':
            # UAV使用直线飞行
            self._handle_uav_direct_flight(vehicle, target_node, "配送点")
        else:
            # Carrier使用路网移动
            self._handle_carrier_road_movement(vehicle, target_node, "配送点")
    
    def _is_at_pickup_location(self, vehicle, order):
        """检查是否在取货位置"""
        # 检查是否在同一节点
        return vehicle.current_node == order.start_node
    
    def _is_at_delivery_location(self, vehicle, order):
        """检查是否在配送位置"""
        # 检查是否在同一节点
        return vehicle.current_node == order.end_node
    
    def _get_next_move_toward_target(self, vehicle, target_node):
        """获取朝目标的下一步移动"""
        if vehicle.current_node == target_node:
            return target_node
        
        try:
            if self.road_graph is not None:
                # 使用最短路径的第一步
                path = nx.shortest_path(self.road_graph, vehicle.current_node, target_node)
                if len(path) > 1:
                    return path[1]  # 下一个节点
            
            # 回退策略：选择最近的邻居
            neighbors = list(self.road_graph.neighbors(vehicle.current_node)) if self.road_graph is not None else []
            if neighbors:
                best_neighbor = min(neighbors, 
                                  key=lambda n: self._get_distance(n, target_node))
                return best_neighbor
                
        except Exception as e:
            print(f"⚠️ 车辆{vehicle.vehicle_id} 路径规划失败: {e}")
            
        return vehicle.current_node
    
    # def _get_order_distribution_features(self, vehicle) -> List[float]:
    #     """获取车辆视角的订单分布特征"""
    #     features = []
        
    #     if not self.active_orders or self.road_graph is None:
    #         return [0.0] * 8
        
    #     current_node = vehicle.current_node
    #     vehicle_range = (100 if vehicle.vehicle_type == "carrier" else 200)
        
    #     # 1. 能力范围内的订单统计
    #     orders_in_range = 0
    #     total_orders = len(self.active_orders)
    #     urgent_orders_in_range = 0
    #     avg_distance_to_orders = 0
        
    #     order_distances = []
    #     for order in self.active_orders.values():
    #         try:
    #             dist_to_start = nx.shortest_path_length(self.road_graph, current_node, order.start_node, )
    #             if dist_to_start <= vehicle_range:
    #                 orders_in_range += 1
    #                 order_distances.append(dist_to_start)
    #                 if order.priority >= 2:  # 高优先级订单
    #                     urgent_orders_in_range += 1
    #         except:
    #             continue
        
    #     if order_distances:
    #         avg_distance_to_orders = sum(order_distances) / len(order_distances)
        
    #     # 2. 方向性订单分布 (东南西北四个方向)
    #     direction_orders = [0, 0, 0, 0]  # 东、南、西、北
    #     if self.road_graph is not None:
    #         neighbors = list(self.road_graph.neighbors(current_node))
    #         for neighbor in neighbors[:4]:  # 最多检查4个邻居
    #             direction_idx = hash(neighbor) % 4  # 简单的方向映射
    #             nearby_orders = 0
    #             for order in self.active_orders.values():
    #                 try:
    #                     dist = nx.shortest_path_length(self.road_graph, neighbor, order.start_node, )
    #                     if dist <= 50:
    #                         nearby_orders += 1
    #                 except:
    #                     continue
    #             direction_orders[direction_idx] = int(min(nearby_orders / 10.0, 1.0))
        
    #     features = [
    #         orders_in_range / max(total_orders, 1),           # 范围内订单比例
    #         urgent_orders_in_range / max(orders_in_range, 1), # 范围内紧急订单比例
    #         min(avg_distance_to_orders / vehicle_range, 1.0), # 平均距离比例
    #         len(self.active_orders) / 50.0,                   # 总订单密度
    #         direction_orders[0],  # 东方向订单密度
    #         direction_orders[1],  # 南方向订单密度
    #         direction_orders[2],  # 西方向订单密度
    #         direction_orders[3],  # 北方向订单密度
    #     ]
        
    #     return features
    
    def _get_nearby_orders(self, node: int, max_orders: int = 3) -> List[RealRoadOrder]:
        """获取附近的订单"""
        orders_with_dist = []
        
        if self.road_graph is None:
            return []
        
        for order in self.active_orders.values():
            try:
                dist_to_start = nx.shortest_path_length(self.road_graph, node, order.start_node)
                orders_with_dist.append((order, dist_to_start))
            except nx.NetworkXNoPath:
                orders_with_dist.append((order, 1000))
            except Exception:
                continue
        
        orders_with_dist.sort(key=lambda x: x[1])
        return [order for order, _ in orders_with_dist[:max_orders]]
    
    def get_state(self) -> np.ndarray:
        """🆕 获取平台DDPG的全局状态 - 单一状态向量"""
        # 🏗️ 新状态空间结构：
        # 1. 全局订单信息 (最多20个订单 × 6维) = 120维
        # 2. 全局车辆信息 (9个车辆 × 5维) = 45维  
        # 3. 时间和系统状态 (8维)
        # 总计: 120 + 45 + 8 = 173维
        
        state = []
        
        # === 1. 全局订单信息 (120维) ===
        MAX_ORDERS = 20  # 最多考虑20个活跃订单
        active_orders_list = list(self.active_orders.values())[:MAX_ORDERS]
        
        for i in range(MAX_ORDERS):
            if i < len(active_orders_list):
                order = active_orders_list[i]
                # 每个订单6维特征
                state.extend([
                    order.start_node / 600000.0,               # 取货节点(归一化)
                    order.end_node / 600000.0,                 # 配送节点(归一化)  
                    order.start_time / self.max_time_steps,    # 开始时间片(归一化)
                    order.deadline / self.max_time_steps,      # 截止时间片(归一化)
                    order.weight / 10.0,                       # 订单重量(归一化)
                    order.priority / 3.0                       # 订单优先级(归一化)
                ])
            else:
                # 填充空订单
                state.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # === 2. 全局车辆信息 (45维) ===
        for vehicle in self.vehicles:
            # 每个车辆5维特征
            vehicle_state = self.vehicle_states.get(vehicle.vehicle_id, VehicleState.IDLE)
            is_available = 1.0 if vehicle_state == VehicleState.IDLE else 0.0
            
            # UAV能量约束 vs Carrier时间约束
            if vehicle.vehicle_type == 'uav':
                constraint_value = getattr(vehicle, 'battery_level', 1) / getattr(vehicle, 'battery_capacity', 1)
            else:
                # Carrier: 简化处理，设为1.0 (无时间限制)
                constraint_value = 1.0
            
            state.extend([
                vehicle.current_node / 600000.0,              # 车辆位置(归一化)
                1.0 if vehicle.vehicle_type == 'uav' else 0.0, # 车辆类型(UAV=1, Carrier=0)
                getattr(vehicle, 'speed', 45) / 60.0,         # 移动速度(归一化)
                is_available,                                  # 可用状态
                constraint_value                               # 约束状态(电池/时间)
            ])
        
        # === 3. 时间和系统状态 (8维) ===
        system_info = [
            self.time_step / self.max_time_steps,                                    # 时间进度
            len(self.active_orders) / MAX_ORDERS,                                   # 订单负载
            len(self.completed_orders) / max(self.total_orders_generated, 1),      # 完成率
            sum(1 for v in self.vehicles if v.vehicle_id in self.vehicle_assigned_orders) / len(self.vehicles), # 车辆利用率
            self.total_orders_matched / max(self.total_orders_generated, 1),       # 匹配率
            self.uav_energy_failures / max(self.total_orders_generated, 1),        # UAV能量失败率
            self.charging_events / max(self.time_step, 1),                         # 充电频率
            len([v for v in self.vehicles if v.vehicle_type == 'uav']) / len(self.vehicles) # UAV比例
        ]
        state.extend(system_info)
        
        # 确保状态维度为173维
        state_array = np.array(state, dtype=np.float32)
        if len(state_array) != 173:
            if len(state_array) < 173:
                state_array = np.pad(state_array, (0, 173 - len(state_array)), 'constant')
            else:
                state_array = state_array[:173]
        
        return state_array
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """环境步进 - 包含约束检查"""

        # 🆕 全局订单-车辆匹配决策
        assignment_count = self._perform_platform_order_assignment(action)

        # 执行车辆动作 (带约束检查)
        pickup_count, delivery_count = self._execute_automatic_vehicle_behaviors()
        
        # 生成新订单
        new_orders = self.generate_orders()
        
        # 🆕 更新车辆行为（真实移动）
        self._update_vehicle_behaviors()
        
        # 🆕 关键修复：检查DDL超时
        timeout_count = self._check_ddl_timeouts()
        
        # 更新订单状态
        expired_count = self._update_order_status()
        
        # 处理约束相关的维护动作
        self._handle_constraint_maintenance()
        
        # 计算奖励 (包含约束惩罚)
        reward = self._calculate_platform_reward(pickup_count, delivery_count, assignment_count)
        
        # 检查是否结束
        done = self.time_step >= self.max_time_steps - 1
        
        # 更新时间步
        self.time_step += 1
        
        # 获取新状态
        next_states = self.get_state()
        
        # 统计信息 (增加约束相关统计)
        info = {
            'total_orders': self.total_orders_generated,
            'active_orders': len(self.active_orders),
            'completed_orders': len(self.completed_orders),
            'total_generated': self.total_orders_generated,
            'total_completed': self.total_orders_completed,
            'total_matched': self.total_orders_matched,
            'delivery_rate': self.total_orders_completed / max(self.total_orders_generated, 1),
            'match_rate': self.total_orders_matched / max(self.total_orders_generated, 1),
            'new_orders_count': len(new_orders),
            'pickup_count': pickup_count,
            #'assignment_count': assignment_count,  # 🆕 添加订单分配数量
            'delivery_count': delivery_count,
            'expired_count': expired_count,
            'timeout_count': timeout_count,                       # 🆕 添加超时订单数量
            'path_planning_failures': self.path_planning_failures,
            'uav_energy_failures': self.uav_energy_failures,      # 新增
            #'carrier_time_failures': self.carrier_time_failures,  # 新增
            'charging_events': self.charging_events,              # 新增
        }
        
        return next_states, reward, done, info
    
    def _execute_automatic_vehicle_behaviors(self) -> Tuple[int, int]:
        """🆕 执行自动车辆行为 - 无需动作输入，完全状态驱动"""
        # 车辆行为完全由状态机驱动，无需显式动作
        # 车辆移动和任务执行通过_update_vehicle_behaviors()自动处理
        
        # 只需要进行约束状态检查
        for vehicle in self.vehicles:
            if vehicle.vehicle_type == "uav":
                # UAV约束状态检查
                if hasattr(vehicle, "battery_level") and hasattr(vehicle, "battery_capacity"):
                    battery_ratio = vehicle.battery_level / vehicle.battery_capacity
                    if battery_ratio < 0.1:  # 电量低于10%时记录
                        if self.time_step % 10 == 0:  # 每10步提醒一次
                            print(f"⚠️ UAV{vehicle.vehicle_id} 电量低: {battery_ratio:.1%}")
        
        # 返回当前episode的取货和配送统计
        return self.episode_pickups, self.episode_deliveries

    # def _execute_constrained_vehicle_actions(self, actions: List[np.ndarray]) -> Tuple[int, int]:
    #     """执行带约束检查的车辆动作 - 纯订单评分模式（无移动动作）"""
    #     pickup_count = 0
    #     delivery_count = 0
        
    #     # �� 新设计：动作只包含订单评分，不包含移动动作
    #     # 车辆移动完全由任务驱动，通过_update_vehicle_behaviors()自动处理
        
    #     for vehicle, action in zip(self.vehicles, actions):
    #         # 调试：打印动作值（现在只有订单评分）
    #         if self.time_step < 2:  # 只在前2步打印
    #             print(f"   🎮 车辆{vehicle.vehicle_id}: 订单评分动作={action[:3]}")
            
    #         # ✅ 移动逻辑完全移除 - 车辆移动由订单分配后的_update_vehicle_behaviors()自动处理
    #         # ✅ 取货/配送逻辑完全移除 - 由车辆行为状态机自动处理
            
    #         # 1. 约束状态检查（被动检查，不执行动作）
    #         vehicle_state = self.vehicle_states.get(vehicle.vehicle_id, VehicleState.IDLE)
            
    #         # 2. 记录动作用于订单分配（在_perform_global_order_assignment中使用）
    #         # 这里不直接处理动作，只确保车辆状态正常
            
    #         if vehicle.vehicle_type == "uav":
    #             # UAV约束状态检查
    #             if hasattr(vehicle, "battery_level") and hasattr(vehicle, "battery_capacity"):
    #                 battery_ratio = vehicle.battery_level / vehicle.battery_capacity
    #                 if battery_ratio < 0.1:  # 电量低于10%时记录
    #                     if self.time_step % 10 == 0:  # 每10步提醒一次
    #                         print(f"⚠️ UAV{vehicle.vehicle_id} 电量低: {battery_ratio:.1%}")
    #         # else:
    #         #     # Carrier约束状态检查
    #         #     if hasattr(vehicle, "worked_hours") and hasattr(vehicle, "max_work_hours"):
    #         #         work_ratio = vehicle.worked_hours / vehicle.max_work_hours
    #         #         if work_ratio > 0.9:  # 工作时间超过90%时记录
    #         #             if self.time_step % 10 == 0:  # 每10步提醒一次
    #         #                 print(f"⚠️ Carrier{vehicle.vehicle_id} 工作时间将满: {work_ratio:.1%}")
        
    #     # 3. 返回当前episode的取货和配送统计
    #     # 这些统计由车辆行为状态机在其他函数中更新
    #     return self.episode_pickups, self.episode_deliveries
    
    def _handle_constraint_maintenance(self):
        """处理约束相关的维护动作"""
        for vehicle in self.vehicles:
            if vehicle.vehicle_type == 'uav':
                # UAV充电检查
                if hasattr(vehicle, 'needs_return_to_charge'):
                    result = vehicle.needs_return_to_charge()
                    if isinstance(result, tuple):
                        needs_charge, reason = result
                    else:
                        needs_charge = result
                        reason = "Legacy check"
                    if needs_charge:
                        print(f"🔋 UAV {vehicle.vehicle_id} 需要返回充电桩")
                        # 强制返回充电桩，取消当前任务
                        if vehicle.vehicle_id in self.vehicle_assigned_orders:
                            order = self.vehicle_assigned_orders[vehicle.vehicle_id]
                            print(f"⚠️ UAV {vehicle.vehicle_id} 因电量不足取消订单{order.order_id}")
                            self._handle_order_failure(vehicle.vehicle_id, order)
                            self.uav_energy_failures += 1
                        
                        if hasattr(vehicle, 'return_to_charging_station'):
                            success = vehicle.return_to_charging_station()
                            if success:
                                self.charging_events += 1
            # else:
            #     # Carrier工作时间检查
            #     if hasattr(vehicle, 'worked_hours') and hasattr(vehicle, 'max_work_hours'):
            #         if vehicle.worked_hours >= vehicle.max_work_hours:
            #             print(f"⏰ Carrier {vehicle.vehicle_id} 已达工作时间上限")
            #             # 强制结束工作，取消当前任务
            #             if vehicle.vehicle_id in self.vehicle_assigned_orders:
            #                 order = self.vehicle_assigned_orders[vehicle.vehicle_id]
            #                 print(f"⚠️ Carrier {vehicle.vehicle_id} 因工作时间超限取消订单{order.order_id}")
            #                 self._handle_order_failure(vehicle.vehicle_id, order)
            #                 self.carrier_time_failures += 1
                        
            #             # 设置车辆为不可用状态
            #             self.vehicle_states[vehicle.vehicle_id] = VehicleState.IDLE
    
    def _calculate_platform_reward(self, pickup_count: int, delivery_count: int, assignment_count: int) -> float:
        """🆕 平台DDPG单一奖励计算"""
        reward = 0.0
        
        # 1. 基础系统运行奖励
        reward += 1.0  # 基础奖励
        
        # 2. 订单处理奖励（主要奖励来源）
        reward += pickup_count * 3.0      # 取货奖励
        reward += delivery_count * 10.0   # 配送奖励（最高权重）
        reward += assignment_count * 1.5  # 匹配奖励
        
        # 3. 系统效率奖励
        if len(self.active_orders) > 0:
            vehicle_utilization = sum(1 for v in self.vehicles if v.vehicle_id in self.vehicle_assigned_orders) / len(self.vehicles)
            reward += vehicle_utilization * 2.0  # 车辆利用率奖励
            
            order_processing_rate = (pickup_count + delivery_count) / len(self.active_orders)
            reward += order_processing_rate * 1.5  # 订单处理率奖励
        
        # 4. 长期性能奖励
        if self.total_orders_generated > 0:
            completion_rate = self.total_orders_completed / self.total_orders_generated
            reward += completion_rate * 3.0  # 完成率奖励
        
        # 5. 约束管理奖励/惩罚
        # UAV能量管理
        uav_count = len([v for v in self.vehicles if v.vehicle_type == "uav"])
        if uav_count > 0:
            avg_battery = sum(getattr(v, "battery_level", 1) / getattr(v, "battery_capacity", 1) 
                            for v in self.vehicles if v.vehicle_type == "uav") / uav_count
            if avg_battery > 0.8:
                reward += 0.5  # 高电量奖励
            elif avg_battery < 0.2:
                reward -= 1.0  # 低电量惩罚
        
        # 6. 失败惩罚
        reward -= self.uav_energy_failures * 0.5  # 能量失败惩罚
        
        # 7. 时间进度奖励（避免拖延）
        progress_bonus = (1.0 - self.time_step / self.max_time_steps) * 0.2
        reward += progress_bonus
        
        return reward

    # def _calculate_rewards_with_constraints(self, pickup_count: int, delivery_count: int) -> List[float]:
    #     """增强版奖励计算 - 解决奖励稀疏和收敛问题"""
    #     rewards = []
        
    #     # 🆕 增强的全局奖励组件
    #     global_pickup_reward = pickup_count * 2.0  # 从1.0增加到2.0
    #     global_delivery_reward = delivery_count * 8.0  # 从5.0增加到8.0
        
    #     # 🆕 系统效率奖励
    #     system_efficiency = (pickup_count + delivery_count) / max(len(self.active_orders), 1)
    #     efficiency_bonus = system_efficiency * 1.0
        
    #     # 🆕 协作奖励 - 鼓励车辆间协作
    #     active_vehicles = sum(1 for v in self.vehicles if v.vehicle_id in self.vehicle_assigned_orders)
    #     cooperation_bonus = (active_vehicles / self.num_vehicles) * 0.5
        
    #     for vehicle in self.vehicles:
    #         reward = 0.0
            
    #         # 1. 🆕 增强的基础存在奖励
    #         reward += 0.5  # 从0.1增加到0.5
            
    #         # 2. 🆕 个人贡献奖励（70%个人，30%集体）
    #         individual_bonus = 0.0
    #         if vehicle.vehicle_id in self.vehicle_assigned_orders:
    #             individual_bonus += 1.4  # 接单奖励
                
    #         # 检查是否完成配送
    #         if hasattr(vehicle, 'completed_deliveries_this_step'):
    #             individual_bonus += vehicle.completed_deliveries_this_step * 5.6  # 配送奖励
            
    #         reward += individual_bonus
            
    #         # 3. 🆕 集体协作奖励（30%）
    #         collective_reward = (global_pickup_reward + global_delivery_reward) * 0.3 / self.num_vehicles
    #         reward += collective_reward
            
    #         # 4. 🆕 系统效率和协作奖励
    #         reward += efficiency_bonus + cooperation_bonus
            
    #         # 5. 🆕 学习进度奖励
    #         if hasattr(self, 'episode_count'):
    #             progress_bonus = min(0.1 * self.episode_count / 50.0, 0.2)
    #             reward += progress_bonus
            
    #         # 6. 🆕 距离塑形奖励
    #         if vehicle.vehicle_id in self.vehicle_assigned_orders:
    #             order = self.vehicle_assigned_orders[vehicle.vehicle_id]
    #             try:
    #                 if hasattr(order, 'start_node') and self.road_graph:
    #                     if vehicle.current_node in self.road_graph and order.start_node in self.road_graph:
    #                         current_distance = nx.shortest_path_length(
    #                             self.road_graph, vehicle.current_node, order.start_node
    #                         )
    #                         # 距离越近奖励越高
    #                         distance_reward = max(0, 1.0 - current_distance / 20.0) * 0.2
    #                         reward += distance_reward
    #             except:
    #                 pass
            
    #         # 7. 约束管理奖励（保持原有逻辑但调整权重）
    #         if vehicle.vehicle_type == 'uav':
    #             if hasattr(vehicle, 'battery_level') and hasattr(vehicle, 'battery_capacity'):
    #                 battery_ratio = vehicle.battery_level / vehicle.battery_capacity
    #                 if battery_ratio > 0.8:
    #                     reward += 0.3  # 增加高电量奖励
    #                 elif battery_ratio < 0.2:
    #                     reward -= 0.8  # 增加低电量惩罚
    #         else:
    #             if hasattr(vehicle, 'worked_hours') and hasattr(vehicle, 'max_work_hours'):
    #                 work_ratio = vehicle.worked_hours / vehicle.max_work_hours
    #                 if work_ratio < 0.7:
    #                     reward += 0.2  # 时间充足奖励
    #                 elif work_ratio > 0.9:
    #                     reward -= 0.5  # 时间不足惩罚
            
    #         rewards.append(reward)
        
    #     return rewards
    
    # def _calculate_movement_reward(self, vehicle) -> float:
    #     """计算智能移动奖励 - 奖励朝订单移动的行为"""
    #     if not hasattr(vehicle, "previous_node") or not self.active_orders:
    #         vehicle.previous_node = vehicle.current_node
    #         return 0.0
        
    #     current_node = vehicle.current_node
    #     previous_node = getattr(vehicle, "previous_node", current_node)
    #     vehicle.previous_node = current_node
        
    #     if current_node == previous_node or self.road_graph is None:
    #         return 0.0
        
    #     # 计算移动前后到最近订单的距离变化
    #     vehicle_range = (100 if vehicle.vehicle_type == "carrier" else 200)
        
    #     min_prev_dist = float("inf")
    #     min_curr_dist = float("inf")
        
    #     for order in list(self.active_orders.values())[:5]:  # 只检查前5个订单，避免计算过多
    #         try:
    #             # 计算从之前位置到订单的距离
    #             prev_dist = nx.shortest_path_length(self.road_graph, previous_node, order.start_node)
    #             if prev_dist <= vehicle_range:
    #                 min_prev_dist = min(min_prev_dist, prev_dist)
                
    #             # 计算从当前位置到订单的距离
    #             curr_dist = nx.shortest_path_length(self.road_graph, current_node, order.start_node)
    #             if curr_dist <= vehicle_range:
    #                 min_curr_dist = min(min_curr_dist, curr_dist)
    #         except:
    #             continue
        
    #     # 如果找不到可达订单，给予小的探索奖励
    #     if min_prev_dist == float("inf") or min_curr_dist == float("inf"):
    #         return 0.05  # 小的探索奖励
        
    #     # 计算距离改善奖励
    #     distance_improvement = min_prev_dist - min_curr_dist
    #     if distance_improvement > 0:
    #         # 朝订单移动，给予奖励
    #         return min(distance_improvement * 0.01, 0.2)  # 最大0.2的奖励
    #     elif distance_improvement < 0:
    #         # 远离订单，给予小惩罚
    #         return max(distance_improvement * 0.005, -0.1)  # 最大-0.1的惩罚
    #     else:
    #         return 0.0
    
    def _update_order_status(self) -> int:
        """更新订单状态，清理超时订单"""
        expired_orders = []
        for order in self.active_orders.values():
            if self.time_step > order.deadline:
                expired_orders.append(order.order_id)
        
        for order_id in expired_orders:
            del self.active_orders[order_id]
        
        return len(expired_orders)
    
    def _handle_uav_direct_flight(self, vehicle, target_node, target_type):
        """🆕 处理UAV直线飞行 - 集成能量管理"""
        current_node = vehicle.current_node
        
        if current_node == target_node:
            return
        
        # 🆕 获取当前载重（用于能量计算）
        current_payload = 0.5 if (hasattr(vehicle, 'vehicle_state') and 
                                 self.vehicle_states.get(vehicle.vehicle_id) == VehicleState.DELIVERING) else 0.0
        
        # 🆕 使用增强版UAV的飞行时间和能量计算
        if hasattr(vehicle, 'calculate_flight_time_to_node') and hasattr(vehicle, 'fly_to_node_with_energy_management'):
            # 使用增强版UAV的方法
            flight_success = vehicle.fly_to_node_with_energy_management(target_node, current_payload)
            
            if flight_success:
                print(f"🚁 UAV{vehicle.vehicle_id} 到达{target_type}: {target_node}")
                # 使用现有的电池比例计算（避免重复函数）
                battery_ratio = vehicle.battery_level / vehicle.battery_capacity
                print(f"   ⚡ 剩余电量: {vehicle.battery_level:.1f}kWh ({battery_ratio*100:.1f}%)")
                
                # 🆕 检查是否需要充电 - 使用现有方法
                need_charge, reason = vehicle.needs_return_to_charge()
                if need_charge:
                    print(f"⚠️ UAV{vehicle.vehicle_id} 需要充电: {reason}")
                    # 使用现有的返回充电桩方法（避免重复函数）
                    charge_success = vehicle.return_to_charging_station()
                    if charge_success:
                        self.charging_events += 1
            else:
                print(f"❌ UAV{vehicle.vehicle_id} 飞行失败：电量不足")
                # 强制返回充电桩
                self._force_uav_return_to_charge(vehicle)
        else:
            # 回退到原始实现（如果UAV类没有增强版方法）
            self._handle_uav_legacy_flight(vehicle, target_node, target_type)
    
    def _handle_uav_legacy_flight(self, vehicle, target_node, target_type):
        """原始UAV飞行实现（用作回退）"""
        current_node = vehicle.current_node
        
        if current_node == target_node:
            return
        
        # 计算直线距离
        distance = self._calculate_euclidean_distance(current_node, target_node)
        
        # UAV飞行速度 (假设30km/h)
        uav_speed = 60.0  # km/h
        
        # 计算飞行时间 (小时)
        flight_time = distance / uav_speed
        
        # 转换为时间步 (假设每个时间步是5分钟)
        time_steps_needed = max(1, int(flight_time * 12))  # 12 = 60min/5min
        
        # 检查是否有足够的飞行时间记录
        if not hasattr(vehicle, 'flight_remaining_steps'):
            vehicle.flight_remaining_steps = 0
            vehicle.flight_target = None
        
        # 开始新的飞行或继续当前飞行
        if vehicle.flight_target != target_node:
            # 开始新飞行
            vehicle.flight_remaining_steps = time_steps_needed
            vehicle.flight_target = target_node
            print(f"🚁 UAV{vehicle.vehicle_id} 开始直线飞行到{target_type}: {current_node} → {target_node} (预计{time_steps_needed}步)")
        
        # 减少剩余飞行时间
        vehicle.flight_remaining_steps -= 1
        
        # 检查是否到达目标
        if vehicle.flight_remaining_steps <= 0:
            vehicle.current_node = target_node
            vehicle.flight_target = None
            print(f"🚁 UAV{vehicle.vehicle_id} 到达{target_type}: {target_node}")
        else:
            print(f"🚁 UAV{vehicle.vehicle_id} 飞行中...剩余{vehicle.flight_remaining_steps}步到达{target_type}")
    
    def _force_uav_return_to_charge(self, vehicle):
        """🆕 强制UAV返回充电桩"""
        print(f"🔋 强制UAV{vehicle.vehicle_id}返回充电桩: {vehicle.charging_station_node}")
        
        # 重置车辆状态
        self.vehicle_states[vehicle.vehicle_id] = VehicleState.IDLE
        
        # 清除分配的订单（如果有）
        if vehicle.vehicle_id in self.vehicle_assigned_orders:
            failed_order = self.vehicle_assigned_orders[vehicle.vehicle_id]
            print(f"❌ 订单{failed_order.order_id}因电量不足被取消")
            del self.vehicle_assigned_orders[vehicle.vehicle_id]
            # 记录能量失败
            self.uav_energy_failures += 1
        
        # 立即移动到充电桩并充电
        vehicle.current_node = vehicle.charging_station_node
        # 使用UAV自己的充电逻辑（避免重复函数）
        old_level = vehicle.battery_level
        vehicle.battery_level = vehicle.battery_capacity
        print(f"🔋 UAV{vehicle.vehicle_id}已充电完成: {old_level:.1f}→{vehicle.battery_level}kWh")
    
    
    def _handle_carrier_road_movement(self, vehicle, target_node, target_type):
        """处理Carrier路网移动 - 优化速度"""
        current_node = vehicle.current_node
        
        if current_node == target_node:
            return
        
        # 🆕 Carrier每步可以移动50个节点（提高速度）
        moves_per_step = 50
        path_taken = [current_node]
        
        for step in range(moves_per_step):
            next_node = self._get_next_move_toward_target(vehicle, target_node)
            if next_node and next_node != vehicle.current_node and next_node != target_node:
                vehicle.current_node = next_node
                path_taken.append(next_node)
            elif next_node == target_node:
                vehicle.current_node = target_node
                path_taken.append(target_node)
                break
            else:
                break
        
        if len(path_taken) > 1:
            print(f"🚚 Carrier{vehicle.vehicle_id} 向{target_type}移动: {path_taken[0]} → {path_taken[-1]} (移动{len(path_taken)-1}步)")
    
    #路径-时间转换1
    def _calculate_euclidean_distance(self, node1, node2):
        return distance_calculator.get_straight_line_distance(node1, node2)
    
    
    def reset(self, day: Optional[int] = None) -> List[np.ndarray]:
        """重置环境"""
        self.time_step = 0
        self.active_orders = {}
        self.completed_orders = []
        
        # 🆕 加载订单数据（修复关键问题）
        if not hasattr(self, 'orders_data') or not self.orders_data:
            self.load_road_orders(day=day if day is not None else 1)  # 动态加载指定天数数据
        
        # # 🆕 加载订单数据（修复关键问题）
        # if not hasattr(self, 'orders_data') or not self.orders_data:
        #     self.load_road_orders(day=1)  # 默认加载第0天数据
        
        # 重置统计
        self.total_orders_generated = 0
        self.total_orders_completed = 0
        self.total_orders_matched = 0
        self.path_planning_failures = 0
        self.uav_energy_failures = 0
        #self.carrier_time_failures = 0
        self.charging_events = 0

        # 📊 Episode统计计数器
        self.episode_pickups = 0           # 本episode取货次数
        self.episode_deliveries = 0        # 本episode送货次数
        # max_concurrent_orders already set in __init__

        # 🆕 重置状态跟踪
        self.vehicle_states = {}
        self.vehicle_assigned_orders = {}
        self.order_statuses = {}
        
        # 重置车辆状态
        if self.largest_component_nodes is not None and len(self.largest_component_nodes) >= self.num_vehicles:
            start_nodes = np.random.choice(self.largest_component_nodes, self.num_vehicles, replace=False)
        elif self.largest_component_nodes is not None:
            start_nodes = np.random.choice(self.largest_component_nodes, self.num_vehicles, replace=True)
        else:
            available_nodes = list(self.road_graph.nodes())[:self.num_vehicles] if self.road_graph else [0] * self.num_vehicles
            start_nodes = available_nodes
        
        for i, vehicle in enumerate(self.vehicles):
            vehicle.current_node = start_nodes[i]
            
            # 重置车辆特定状态
            if vehicle.vehicle_type == 'uav':
                # 重置UAV状态
                if hasattr(vehicle, 'battery_level') and hasattr(vehicle, 'battery_capacity'):
                    vehicle.battery_level = vehicle.battery_capacity  # 满电
            else:
                # 重置Carrier状态
                if hasattr(vehicle, 'worked_hours'):
                    vehicle.worked_hours = 0  # 重置工作时间

            # 🆕 重置状态管理
            self.vehicle_states[vehicle.vehicle_id] = VehicleState.IDLE
            
            # 清空订单
            if hasattr(vehicle, 'assigned_orders'):
                pass
            
            # 保持兼容性
            vehicle.current_grid = vehicle.current_node
        
        print(f"🔄 带约束环境重置完成，车辆位置: {[v.current_node for v in self.vehicles]}")
        return self.get_state()



def test_constrained_environment():
    """测试带约束的环境"""
    print("🧪 测试带约束的纯路网环境...")
    
    try:
        # 创建环境
        env = PureRealRoadNetworkEnvironmentWithConstraints(num_ground_vehicles=2, num_uavs=2)
        
        # 加载订单数据
        success = env.load_road_orders(1)
        if not success:
            print("❌ 订单数据加载失败")
            return False
        
        # 重置环境
        states = env.reset()
        print(f"✅ 环境重置成功，状态维度: {len(states)} x {len(states[0])}")
        
        # 验证状态维度
        expected_dim = 52  # 修正：实际状态维度是52  
        if len(states[0]) == expected_dim:
            print(f"✅ 状态维度正确: {expected_dim}")
        else:
            print(f"❌ 状态维度错误: {len(states[0])} != {expected_dim}")
        
        # 运行几步测试约束
        for step in range(10):
            actions = []
            for i in range(len(states)):
                action = np.array([0.8, 0.9, 0.9] + [0.5] * 33)
                actions.append(action)
            
            next_states, rewards, done, info = env.step(actions)
            
            print(f"步骤 {step:2d}: "
                  f"奖励={rewards:5.1f}, "
                  f"订单={info.get('new_orders_count', 0)}, "
                  f"活跃={info.get('active_orders', 0)}, "
                  f"能量失败={info.get('uav_energy_failures', 0)}, "
                  #f"时间失败={info.get('carrier_time_failures', 0)}, "
                  f"充电={info.get('charging_events', 0)}")
            
            states = next_states
            
            if done:
                break
        
        print("✅ 带约束环境测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 带约束环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_constrained_environment()

    

