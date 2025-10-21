#!/usr/bin/env python3
"""
路网车辆适配器 - 增强版UAV能量管理系统
实现三阶段能量消耗模型：空载取货、载重配送、空载返回/充电
"""

import numpy as np
import networkx as nx
import sys
import os

# 添加shared_components路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_components'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_components', 'simulator'))

from simulator.objects import UAV, Carrier, Order, Vehicle
from distance_utils import distance_calculator


from enum import Enum

class VehicleState(Enum):
    """车辆工作状态枚举"""
    IDLE = "idle"                    # 空闲状态，可以接受新订单
    ASSIGNED = "assigned"            # 已分配订单，前往取货
    DELIVERING = "delivering"        # 配送中，不能接受新订单
    CHARGING = "charging"            # 充电中，无法接受订单

class RoadNetworkUAV(UAV):
    """路网无人机 - 增强能量管理"""
    
    def __init__(self, uav_id, start_node, road_graph, capacity=5, battery_capacity=50, 
                 charging_station_node=None, charging_stations_list=None, speed=60):
        # 使用虚拟网格ID初始化父类 (为了兼容)
        super().__init__(uav_id, start_node, capacity, battery_capacity, 
                         charging_station_node or start_node, speed)
        
        # 路网特有属性
        self.current_node = start_node  # 当前路网节点
        self.road_graph = road_graph    # 路网图引用
        self.charging_station_node = charging_station_node or start_node  # 充电桩节点
        # 处理numpy数组的充电桩列表
        if charging_stations_list is not None:
            self.charging_stations_list = list(charging_stations_list)  # 转为列表
        else:
            self.charging_stations_list = [charging_station_node or start_node]  # 默认充电桩列表
        
        # 重新设置类型标识
        self.vehicle_type = 'uav'
        self.start_grid = start_node  # 保持兼容性
        self.current_grid = start_node
        
        # 🆕 优化的能量消耗参数 - 基于50kWh电池和0.5kg标准载重
        self.battery_capacity = battery_capacity  # 使用50kWh
        self.battery_level = battery_capacity     # 初始满电
        self.base_power = 0.32                     # kW (空载功率，适中设置)
        self.load_power_factor = 0.08              # kW/kg (载重功率因子，体现载重影响)
        self.safety_margin = 0.25                 # 25%安全电量阈值
        self.standard_payload = 0.5               # kg (标准货物重量)
        
        # 兼容性属性 - 环境需要这些属性
        self.current_load = 0
        self.orders = []
        self.target_node = None
        self.path = []
        self.path_index = 0

        # 🆕 三阶段能量管理状态
        self.vehicle_state = VehicleState.IDLE
        self.assigned_order = None
        self.pickup_deadline = None
        self.delivery_deadline = None
        self.energy_debug = True  # 开启调试信息
        
        # 🆕 能量统计
        self.total_energy_consumed = 0.0
        self.total_flight_time = 0.0
        self.completed_orders_count = 0
        
        print(f"🚁 UAV {uav_id} 初始化: 节点={start_node}, 电池={battery_capacity}kWh, 充电桩={self.charging_station_node}")
        print(f"   能量参数: 空载={self.base_power}kW, 载重因子={self.load_power_factor}kW/kg, 安全阈值={self.safety_margin*100}%")

    # def calculate_flight_time_to_node(self, target_node):
    #     """计算飞行到目标节点的时间(小时) - UAV使用直线距离"""
    #     node_dist = abs(self.current_node - target_node)
    #     estimated_km = node_dist * 0.3  # 假设每个节点间距300米 = 0.3km
    #     return estimated_km / self.speed
    
    # def calculate_flight_time_to_node(self, target_node):
    #     """计算飞行到目标节点的时间(小时) - UAV使用直线距离"""
    #     # UAV可以直线飞行，不受路网限制
    #     if hasattr(self, 'node_coordinates') and target_node in self.node_coordinates:
    #         # 如果有坐标信息，使用真实直线距离
    #         current_coord = self.node_coordinates.get(self.current_node)
    #         target_coord = self.node_coordinates.get(target_node)
    #         if current_coord and target_coord:
    #             # 使用haversine公式计算真实地理距离
    #             from math import radians, cos, sin, asin, sqrt
    #             lon1, lat1 = current_coord
    #             lon2, lat2 = target_coord
    #             lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    #             dlon = lon2 - lon1
    #             dlat = lat2 - lat1
    #             a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    #             c = 2 * asin(sqrt(a))
    #             dist_km = c * 6371  # 地球半径6371km
    #             return dist_km / self.speed
        
    #     # 使用节点差值估算直线距离（更保守的估算）
    #     node_dist = abs(self.current_node - target_node)
    #     estimated_km = node_dist * 0.0001  # 基于真实边长度的估算
    #     return estimated_km / self.speed
    
    def calculate_flight_time_to_node(self, target_node):
        """计算飞行到目标节点的时间(小时) - UAV使用直线距离"""
        # 使用统一的距离计算器获取直线距离
        dist_km = distance_calculator.get_straight_line_distance(self.current_node, target_node)
        return dist_km / self.speed

    def calculate_enhanced_energy_consumption(self, flight_time_hours, payload_kg=0.0):
        """🆕 增强版能量消耗计算 - 基于三阶段模型"""
        base_energy = flight_time_hours * self.base_power
        load_energy = flight_time_hours * self.load_power_factor * payload_kg
        total_energy = base_energy + load_energy
        
        if self.energy_debug:
            print(f"   ⚡ UAV {self.vehicle_id} 能量计算: {flight_time_hours:.3f}h × ({self.base_power}kW + {self.load_power_factor}×{payload_kg}kg) = {total_energy:.3f}kWh")
        
        return total_energy

    def can_complete_three_phase_mission(self, order):
        """🆕 三阶段任务可行性检查"""
        if not hasattr(order, 'start_node') or not hasattr(order, 'end_node'):
            return False, "订单缺少节点信息"
        
        # 阶段1：空载飞行到取货点
        pickup_time = self.calculate_flight_time_to_node(order.start_node)
        pickup_energy = self.calculate_enhanced_energy_consumption(pickup_time, 0.0)
        
        # 阶段2：载重配送到终点
        # 从取货点到配送点的时间计算（UAV直线飞行）
        delivery_time = distance_calculator.get_straight_line_distance(order.start_node, order.end_node) / self.speed
        #delivery_time = pickup_to_delivery_dist / self.speed
        payload = getattr(order, 'weight', self.standard_payload)
        delivery_energy = self.calculate_enhanced_energy_consumption(delivery_time, payload)
        
        # 阶段3：空载返回充电桩
        # 从配送点到充电桩的时间计算
        return_time = distance_calculator.get_straight_line_distance(order.end_node, self.charging_station_node) / self.speed
        #return_time = delivery_to_charging_dist / self.speed
        return_energy = self.calculate_enhanced_energy_consumption(return_time, 0.0)
        
        # 总能量需求
        total_energy_required = pickup_energy + delivery_energy + return_energy
        total_time = pickup_time + delivery_time + return_time
        
        # 安全检查：加20%安全余量
        safe_energy_required = total_energy_required * 1.2
        
        if self.energy_debug:
            print(f"   📊 UAV {self.vehicle_id} 三阶段能量分析 订单{order.order_id}:")
            print(f"      阶段1 空载取货: {pickup_time:.3f}h → {pickup_energy:.3f}kWh")
            print(f"      阶段2 载重配送: {delivery_time:.3f}h × {payload}kg → {delivery_energy:.3f}kWh")
            print(f"      阶段3 空载返回: {return_time:.3f}h → {return_energy:.3f}kWh")
            print(f"      总计: {total_time:.3f}h → {total_energy_required:.3f}kWh (含安全余量: {safe_energy_required:.3f}kWh)")
            print(f"      当前电量: {self.battery_level:.3f}kWh, 可行性: {self.battery_level >= safe_energy_required}")
        
        feasible = self.battery_level >= safe_energy_required
        reason = "可行" if feasible else f"电量不足 ({self.battery_level:.1f} < {safe_energy_required:.1f} kWh)"
        
        return feasible, reason

    def execute_flight_phase(self, target_node, payload_kg=0.0, phase_name="飞行"):
        """🆕 执行单阶段飞行并更新电量"""
        if target_node == self.current_node:
            return True, "已在目标位置"
        
        flight_time = self.calculate_flight_time_to_node(target_node)
        energy_cost = self.calculate_enhanced_energy_consumption(flight_time, payload_kg)
        
        # 检查是否有足够电量
        if self.battery_level < energy_cost:
            return False, f"电量不足 ({self.battery_level:.1f} < {energy_cost:.1f} kWh)"
        
        # 执行飞行
        old_node = self.current_node
        self.current_node = target_node
        self.current_grid = target_node  # 保持兼容性
        
        # 更新电量和统计
        self.battery_level -= energy_cost
        self.total_energy_consumed += energy_cost
        self.total_flight_time += flight_time
        
        if self.energy_debug:
            print(f"🚁 UAV {self.vehicle_id} {phase_name}: {old_node}→{target_node}, 载重={payload_kg}kg")
            print(f"   消耗: {energy_cost:.3f}kWh, 剩余: {self.battery_level:.3f}kWh ({self.battery_level/self.battery_capacity*100:.1f}%)")
        
        return True, "飞行成功"

    def find_nearest_charging_station(self):
        """找到距离当前位置最近的充电桩"""
        if not self.charging_stations_list:
            return self.charging_station_node
        
        current_pos = self.current_node
        min_distance = float('inf')
        nearest_station = self.charging_stations_list[0]
        
        # for station in self.charging_stations_list:
        #     # 使用节点ID差值计算距离（简化）
        #     distance = abs(station - current_pos)
        #     if distance < min_distance:
        #         min_distance = distance
        #         nearest_station = station

        for station in self.charging_stations_list:
            # 使用统一的距离计算器
            distance = distance_calculator.get_straight_line_distance(current_pos, station)
            if distance < min_distance:
                min_distance = distance
                nearest_station = station
        
        return nearest_station

    def update_charging_station(self):
        """更新到最近的充电桩"""
        nearest = self.find_nearest_charging_station()
        if nearest != self.charging_station_node:
            old_station = self.charging_station_node
            self.charging_station_node = nearest
            print(f"🔄 UAV {self.vehicle_id} 切换充电桩: {old_station} → {nearest}")
        return self.charging_station_node

    def needs_return_to_charge(self, next_order=None):
        """🆕 增强版充电需求检查"""
        # 基础电量检查
        battery_ratio = self.battery_level / self.battery_capacity
        if battery_ratio <= self.safety_margin:
            return True, f"电量低于安全阈值 ({battery_ratio*100:.1f}% <= {self.safety_margin*100}%)"
        
        # 如果有待执行订单，检查是否能完成
        if next_order:
            can_complete, reason = self.can_complete_three_phase_mission(next_order)
            if not can_complete:
                return True, f"无法完成下一订单: {reason}"
        
        # 检查是否能安全返回充电桩
        # 使用最近的充电桩计算返回时间
        nearest_station = self.find_nearest_charging_station()
        return_time = self.calculate_flight_time_to_node(nearest_station)
        return_energy = self.calculate_enhanced_energy_consumption(return_time, 0.0) * 1.1  # 10%安全余量
        
        if self.battery_level < return_energy:
            return True, f"无法安全返回充电桩 ({self.battery_level:.1f} < {return_energy:.1f} kWh)"
        
        return False, "电量充足"

    def return_to_charging_station(self):
        """返回充电桩并充电"""
        # 选择最近的充电桩
        self.update_charging_station()

        if self.current_node == self.charging_station_node:
            # 已在充电桩，直接充电
            old_level = self.battery_level
            self.battery_level = self.battery_capacity
            self.vehicle_state = VehicleState.IDLE  # 充电后回到空闲状态
            print(f"🔋 UAV {self.vehicle_id} 在充电桩充电: {old_level:.1f}→{self.battery_level:.1f}kWh")
            return True
        
        # 尝试飞回充电桩
        success, message = self.execute_flight_phase(self.charging_station_node, 0.0, "返回充电桩")
        
        if success:
            # 充电
            old_level = self.battery_level
            self.battery_level = self.battery_capacity
            self.vehicle_state = VehicleState.IDLE
            print(f"🔋 UAV {self.vehicle_id} 返回充电桩并充电: {old_level:.1f}→{self.battery_level:.1f}kWh")
            return True
        else:
            print(f"❌ UAV {self.vehicle_id} 无法返回充电桩: {message}")
            return False

    def can_deliver_order_safely(self, order):
        """兼容性方法 - 使用新的三阶段检查"""
        can_complete, reason = self.can_complete_three_phase_mission(order)
        if self.energy_debug and not can_complete:
            print(f"   ❌ UAV {self.vehicle_id} 无法安全配送订单 {order.order_id}: {reason}")
        return can_complete
    
    def fly_to_node_with_energy_management(self, target_node, payload_kg=0.0):
        """🆕 带能量管理的飞行方法 - 环境调用接口"""
        success, reason = self.execute_flight_phase(target_node, payload_kg, "能量管理飞行")
        return success

    def get_energy_statistics(self):
        """获取能量统计信息"""
        if self.completed_orders_count > 0:
            avg_energy_per_order = self.total_energy_consumed / self.completed_orders_count
            avg_time_per_order = self.total_flight_time / self.completed_orders_count
        else:
            avg_energy_per_order = 0
            avg_time_per_order = 0
            
        return {
            'battery_level': self.battery_level,
            'battery_percentage': self.battery_level / self.battery_capacity * 100,
            'total_energy_consumed': self.total_energy_consumed,
            'total_flight_time': self.total_flight_time,
            'completed_orders': self.completed_orders_count,
            'avg_energy_per_order': avg_energy_per_order,
            'avg_time_per_order': avg_time_per_order,
            'needs_charging': self.needs_return_to_charge()[0]
        }

class RoadNetworkCarrier:
    """路网地面载体 - 基于工作时间约束的载重车辆"""
    
    def __init__(self, carrier_id, start_node, road_graph, capacity=float('inf'), 
                 speed=45, range_limit=50, max_work_hours=8):
        """初始化路网载体"""
        # 基本属性
        self.vehicle_id = carrier_id
        self.vehicle_type = 'carrier'  # 类型标识
        self.start_node = start_node
        self.current_node = start_node
        self.road_graph = road_graph
        
        # 车辆性能参数
        self.capacity = capacity  # 载重容量（通常设为无限）
        self.speed = speed  # 平均速度 km/h
        self.range_limit = range_limit  # 工作范围限制 km
        
        # 🆕 工作时间约束（Carrier的核心约束）
        self.max_work_hours = max_work_hours  # 最大工作时间
        self.worked_hours = 0.0  # 当前已工作时间
        
        # 兼容性属性
        self.current_load = 0
        self.start_grid = start_node  # 与UAV保持一致的接口
        self.current_grid = start_node
        
        print(f"🚚 Carrier {carrier_id} 初始化: 节点={start_node}, 工作时间限制={max_work_hours}h")
    
    # #路径-时间转换2
    # def calculate_road_travel_time(self, from_node, to_node):
    #     """计算路网中两点间的旅行时间（小时）"""
    #     if from_node == to_node:
    #         return 0.0
        
    #     try:
    #         if self.road_graph is not None:
    #             # 使用路网最短路径长度
    #             distance = nx.shortest_path_length(self.road_graph, from_node, to_node)
    #             # 转换为实际距离（假设每个路径单位代表100米）
    #             distance_km = distance * 0.1  # 路径长度转为km
    #             travel_time = distance_km / self.speed  # 时间 = 距离/速度
    #             return travel_time
    #         else:
    #             # 回退策略：基于节点差值估算
    #             node_dist = abs(from_node - to_node)
    #             estimated_km = node_dist * 0.00001  # 基于真实节点间距估算
    #             return estimated_km / self.speed
                
    #     except nx.NetworkXNoPath:
    #         print(f"⚠️ Carrier{self.vehicle_id}: 无路径从{from_node}到{to_node}")
    #         return float('inf')  # 无法到达
    #     except Exception as e:
    #         print(f"⚠️ Carrier{self.vehicle_id}: 路径计算错误 {e}")
    #         # 使用保守估算
    #         return abs(from_node - to_node) * 0.0001  # 保守的时间估算
        
    def calculate_road_travel_time(self, from_node, to_node):
        """计算路网中两点间的旅行时间（小时）"""
        if from_node == to_node:
            return 0.0
        try:
            # 使用统一的距离计算器获取路网距离
            distance_km = distance_calculator.get_road_network_distance(from_node, to_node)
            travel_time = distance_km / self.speed  # 时间 = 距离/速度 
            return travel_time
        except Exception as e:
            print(f"⚠️ Carrier{self.vehicle_id}: 路径计算错误 {e}")
            return float("inf")  # 无法到达
    
    def can_deliver_order_within_work_hours(self, road_order):
        """检查是否能在工作时间限制内完成订单"""
        try:
            # 计算完成订单所需的总时间
            pickup_time = self.calculate_road_travel_time(self.current_node, road_order.start_node)
            delivery_time = self.calculate_road_travel_time(road_order.start_node, road_order.end_node)
            
            # 总工作时间 = 当前已工作时间 + 取货时间 + 配送时间
            total_work_time = self.worked_hours + pickup_time + delivery_time
            
            # 检查是否超过工作时间限制
            if total_work_time <= self.max_work_hours:
                remaining_time = self.max_work_hours - total_work_time
                print(f"   ✅ Carrier{self.vehicle_id} 工作时间检查通过，剩余时间: {remaining_time:.2f}h")
                return True
            else:
                overtime = total_work_time - self.max_work_hours
                print(f"   ❌ Carrier{self.vehicle_id} 工作时间不足，超时: {overtime:.2f}h")
                return False
                
        except Exception as e:
            print(f"   ⚠️ Carrier{self.vehicle_id} 工作时间检查出错: {e}")
            return False
    
    def update_work_time(self, additional_hours):
        """更新工作时间"""
        self.worked_hours += additional_hours
        print(f"🕐 Carrier{self.vehicle_id} 工作时间更新: +{additional_hours:.2f}h → 总计{self.worked_hours:.2f}h/{self.max_work_hours}h")
    
    def is_available_for_work(self):
        """检查是否还能继续工作"""
        return self.worked_hours < self.max_work_hours
    
    def get_remaining_work_time(self):
        """获取剩余工作时间"""
        return max(0, self.max_work_hours - self.worked_hours)
    
    def reset_work_time(self):
        """重置工作时间（用于新的工作日）"""
        old_hours = self.worked_hours
        self.worked_hours = 0.0
        print(f"🔄 Carrier{self.vehicle_id} 工作时间重置: {old_hours:.2f}h → 0.0h")


class RoadNetworkOrder:
    """路网订单适配器"""
    
    def __init__(self, order_id, start_node, end_node, start_time, deadline, 
                 weight=1.0, priority=1, created_time=0):
        # 路网特有属性
        self.order_id = order_id
        self.start_node = start_node
        self.end_node = end_node
        self.start_time = start_time
        self.deadline = deadline
        self.weight = weight
        self.priority = priority
        self.created_time = created_time
        
        # 保持兼容性
        self.start_grid = start_node
        self.end_grid = end_node

def test_enhanced_uav_energy():
    """测试增强版UAV能量管理"""
    print("🧪 测试增强版UAV能量管理...")
    
    # 创建简单测试路网
    test_graph = nx.Graph()
    test_graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    
    # 测试UAV
    uav = RoadNetworkUAV(1, 1, test_graph, battery_capacity=50, charging_station_node=1)
    
    # 创建测试订单
    test_order = RoadNetworkOrder(1, 2, 4, 0, 100, weight=0.5)
    
    print(f"\n📊 初始状态:")
    stats = uav.get_energy_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n🧪 测试订单可行性:")
    can_complete, reason = uav.can_complete_three_phase_mission(test_order)
    print(f"结果: {can_complete} - {reason}")
    
    if can_complete:
        print(f"\n🚁 执行模拟配送:")
        # 阶段1: 空载到取货点
        success1, msg1 = uav.execute_flight_phase(test_order.start_node, 0.0, "空载取货")
        print(f"取货阶段: {success1} - {msg1}")
        
        # 阶段2: 载重配送
        success2, msg2 = uav.execute_flight_phase(test_order.end_node, test_order.weight, "载重配送")
        print(f"配送阶段: {success2} - {msg2}")
        
        # 完成订单 
        uav.completed_orders_count += 1
        
        # 阶段3: 返回充电桩
        success3 = uav.return_to_charging_station()
        print(f"返回充电: {success3}")
        
        print(f"\n📈 最终统计:")
        final_stats = uav.get_energy_statistics()
        for key, value in final_stats.items():
            print(f"   {key}: {value}")
    
    print("✅ 增强版UAV能量管理测试完成")

if __name__ == "__main__":
    test_enhanced_uav_energy()

    



