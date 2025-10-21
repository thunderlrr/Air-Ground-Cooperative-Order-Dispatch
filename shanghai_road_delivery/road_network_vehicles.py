#!/usr/bin/env python3

import numpy as np
import networkx as nx
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_components'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_components', 'simulator'))

from simulator.objects import UAV, Carrier, Order, Vehicle
from distance_utils import distance_calculator

from enum import Enum

class VehicleState(Enum):
    IDLE = "idle"
    ASSIGNED = "assigned"
    DELIVERING = "delivering"
    CHARGING = "charging"

class RoadNetworkUAV(UAV):
    
    def __init__(self, uav_id, start_node, road_graph, capacity=5, battery_capacity=50, 
                 charging_station_node=None, charging_stations_list=None, speed=60):
        super().__init__(uav_id, start_node, capacity, battery_capacity, 
                         charging_station_node or start_node, speed)
        
        self.current_node = start_node
        self.road_graph = road_graph
        self.charging_station_node = charging_station_node or start_node
        if charging_stations_list is not None:
            self.charging_stations_list = list(charging_stations_list)
        else:
            self.charging_stations_list = [charging_station_node or start_node]
        
        self.vehicle_type = 'uav'
        self.start_grid = start_node
        self.current_grid = start_node
        
        self.battery_capacity = battery_capacity
        self.battery_level = battery_capacity
        self.base_power = 0.32
        self.load_power_factor = 0.08
        self.safety_margin = 0.25
        self.standard_payload = 0.5
        
        self.current_load = 0
        self.orders = []
        self.target_node = None
        self.path = []
        self.path_index = 0

        self.vehicle_state = VehicleState.IDLE
        self.assigned_order = None
        self.pickup_deadline = None
        self.delivery_deadline = None
        self.energy_debug = True
        
        self.total_energy_consumed = 0.0
        self.total_flight_time = 0.0
        self.completed_orders_count = 0
        
        print(f"UAV {uav_id} initialized: node={start_node}, battery={battery_capacity}kWh, charging_station={self.charging_station_node}")
        print(f"Energy parameters: base_power={self.base_power}kW, load_factor={self.load_power_factor}kW/kg, safety_margin={self.safety_margin*100}%")

    def calculate_flight_time_to_node(self, target_node):
        dist_km = distance_calculator.get_straight_line_distance(self.current_node, target_node)
        return dist_km / self.speed

    def calculate_enhanced_energy_consumption(self, flight_time_hours, payload_kg=0.0):
        base_energy = flight_time_hours * self.base_power
        load_energy = flight_time_hours * self.load_power_factor * payload_kg
        total_energy = base_energy + load_energy
        
        if self.energy_debug:
            print(f"UAV {self.vehicle_id} energy calculation: {flight_time_hours:.3f}h × ({self.base_power}kW + {self.load_power_factor}×{payload_kg}kg) = {total_energy:.3f}kWh")
        
        return total_energy

    def can_complete_three_phase_mission(self, order):
        if not hasattr(order, 'start_node') or not hasattr(order, 'end_node'):
            return False, "Order missing node information"
        
        pickup_time = self.calculate_flight_time_to_node(order.start_node)
        pickup_energy = self.calculate_enhanced_energy_consumption(pickup_time, 0.0)
        
        delivery_time = distance_calculator.get_straight_line_distance(order.start_node, order.end_node) / self.speed
        payload = getattr(order, 'weight', self.standard_payload)
        delivery_energy = self.calculate_enhanced_energy_consumption(delivery_time, payload)
        
        return_time = distance_calculator.get_straight_line_distance(order.end_node, self.charging_station_node) / self.speed
        return_energy = self.calculate_enhanced_energy_consumption(return_time, 0.0)
        
        total_energy_required = pickup_energy + delivery_energy + return_energy
        total_time = pickup_time + delivery_time + return_time
        
        safe_energy_required = total_energy_required * 1.2
        
        if self.energy_debug:
            print(f"UAV {self.vehicle_id} three-phase energy analysis order{order.order_id}:")
            print(f"Phase 1 empty pickup: {pickup_time:.3f}h → {pickup_energy:.3f}kWh")
            print(f"Phase 2 loaded delivery: {delivery_time:.3f}h × {payload}kg → {delivery_energy:.3f}kWh")
            print(f"Phase 3 empty return: {return_time:.3f}h → {return_energy:.3f}kWh")
            print(f"Total: {total_time:.3f}h → {total_energy_required:.3f}kWh (with safety: {safe_energy_required:.3f}kWh)")
            print(f"Current battery: {self.battery_level:.3f}kWh, feasible: {self.battery_level >= safe_energy_required}")
        
        feasible = self.battery_level >= safe_energy_required
        reason = "Feasible" if feasible else f"Insufficient battery ({self.battery_level:.1f} < {safe_energy_required:.1f} kWh)"
        
        return feasible, reason

    def execute_flight_phase(self, target_node, payload_kg=0.0, phase_name="Flight"):
        if target_node == self.current_node:
            return True, "Already at target"
        
        flight_time = self.calculate_flight_time_to_node(target_node)
        energy_cost = self.calculate_enhanced_energy_consumption(flight_time, payload_kg)
        
        if self.battery_level < energy_cost:
            return False, f"Insufficient battery ({self.battery_level:.1f} < {energy_cost:.1f} kWh)"
        
        old_node = self.current_node
        self.current_node = target_node
        self.current_grid = target_node
        
        self.battery_level -= energy_cost
        self.total_energy_consumed += energy_cost
        self.total_flight_time += flight_time
        
        if self.energy_debug:
            print(f"UAV {self.vehicle_id} {phase_name}: {old_node}→{target_node}, payload={payload_kg}kg")
            print(f"Consumed: {energy_cost:.3f}kWh, remaining: {self.battery_level:.3f}kWh ({self.battery_level/self.battery_capacity*100:.1f}%)")
        
        return True, "Flight successful"

    def find_nearest_charging_station(self):
        if not self.charging_stations_list:
            return self.charging_station_node
        
        current_pos = self.current_node
        min_distance = float('inf')
        nearest_station = self.charging_stations_list[0]
        
        for station in self.charging_stations_list:
            distance = distance_calculator.get_straight_line_distance(current_pos, station)
            if distance < min_distance:
                min_distance = distance
                nearest_station = station
        
        return nearest_station

    def update_charging_station(self):
        nearest = self.find_nearest_charging_station()
        if nearest != self.charging_station_node:
            old_station = self.charging_station_node
            self.charging_station_node = nearest
            print(f"UAV {self.vehicle_id} switched charging station: {old_station} → {nearest}")
        return self.charging_station_node

    def needs_return_to_charge(self, next_order=None):
        battery_ratio = self.battery_level / self.battery_capacity
        if battery_ratio <= self.safety_margin:
            return True, f"Battery below safety threshold ({battery_ratio*100:.1f}% <= {self.safety_margin*100}%)"
        
        if next_order:
            can_complete, reason = self.can_complete_three_phase_mission(next_order)
            if not can_complete:
                return True, f"Cannot complete next order: {reason}"
        
        nearest_station = self.find_nearest_charging_station()
        return_time = self.calculate_flight_time_to_node(nearest_station)
        return_energy = self.calculate_enhanced_energy_consumption(return_time, 0.0) * 1.1
        
        if self.battery_level < return_energy:
            return True, f"Cannot safely return to charging station ({self.battery_level:.1f} < {return_energy:.1f} kWh)"
        
        return False, "Sufficient battery"

    def return_to_charging_station(self):
        self.update_charging_station()

        if self.current_node == self.charging_station_node:
            old_level = self.battery_level
            self.battery_level = self.battery_capacity
            self.vehicle_state = VehicleState.IDLE
            print(f"UAV {self.vehicle_id} charged at station: {old_level:.1f}→{self.battery_level:.1f}kWh")
            return True
        
        success, message = self.execute_flight_phase(self.charging_station_node, 0.0, "Return to charging")
        
        if success:
            old_level = self.battery_level
            self.battery_level = self.battery_capacity
            self.vehicle_state = VehicleState.IDLE
            print(f"UAV {self.vehicle_id} returned and charged: {old_level:.1f}→{self.battery_level:.1f}kWh")
            return True
        else:
            print(f"UAV {self.vehicle_id} failed to return to charging: {message}")
            return False

    def can_deliver_order_safely(self, order):
        can_complete, reason = self.can_complete_three_phase_mission(order)
        if self.energy_debug and not can_complete:
            print(f"UAV {self.vehicle_id} cannot safely deliver order {order.order_id}: {reason}")
        return can_complete
    
    def fly_to_node_with_energy_management(self, target_node, payload_kg=0.0):
        success, reason = self.execute_flight_phase(target_node, payload_kg, "Energy-managed flight")
        return success

    def get_energy_statistics(self):
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
    
    def __init__(self, carrier_id, start_node, road_graph, capacity=float('inf'), 
                 speed=45, range_limit=50, max_work_hours=8):
        self.vehicle_id = carrier_id
        self.vehicle_type = 'carrier'
        self.start_node = start_node
        self.current_node = start_node
        self.road_graph = road_graph
        
        self.capacity = capacity
        self.speed = speed
        self.range_limit = range_limit
        
        self.max_work_hours = max_work_hours
        self.worked_hours = 0.0
        
        self.current_load = 0
        self.start_grid = start_node
        self.current_grid = start_node
        
        print(f"Carrier {carrier_id} initialized: node={start_node}, work_time_limit={max_work_hours}h")
        
    def calculate_road_travel_time(self, from_node, to_node):
        if from_node == to_node:
            return 0.0
        try:
            distance_km = distance_calculator.get_road_network_distance(from_node, to_node)
            travel_time = distance_km / self.speed
            return travel_time
        except Exception as e:
            print(f"Carrier{self.vehicle_id}: path calculation error {e}")
            return float("inf")
    
    def can_deliver_order_within_work_hours(self, road_order):
        try:
            pickup_time = self.calculate_road_travel_time(self.current_node, road_order.start_node)
            delivery_time = self.calculate_road_travel_time(road_order.start_node, road_order.end_node)
            
            total_work_time = self.worked_hours + pickup_time + delivery_time
            
            if total_work_time <= self.max_work_hours:
                remaining_time = self.max_work_hours - total_work_time
                print(f"Carrier{self.vehicle_id} work time check passed, remaining: {remaining_time:.2f}h")
                return True
            else:
                overtime = total_work_time - self.max_work_hours
                print(f"Carrier{self.vehicle_id} insufficient work time, overtime: {overtime:.2f}h")
                return False
                
        except Exception as e:
            print(f"Carrier{self.vehicle_id} work time check error: {e}")
            return False
    
    def update_work_time(self, additional_hours):
        self.worked_hours += additional_hours
        print(f"Carrier{self.vehicle_id} work time updated: +{additional_hours:.2f}h → total{self.worked_hours:.2f}h/{self.max_work_hours}h")
    
    def is_available_for_work(self):
        return self.worked_hours < self.max_work_hours
    
    def get_remaining_work_time(self):
        return max(0, self.max_work_hours - self.worked_hours)
    
    def reset_work_time(self):
        old_hours = self.worked_hours
        self.worked_hours = 0.0
        print(f"Carrier{self.vehicle_id} work time reset: {old_hours:.2f}h → 0.0h")

class RoadNetworkOrder:
    
    def __init__(self, order_id, start_node, end_node, start_time, deadline, 
                 weight=1.0, priority=1, created_time=0):
        self.order_id = order_id
        self.start_node = start_node
        self.end_node = end_node
        self.start_time = start_time
        self.deadline = deadline
        self.weight = weight
        self.priority = priority
        self.created_time = created_time
        
        self.start_grid = start_node
        self.end_grid = end_node

def test_enhanced_uav_energy():
    print("Testing enhanced UAV energy management...")
    
    test_graph = nx.Graph()
    test_graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    
    uav = RoadNetworkUAV(1, 1, test_graph, battery_capacity=50, charging_station_node=1)
    
    test_order = RoadNetworkOrder(1, 2, 4, 0, 100, weight=0.5)
    
    print(f"Initial state:")
    stats = uav.get_energy_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"Test order feasibility:")
    can_complete, reason = uav.can_complete_three_phase_mission(test_order)
    print(f"Result: {can_complete} - {reason}")
    
    if can_complete:
        print(f"Execute simulated delivery:")
        success1, msg1 = uav.execute_flight_phase(test_order.start_node, 0.0, "Empty pickup")
        print(f"Pickup phase: {success1} - {msg1}")
        
        success2, msg2 = uav.execute_flight_phase(test_order.end_node, test_order.weight, "Loaded delivery")
        print(f"Delivery phase: {success2} - {msg2}")
        
        uav.completed_orders_count += 1
        
        success3 = uav.return_to_charging_station()
        print(f"Return to charge: {success3}")
        
        print(f"Final statistics:")
        final_stats = uav.get_energy_statistics()
        for key, value in final_stats.items():
            print(f"   {key}: {value}")
    
    print("Enhanced UAV energy management test completed")

if __name__ == "__main__":
    test_enhanced_uav_energy()

    



