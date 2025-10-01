import numpy as np
import networkx as nx
import torch
import pickle
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from road_network_vehicles import RoadNetworkUAV, RoadNetworkCarrier, RoadNetworkOrder

from enum import Enum

class VehicleState(Enum):
    IDLE = "idle"
    ASSIGNED = "assigned"
    DELIVERING = "delivering"

class OrderStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    PICKED_UP = "picked_up"
    COMPLETED = "completed"
    EXPIRED = "expired"

@dataclass
class RealRoadOrder:
    order_id: int
    start_node: int
    end_node: int
    start_time: int
    deadline: int
    weight: float
    priority: int
    created_time: int
    original_road_order_id: int

class PureRealRoadNetworkEnvironmentWithConstraints:
    
    def _get_global_order_distribution(self):
        if not self.active_orders:
            return np.zeros(8, dtype=np.float32)
        
        total_orders = len(self.active_orders)
        urgent_orders = sum(1 for order in self.active_orders.values() if self._is_order_urgent(order))
        
        order_nodes = [order.start_node for order in self.active_orders.values()]
        avg_order_node = np.mean(order_nodes) if order_nodes else 0
        
        matched_orders = sum(1 for vehicle in self.vehicles if vehicle.vehicle_id in self.vehicle_assigned_orders)
        
        global_info = np.array([
            total_orders / 50.0,
            urgent_orders / max(total_orders, 1),
            matched_orders / self.num_vehicles,
            self.time_step / self.max_time_steps,
            len(self.completed_orders) / max(self.total_orders_generated, 1),
            self.total_orders_matched / max(self.total_orders_generated, 1),
            avg_order_node / 600000.0,
            min(len(self.active_orders) / 20.0, 1.0)
        ], dtype=np.float32)
        
        return global_info

    def __init__(self, num_ground_vehicles=6, num_uavs=3, max_time_steps=120, max_concurrent_orders=15):
        self.num_ground_vehicles = num_ground_vehicles
        self.num_uavs = num_uavs
        self.num_vehicles = num_ground_vehicles + num_uavs
        self.max_concurrent_orders = max_concurrent_orders
        
        self.road_graph = None
        self.node_features = None
        self.node_embeddings = {}
        self.largest_component_nodes = None

        self.vehicles = []

        self.vehicle_states = {}
        self.vehicle_assigned_orders = {}
        self.order_statuses = {}
        self.active_orders = {}
        self.completed_orders = []
        
        self.time_step = 0
        self.max_time_steps = max_time_steps
        self.orders_data = None
        
        self.total_orders_generated = 0
        self.total_orders_completed = 0
        self.total_orders_matched = 0
        self.path_planning_failures = 0
        self.uav_energy_failures = 0
        self.carrier_time_failures = 0
        self.charging_events = 0
        self.episode_count = 0

        self.episode_pickups = 0
        self.episode_deliveries = 0

        self.vehicle_states = {}
        self.vehicle_assigned_orders = {}
        self.order_statuses = {}
        
        self.state_dim = 52
        
        self._load_real_road_network()
        self._initialize_vehicles_with_constraints()
        
    def _load_real_road_network(self):
        print("Loading Shanghai road network...")
        
        try:
            with open('./shanghai_road_dataset/road_network/road_graph_fixed.pkl', 'rb') as f:
                self.road_graph = pickle.load(f)
            print(f"Road graph loaded: {len(self.road_graph.nodes)} nodes, {len(self.road_graph.edges)} edges")
            
            self.node_features = np.load('./shanghai_road_dataset/road_network/node_features_fixed.npy')
            print(f"Node features loaded: {self.node_features.shape}")
            
            self.node_embeddings = {}
            for node_id in self.road_graph.nodes:
                if node_id < len(self.node_features):
                    raw_features = self.node_features[node_id][:8]
                    normalized_features = raw_features.copy()
                    normalized_features[0] = (raw_features[0] - 121.0) / 1.0
                    normalized_features[1] = (raw_features[1] - 31.0) / 1.0
                    normalized_features[2:] = raw_features[2:] / (np.abs(raw_features[2:]) + 1e-6)
                    
                    self.node_embeddings[node_id] = normalized_features.astype(np.float32)
                else:
                    self.node_embeddings[node_id] = np.zeros(8, dtype=np.float32)
            
            print(f"Node embeddings computed: {len(self.node_embeddings)} nodes")
            
            connected_components = list(nx.connected_components(self.road_graph))
            self.largest_component_nodes = list(max(connected_components, key=len))
            print(f"Largest connected component: {len(self.largest_component_nodes)} nodes")
            
        except Exception as e:
            print(f"Road network loading failed: {e}")
            raise RuntimeError("Real road network data required!")
    
    def _initialize_vehicles_with_constraints(self):
        self.vehicles = []
        
        if self.largest_component_nodes is None or len(self.largest_component_nodes) < self.num_vehicles:
            if self.largest_component_nodes is None:
                available_nodes = list(self.road_graph.nodes())[:self.num_vehicles] if self.road_graph else [0] * self.num_vehicles
            else:
                available_nodes = self.largest_component_nodes
        else:
            available_nodes = np.random.choice(self.largest_component_nodes, self.num_vehicles, replace=False)
        
        num_charging_stations = 15
        if self.largest_component_nodes is not None and len(self.largest_component_nodes) >= num_charging_stations:
            charging_stations = np.random.choice(self.largest_component_nodes, num_charging_stations, replace=False)
        else:
            charging_stations = self.largest_component_nodes[:num_charging_stations] if self.largest_component_nodes is not None else [0] * num_charging_stations
        
        self.charging_stations = charging_stations
        print(f"Charging stations: {charging_stations}")
        
        for i in range(self.num_ground_vehicles):
            start_node = available_nodes[i % len(available_nodes)]
            carrier = RoadNetworkCarrier(
                carrier_id=i, 
                start_node=start_node, 
                road_graph=self.road_graph,
                capacity=float('inf'),
                speed=30,
                range_limit=50,
                max_work_hours=8
            )
            self.vehicles.append(carrier)
            self.vehicle_states[carrier.vehicle_id] = VehicleState.IDLE
        
        for i in range(self.num_uavs):
            vehicle_id = self.num_ground_vehicles + i
            start_node = available_nodes[vehicle_id % len(available_nodes)]
            charging_station = charging_stations[i % len(charging_stations)]
            
            uav = RoadNetworkUAV(
                uav_id=vehicle_id,
                start_node=start_node,
                road_graph=self.road_graph,
                capacity=5,
                battery_capacity=1,
                charging_station_node=charging_station,
                charging_stations_list=charging_stations,
                speed=50
            )
            self.vehicles.append(uav)
            self.vehicle_states[uav.vehicle_id] = VehicleState.IDLE
        
        print(f"Initialized {len(self.vehicles)} constrained vehicles:")
        print(f"  Ground carriers: {self.num_ground_vehicles} (work time limit: 8h)")
        print(f"  UAVs: {self.num_uavs} (battery limit: 50kWh)")
        print(f"  Vehicle locations: {[v.current_node for v in self.vehicles]}")
    
    def load_road_orders(self, day: int) -> bool:
        try:
            order_file = f'./shanghai_road_dataset/processed_orders_road/Orders_Dataset_shanghai_road_day_{day}'
            with open(order_file, 'rb') as f:
                self.orders_data = pickle.load(f)
            
            print(f"Road orders loaded: day {day}, {len(self.orders_data)} time steps")
            return True
            
        except Exception as e:
            print(f"Road orders loading failed: {e}")
            return False
    
    def generate_orders(self) -> List[RealRoadOrder]:
        new_orders = []
        
        if not hasattr(self, 'orders_data') or not self.orders_data or self.time_step not in self.orders_data:
            return new_orders
        
        if self.road_graph is None:
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
                            deadline=int(end_time) + 50,
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
            print(f"Time step {self.time_step}: Generated {len(new_orders)} road orders")
        
        return new_orders
    
    def assign_order_to_vehicle(self, vehicle_id, order):
        if self.vehicle_states.get(vehicle_id) != VehicleState.IDLE:
            return False
        
        self.vehicle_states[vehicle_id] = VehicleState.ASSIGNED
        self.vehicle_assigned_orders[vehicle_id] = order
        self.order_statuses[order.order_id] = OrderStatus.ASSIGNED
        
        order.assigned_vehicle = vehicle_id
        
        print(f"Vehicle {vehicle_id} accepted order {order.order_id}")
        return True
    
    def vehicle_start_pickup(self, vehicle_id):
        if (self.vehicle_states.get(vehicle_id) == VehicleState.ASSIGNED and 
            vehicle_id in self.vehicle_assigned_orders):
            order = self.vehicle_assigned_orders[vehicle_id]
            self.order_statuses[order.order_id] = OrderStatus.PICKED_UP
            
            vehicle = next((v for v in self.vehicles if v.vehicle_id == vehicle_id), None)
            if vehicle and hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'uav':
                print(f"UAV {vehicle_id} started pickup for order {order.order_id}")
            else:
                print(f"Carrier {vehicle_id} started pickup for order {order.order_id}")

            self.episode_pickups += 1
            return True
        return False
    
    def vehicle_start_delivery(self, vehicle_id):
        if (self.vehicle_states.get(vehicle_id) == VehicleState.ASSIGNED and 
            vehicle_id in self.vehicle_assigned_orders):
            self.vehicle_states[vehicle_id] = VehicleState.DELIVERING
            order = self.vehicle_assigned_orders[vehicle_id]
            
            vehicle = next((v for v in self.vehicles if v.vehicle_id == vehicle_id), None)
            if vehicle and hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'uav':
                print(f"UAV {vehicle_id} started delivery for order {order.order_id}")
            else:
                print(f"Carrier {vehicle_id} started delivery for order {order.order_id}")
            return True
        return False
    
    def vehicle_complete_delivery(self, vehicle_id):
        if (self.vehicle_states.get(vehicle_id) == VehicleState.DELIVERING and 
            vehicle_id in self.vehicle_assigned_orders):
            order = self.vehicle_assigned_orders[vehicle_id]
            
            self.order_statuses[order.order_id] = OrderStatus.COMPLETED
            self.vehicle_states[vehicle_id] = VehicleState.IDLE
            
            if order.order_id in self.active_orders:
                completed_order = self.active_orders.pop(order.order_id)
                self.completed_orders.append(completed_order)
                self.total_orders_completed += 1

                self.episode_deliveries += 1
            
            del self.vehicle_assigned_orders[vehicle_id]
            
            vehicle = next((v for v in self.vehicles if v.vehicle_id == vehicle_id), None)
            if vehicle and hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'uav':
                print(f"UAV {vehicle_id} completed delivery for order {order.order_id}")
            else:
                print(f"Carrier {vehicle_id} completed delivery for order {order.order_id}")
            
            return True
        return False
    
    def _get_distance(self, node1, node2):
        if node1 is None or node2 is None:
            return float('inf')
        
        try:
            if self.road_graph is not None:
                return nx.shortest_path_length(self.road_graph, node1, node2, weight='weight')
            else:
                return abs(node1 - node2) * 0.0003
        except:
            return float('inf')

    def _can_complete_order_within_ddl(self, vehicle, order):
        try:
            if vehicle.vehicle_type == 'uav':
                pickup_time = vehicle.calculate_flight_time_to_node(order.start_node)
                delivery_time = vehicle.calculate_flight_time_to_node(order.end_node)
            else:
                pickup_time = vehicle.calculate_road_travel_time(vehicle.current_node, order.start_node)
                delivery_time = vehicle.calculate_road_travel_time(order.start_node, order.end_node)
            
            total_completion_time = self.time_step + pickup_time + delivery_time
            ddl_margin = order.deadline - total_completion_time
            
            if ddl_margin >= 0:
                return True
            else:
                return False
                
        except Exception as e:
            return False
    
    def _is_order_urgent(self, order):
        time_remaining = order.deadline - self.time_step
        return time_remaining <= 10

    def _perform_global_order_assignment(self, actions: List[np.ndarray]):
        if not self.active_orders:
            return 0
        
        assignment_count = 0
        
        available_orders = []

        for order in self.active_orders.values():
            if order.order_id not in [assigned_order.order_id for assigned_order in self.vehicle_assigned_orders.values()]:
                available_orders.append(order)
        
        available_orders.sort(key=lambda x: x.created_time)
        
        if not available_orders:
            return 0
        
        for vehicle in self.vehicles:
            if self.vehicle_states.get(vehicle.vehicle_id) != VehicleState.IDLE:
                continue
            
            if vehicle.vehicle_id >= len(actions):
                continue
                
            action = actions[vehicle.vehicle_id]
            
            order_scores = action if len(action) > 0 else []
            
            best_order = None
            best_score = -float('inf')
            
            for idx, order in enumerate(available_orders[:self.max_concurrent_orders]):
                if not self._can_vehicle_handle_order(vehicle, order):
                    continue
                
                if idx < len(order_scores):
                    action_score = (order_scores[idx] + 1) / 2
                else:
                    action_score = 0.5
                
                final_score = action_score
                
                if final_score > best_score:
                    best_score = final_score
                    best_order = order
            
            if best_order is not None and best_score > 0.1:
                if self.assign_order_to_vehicle(vehicle.vehicle_id, best_order):
                    assignment_count += 1
                    available_orders.remove(best_order)
                    print(f"   Vehicle{vehicle.vehicle_id} ← Order{best_order.order_id}: score={best_score:.3f}")
        
        return assignment_count
    
    def _can_vehicle_handle_order(self, vehicle, order):
        if not self._can_complete_order_within_ddl(vehicle, order):
            return False
        
        if vehicle.vehicle_type == "uav":
            if hasattr(vehicle, "can_complete_three_phase_mission"):
                can_handle, reason = vehicle.can_complete_three_phase_mission(order)
                if not can_handle:
                    return can_handle
            elif hasattr(vehicle, "can_deliver_order_safely"):
                road_order = RoadNetworkOrder(
                    order.order_id, order.start_node, order.end_node,
                    order.start_time, order.deadline, order.weight, order.priority
                )
                return vehicle.can_deliver_order_safely(road_order)
        else:
            road_order = RoadNetworkOrder(
                order.order_id, order.start_node, order.end_node,
                order.start_time, order.deadline, order.weight, order.priority
            )
            if hasattr(vehicle, "can_deliver_order_within_work_hours"):
                return vehicle.can_deliver_order_within_work_hours(road_order)
        
        return True

    def _update_vehicle_behaviors(self):
        for vehicle in self.vehicles:
            vehicle_id = vehicle.vehicle_id
            current_state = self.vehicle_states.get(vehicle_id, VehicleState.IDLE)
            
            if current_state == VehicleState.ASSIGNED:
                assigned_order = self.vehicle_assigned_orders.get(vehicle_id)
                if assigned_order:
                    self._handle_pickup_movement(vehicle, assigned_order)
                    
            elif current_state == VehicleState.DELIVERING:
                assigned_order = self.vehicle_assigned_orders.get(vehicle_id)
                if assigned_order:
                    self._handle_delivery_movement(vehicle, assigned_order)

    def _check_ddl_timeouts(self):
        expired_vehicles = []
        
        for vehicle_id, order in list(self.vehicle_assigned_orders.items()):
            if order.deadline <= self.time_step:
                vehicle_state = self.vehicle_states.get(vehicle_id)
                
                if vehicle_state == VehicleState.DELIVERING:
                    print(f"Order{order.order_id} completed at DDL")
                    self.vehicle_complete_delivery(vehicle_id)
                else:
                    print(f"Order{order.order_id} DDL timeout, delivery failed")
                    self._handle_order_failure(vehicle_id, order)
                    
                expired_vehicles.append(vehicle_id)
        
        return len(expired_vehicles)
    
    def _handle_order_failure(self, vehicle_id, order):
        self.order_statuses[order.order_id] = OrderStatus.EXPIRED
        self.vehicle_states[vehicle_id] = VehicleState.IDLE
        
        if vehicle_id in self.vehicle_assigned_orders:
            del self.vehicle_assigned_orders[vehicle_id]
        
        if order.order_id in self.active_orders:
            expired_order = self.active_orders.pop(order.order_id)
            if not hasattr(self, "failed_orders"):
                self.failed_orders = []
            self.failed_orders.append(expired_order)
    
    def _handle_pickup_movement(self, vehicle, order):
        target_node = order.start_node
        
        if self._is_at_pickup_location(vehicle, order):
            if self.vehicle_start_pickup(vehicle.vehicle_id):
                self.vehicle_start_delivery(vehicle.vehicle_id)
            return
        
        if hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'uav':
            self._handle_uav_direct_flight(vehicle, target_node, "pickup")
        else:
            self._handle_carrier_road_movement(vehicle, target_node, "pickup")
    
    def _handle_delivery_movement(self, vehicle, order):
        target_node = order.end_node
        
        if self._is_at_delivery_location(vehicle, order):
            self.vehicle_complete_delivery(vehicle.vehicle_id)
            return
        
        if hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'uav':
            self._handle_uav_direct_flight(vehicle, target_node, "delivery")
        else:
            self._handle_carrier_road_movement(vehicle, target_node, "delivery")
    
    def _is_at_pickup_location(self, vehicle, order):
        return vehicle.current_node == order.start_node
    
    def _is_at_delivery_location(self, vehicle, order):
        return vehicle.current_node == order.end_node
    
    def _get_next_move_toward_target(self, vehicle, target_node):
        if vehicle.current_node == target_node:
            return target_node
        
        try:
            if self.road_graph is not None:
                path = nx.shortest_path(self.road_graph, vehicle.current_node, target_node)
                if len(path) > 1:
                    return path[1]
            
            neighbors = list(self.road_graph.neighbors(vehicle.current_node)) if self.road_graph is not None else []
            if neighbors:
                best_neighbor = min(neighbors, 
                                  key=lambda n: self._get_distance(n, target_node))
                return best_neighbor
                
        except Exception as e:
            print(f"Vehicle{vehicle.vehicle_id} path planning failed: {e}")
            
        return vehicle.current_node
    
    def _get_nearby_orders(self, node: int, max_orders: int = 3) -> List[RealRoadOrder]:
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
    
    def get_state(self) -> List[np.ndarray]:
        states = []
        
        global_order_info = self._get_global_order_distribution()
        
        for vehicle in self.vehicles:
            state = []
            
            current_embedding = self.node_embeddings.get(vehicle.current_node, np.zeros(8, dtype=np.float32))
            state.extend(current_embedding)
            
            if hasattr(vehicle, 'capacity') and vehicle.capacity != float('inf'):
                load_ratio = vehicle.current_load / vehicle.capacity
            else:
                load_ratio = len(getattr(vehicle, 'assigned_orders', [])) / 10.0
            
            state.extend([
                load_ratio,
                len(getattr(vehicle, 'assigned_orders', [])) / 10.0,
                1.0 if vehicle.vehicle_type == 'uav' else 0.0,
                getattr(vehicle, 'speed', 30) / 60.0
            ])
            
            state.extend(global_order_info)
            
            if self.road_graph is not None and vehicle.current_node in self.road_graph:
                neighbors = list(self.road_graph.neighbors(vehicle.current_node))[:4]
            else:
                neighbors = []
            for i in range(4):
                if i < len(neighbors):
                    neighbor = neighbors[i]
                    distance = 1.0
                    has_order = any(order.start_node == neighbor or order.end_node == neighbor 
                                  for order in self.active_orders.values())
                    state.extend([distance / 10.0, 1.0 if has_order else 0.0])
                else:
                    state.extend([0.0, 0.0])
            
            other_vehicles_info = []
            for other_vehicle in self.vehicles:
                if other_vehicle.vehicle_id != vehicle.vehicle_id:
                    other_has_order = 1.0 if other_vehicle.vehicle_id in self.vehicle_assigned_orders else 0.0
                    other_type = 1.0 if other_vehicle.vehicle_type == 'uav' else 0.0
                    other_vehicles_info.extend([other_has_order, other_type])
            
            while len(other_vehicles_info) < 12:
                other_vehicles_info.append(0.0)
            state.extend(other_vehicles_info[:12])
            
            nearby_orders = self._get_nearby_orders(vehicle.current_node, max_orders=2)
            for i in range(2):
                if i < len(nearby_orders):
                    order = nearby_orders[i]
                    try:
                        if self.road_graph and vehicle.current_node in self.road_graph and order.start_node in self.road_graph:
                            distance = nx.shortest_path_length(self.road_graph, vehicle.current_node, order.start_node)
                        else:
                            distance = 10
                        urgency = max(0, (order.deadline - self.time_step)) / 20.0 if hasattr(order, 'deadline') else 0.5
                        priority = getattr(order, 'priority', 1) / 3.0
                        weight = getattr(order, 'weight', 1.0) / 5.0
                        state.extend([distance / 20.0, urgency, priority, weight])
                    except:
                        state.extend([0.5, 0.5, 0.5, 0.5])
                else:
                    state.extend([0.0, 0.0, 0.0, 0.0])
            
            constraint_info = []
            if vehicle.vehicle_type == 'uav':
                battery_ratio = getattr(vehicle, 'battery_level', 100) / getattr(vehicle, 'battery_capacity', 100)
                charging_distance = 0.5
                constraint_info.extend([battery_ratio, charging_distance, 1.0, 0.0])
            else:
                work_ratio = getattr(vehicle, 'worked_hours', 0) / getattr(vehicle, 'max_work_hours', 8)
                rest_time = max(0, 8 - getattr(vehicle, 'worked_hours', 0)) / 8.0
                constraint_info.extend([work_ratio, rest_time, 0.0, 1.0])
            
            state.extend(constraint_info)
            
            state_array = np.array(state, dtype=np.float32)
            if len(state_array) != 52:
                if len(state_array) < 52:
                    state_array = np.pad(state_array, (0, 52 - len(state_array)), 'constant')
                else:
                    state_array = state_array[:52]
            
            states.append(state_array)
        
        return states
    
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        self._perform_global_order_assignment(actions)

        pickup_count, delivery_count = self._execute_constrained_vehicle_actions(actions)
        
        new_orders = self.generate_orders()
        
        self._update_vehicle_behaviors()
        
        timeout_count = self._check_ddl_timeouts()
        
        expired_count = self._update_order_status()
        
        self._handle_constraint_maintenance()
        
        rewards = self._calculate_rewards_with_constraints(pickup_count, delivery_count)
        
        done = self.time_step >= self.max_time_steps - 1
        
        self.time_step += 1
        
        next_states = self.get_state()
        
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
            'delivery_count': delivery_count,
            'expired_count': expired_count,
            'timeout_count': timeout_count,
            'path_planning_failures': self.path_planning_failures,
            'uav_energy_failures': self.uav_energy_failures,
            'carrier_time_failures': self.carrier_time_failures,
            'charging_events': self.charging_events,
        }
        
        return next_states, rewards, done, info
    
    def _execute_constrained_vehicle_actions(self, actions: List[np.ndarray]) -> Tuple[int, int]:
        pickup_count = 0
        delivery_count = 0
        
        for vehicle, action in zip(self.vehicles, actions):
            if self.time_step < 2:
                print(f"   Vehicle{vehicle.vehicle_id}: order scoring action={action[:3]}")
            
            vehicle_state = self.vehicle_states.get(vehicle.vehicle_id, VehicleState.IDLE)
            
            if vehicle.vehicle_type == "uav":
                if hasattr(vehicle, 'battery_level') and hasattr(vehicle, 'battery_capacity'):
                    battery_ratio = vehicle.battery_level / vehicle.battery_capacity
                    if battery_ratio < 0.1:
                        if self.time_step % 10 == 0:
                            print(f"UAV{vehicle.vehicle_id} low battery: {battery_ratio:.1%}")
            else:
                if hasattr(vehicle, 'worked_hours') and hasattr(vehicle, 'max_work_hours'):
                    work_ratio = vehicle.worked_hours / vehicle.max_work_hours
                    if work_ratio > 0.9:
                        if self.time_step % 10 == 0:
                            print(f"Carrier{vehicle.vehicle_id} work time almost full: {work_ratio:.1%}")
        
        return self.episode_pickups, self.episode_deliveries
    
    def _handle_constraint_maintenance(self):
        for vehicle in self.vehicles:
            if vehicle.vehicle_type == 'uav':
                if hasattr(vehicle, 'needs_return_to_charge'):
                    result = vehicle.needs_return_to_charge()
                    if isinstance(result, tuple):
                        needs_charge, reason = result
                    else:
                        needs_charge = result
                        reason = "Legacy check"
                    if needs_charge:
                        print(f"UAV {vehicle.vehicle_id} needs to return to charging station")
                        if vehicle.vehicle_id in self.vehicle_assigned_orders:
                            order = self.vehicle_assigned_orders[vehicle.vehicle_id]
                            print(f"UAV {vehicle.vehicle_id} cancelled order{order.order_id} due to low battery")
                            self._handle_order_failure(vehicle.vehicle_id, order)
                            self.uav_energy_failures += 1
                        
                        if hasattr(vehicle, 'return_to_charging_station'):
                            success = vehicle.return_to_charging_station()
                            if success:
                                self.charging_events += 1
            else:
                if hasattr(vehicle, 'worked_hours') and hasattr(vehicle, 'max_work_hours'):
                    if vehicle.worked_hours >= vehicle.max_work_hours:
                        print(f"Carrier {vehicle.vehicle_id} reached maximum work hours")
                        if vehicle.vehicle_id in self.vehicle_assigned_orders:
                            order = self.vehicle_assigned_orders[vehicle.vehicle_id]
                            print(f"Carrier {vehicle.vehicle_id} cancelled order{order.order_id} due to work time limit")
                            self._handle_order_failure(vehicle.vehicle_id, order)
                            self.carrier_time_failures += 1
                        
                        self.vehicle_states[vehicle.vehicle_id] = VehicleState.IDLE
    
    def _calculate_rewards_with_constraints(self, pickup_count: int, delivery_count: int) -> List[float]:
        rewards = []
        
        global_pickup_reward = pickup_count * 2.0
        global_delivery_reward = delivery_count * 8.0
        
        system_efficiency = (pickup_count + delivery_count) / max(len(self.active_orders), 1)
        efficiency_bonus = system_efficiency * 1.0
        
        active_vehicles = sum(1 for v in self.vehicles if v.vehicle_id in self.vehicle_assigned_orders)
        cooperation_bonus = (active_vehicles / self.num_vehicles) * 0.5
        
        for vehicle in self.vehicles:
            reward = 0.0
            
            reward += 0.5
            
            individual_bonus = 0.0
            if vehicle.vehicle_id in self.vehicle_assigned_orders:
                individual_bonus += 1.4
                
            if hasattr(vehicle, 'completed_deliveries_this_step'):
                individual_bonus += vehicle.completed_deliveries_this_step * 5.6
            
            reward += individual_bonus
            
            collective_reward = (global_pickup_reward + global_delivery_reward) * 0.3 / self.num_vehicles
            reward += collective_reward
            
            reward += efficiency_bonus + cooperation_bonus
            
            if hasattr(self, 'episode_count'):
                progress_bonus = min(0.1 * self.episode_count / 50.0, 0.2)
                reward += progress_bonus
            
            if vehicle.vehicle_id in self.vehicle_assigned_orders:
                order = self.vehicle_assigned_orders[vehicle.vehicle_id]
                try:
                    if hasattr(order, 'start_node') and self.road_graph:
                        if vehicle.current_node in self.road_graph and order.start_node in self.road_graph:
                            current_distance = nx.shortest_path_length(
                                self.road_graph, vehicle.current_node, order.start_node
                            )
                            distance_reward = max(0, 1.0 - current_distance / 20.0) * 0.2
                            reward += distance_reward
                except:
                    pass
            
            if vehicle.vehicle_type == 'uav':
                if hasattr(vehicle, 'battery_level') and hasattr(vehicle, 'battery_capacity'):
                    battery_ratio = vehicle.battery_level / vehicle.battery_capacity
                    if battery_ratio > 0.8:
                        reward += 0.3
                    elif battery_ratio < 0.2:
                        reward -= 0.8
            else:
                if hasattr(vehicle, 'worked_hours') and hasattr(vehicle, 'max_work_hours'):
                    work_ratio = vehicle.worked_hours / vehicle.max_work_hours
                    if work_ratio < 0.7:
                        reward += 0.2
                    elif work_ratio > 0.9:
                        reward -= 0.5
            
            rewards.append(reward)
        
        return rewards
    
    def _update_order_status(self) -> int:
        expired_orders = []
        for order in self.active_orders.values():
            if self.time_step > order.deadline:
                expired_orders.append(order.order_id)
        
        for order_id in expired_orders:
            del self.active_orders[order_id]
        
        return len(expired_orders)
    
    def _handle_uav_direct_flight(self, vehicle, target_node, target_type):
        current_node = vehicle.current_node
        
        if current_node == target_node:
            return
        
        current_payload = 0.5 if (hasattr(vehicle, 'vehicle_state') and 
                                 self.vehicle_states.get(vehicle.vehicle_id) == VehicleState.DELIVERING) else 0.0
        
        if hasattr(vehicle, 'calculate_flight_time_to_node') and hasattr(vehicle, 'fly_to_node_with_energy_management'):
            flight_success = vehicle.fly_to_node_with_energy_management(target_node, current_payload)
            
            if flight_success:
                print(f"UAV{vehicle.vehicle_id} reached {target_type}: {target_node}")
                battery_ratio = vehicle.battery_level / vehicle.battery_capacity
                print(f"   Battery remaining: {vehicle.battery_level:.1f}kWh ({battery_ratio*100:.1f}%)")
                
                need_charge, reason = vehicle.needs_return_to_charge()
                if need_charge:
                    print(f"UAV{vehicle.vehicle_id} needs charging: {reason}")
                    charge_success = vehicle.return_to_charging_station()
                    if charge_success:
                        self.charging_events += 1
            else:
                print(f"UAV{vehicle.vehicle_id} flight failed: insufficient battery")
                self._force_uav_return_to_charge(vehicle)
        else:
            self._handle_uav_legacy_flight(vehicle, target_node, target_type)
    
    def _handle_uav_legacy_flight(self, vehicle, target_node, target_type):
        current_node = vehicle.current_node
        
        if current_node == target_node:
            return
        
        distance = self._calculate_euclidean_distance(current_node, target_node)
        
        uav_speed = 30.0
        
        flight_time = distance / uav_speed
        
        time_steps_needed = max(1, int(flight_time * 12))
        
        if not hasattr(vehicle, 'flight_remaining_steps'):
            vehicle.flight_remaining_steps = 0
            vehicle.flight_target = None
        
        if vehicle.flight_target != target_node:
            vehicle.flight_remaining_steps = time_steps_needed
            vehicle.flight_target = target_node
            print(f"UAV{vehicle.vehicle_id} started flight to {target_type}: {current_node} → {target_node} ({time_steps_needed} steps)")
        
        vehicle.flight_remaining_steps -= 1
        
        if vehicle.flight_remaining_steps <= 0:
            vehicle.current_node = target_node
            vehicle.flight_target = None
            print(f"UAV{vehicle.vehicle_id} reached {target_type}: {target_node}")
        else:
            print(f"UAV{vehicle.vehicle_id} flying... {vehicle.flight_remaining_steps} steps remaining")
    
    def _force_uav_return_to_charge(self, vehicle):
        print(f"UAV{vehicle.vehicle_id} forced to return to charging station: {vehicle.charging_station_node}")
        
        self.vehicle_states[vehicle.vehicle_id] = VehicleState.IDLE
        
        if vehicle.vehicle_id in self.vehicle_assigned_orders:
            failed_order = self.vehicle_assigned_orders[vehicle.vehicle_id]
            print(f"Order{failed_order.order_id} cancelled due to insufficient battery")
            del self.vehicle_assigned_orders[vehicle.vehicle_id]
            self.uav_energy_failures += 1
        
        vehicle.current_node = vehicle.charging_station_node
        old_level = vehicle.battery_level
        vehicle.battery_level = vehicle.battery_capacity
        print(f"UAV{vehicle.vehicle_id} charging completed: {old_level:.1f}→{vehicle.battery_level}kWh")
    
    def _handle_carrier_road_movement(self, vehicle, target_node, target_type):
        current_node = vehicle.current_node
        
        if current_node == target_node:
            return
        
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
            print(f"Carrier{vehicle.vehicle_id} moving to {target_type}: {path_taken[0]} → {path_taken[-1]} ({len(path_taken)-1} moves)")
    
    def _calculate_euclidean_distance(self, node1, node2):
        if not hasattr(self, 'node_features') or self.node_features is None:
            return abs(float(node1) - float(node2)) / 1000.0
        
        try:
            return abs(float(node1) - float(node2)) / 1000.0
            
        except Exception:
            return abs(float(node1) - float(node2)) / 1000.0
    
    def reset(self) -> List[np.ndarray]:
        self.time_step = 0
        self.active_orders = {}
        self.completed_orders = []
        
        if not hasattr(self, 'orders_data') or not self.orders_data:
            self.load_road_orders(day=1)
        
        self.total_orders_generated = 0
        self.total_orders_completed = 0
        self.total_orders_matched = 0
        self.path_planning_failures = 0
        self.uav_energy_failures = 0
        self.carrier_time_failures = 0
        self.charging_events = 0

        self.episode_pickups = 0
        self.episode_deliveries = 0

        self.vehicle_states = {}
        self.vehicle_assigned_orders = {}
        self.order_statuses = {}
        
        if self.largest_component_nodes is not None and len(self.largest_component_nodes) >= self.num_vehicles:
            start_nodes = np.random.choice(self.largest_component_nodes, self.num_vehicles, replace=False)
        elif self.largest_component_nodes is not None:
            start_nodes = np.random.choice(self.largest_component_nodes, self.num_vehicles, replace=True)
        else:
            available_nodes = list(self.road_graph.nodes())[:self.num_vehicles] if self.road_graph else [0] * self.num_vehicles
            start_nodes = available_nodes
        
        for i, vehicle in enumerate(self.vehicles):
            vehicle.current_node = start_nodes[i]
            
            if vehicle.vehicle_type == 'uav':
                if hasattr(vehicle, 'battery_level') and hasattr(vehicle, 'battery_capacity'):
                    vehicle.battery_level = vehicle.battery_capacity
            else:
                if hasattr(vehicle, 'worked_hours'):
                    vehicle.worked_hours = 0

            self.vehicle_states[vehicle.vehicle_id] = VehicleState.IDLE
            
            if hasattr(vehicle, 'assigned_orders'):
                pass
            
            vehicle.current_grid = vehicle.current_node
        
        print(f"Constrained environment reset, vehicle locations: {[v.current_node for v in self.vehicles]}")
        return self.get_state()



def test_constrained_environment():
    print("Testing constrained road network environment...")
    
    try:
        env = PureRealRoadNetworkEnvironmentWithConstraints(num_ground_vehicles=2, num_uavs=2)
        
        success = env.load_road_orders(1)
        if not success:
            print("Order data loading failed")
            return False
        
        states = env.reset()
        print(f"Environment reset successful, state dimension: {len(states)} x {len(states[0])}")
        
        expected_dim = 52
        if len(states[0]) == expected_dim:
            print(f"State dimension correct: {expected_dim}")
        else:
            print(f"State dimension error: {len(states[0])} != {expected_dim}")
        
        for step in range(10):
            actions = []
            for i in range(len(states)):
                action = np.array([0.8, 0.9, 0.9] + [0.5] * 33)
                actions.append(action)
            
            next_states, rewards, done, info = env.step(actions)
            
            print(f"Step {step:2d}: "
                  f"Reward={sum(rewards):5.1f}, "
                  f"Orders={info.get('new_orders_count', 0)}, "
                  f"Active={info.get('active_orders', 0)}, "
                  f"Energy failures={info.get('uav_energy_failures', 0)}, "
                  f"Time failures={info.get('carrier_time_failures', 0)}, "
                  f"Charging={info.get('charging_events', 0)}")
            
            states = next_states
            
            if done:
                break
        
        print("Constrained environment test successful!")
        return True
        
    except Exception as e:
        print(f"Constrained environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_constrained_environment()