import os
import sys
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'simulator'))

from pure_road_network_environment_with_energy_constraints_centralized import (
    PureRealRoadNetworkEnvironmentWithConstraints, 
    OrderStatus, 
    VehicleState
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GreedyGAPBaseline:
    
    def __init__(self, num_ground_vehicles=5, num_uavs=2):
        self.num_ground_vehicles = num_ground_vehicles  
        self.num_uavs = num_uavs
        self.total_vehicles = num_ground_vehicles + num_uavs
        
        self.distance_weight = 0.4
        self.capacity_weight = 0.3
        self.success_weight = 0.3
        
        logger.info(f"Initialized Greedy GAP algorithm: {num_ground_vehicles} carriers + {num_uavs} UAVs")
        
    def calculate_assignment_profit(self, vehicle, order, env) -> float:
        profit = 0.0
        
        try:
            distance = abs(vehicle.current_node - order.start_node)
            max_distance = 100.0  
            distance_score = max(0, 1.0 - distance / max_distance)
            profit += self.distance_weight * distance_score
        except:
            profit += self.distance_weight * 0.5
        
        if vehicle.vehicle_type == 'uav':
            if hasattr(vehicle, 'battery_level') and hasattr(vehicle, 'battery_capacity'):
                battery_ratio = vehicle.battery_level / vehicle.battery_capacity
                capacity_score = battery_ratio
            else:
                capacity_score = 0.8
        else:
            if hasattr(vehicle, 'worked_hours') and hasattr(vehicle, 'max_work_hours'):
                work_ratio = vehicle.worked_hours / vehicle.max_work_hours
                capacity_score = max(0, 1.0 - work_ratio)
            else:
                capacity_score = 0.8
                
        profit += self.capacity_weight * capacity_score
        
        success_score = 1.0
        if vehicle.vehicle_type == 'uav':
            if hasattr(order, 'weight') and order.weight > 5:
                success_score = 0.1
        
        profit += self.success_weight * success_score
        return profit
    
    def greedy_assignment(self, available_orders, vehicles_status, env) -> Dict[int, int]:
        assignments = {}
        
        if not available_orders:
            return assignments
        
        profit_matrix = []
        
        for vehicle in env.vehicles:
            vehicle_id = vehicle.vehicle_id
            vehicle_state = env.vehicle_states.get(vehicle_id)
            
            if vehicle_state != VehicleState.IDLE:
                continue
            
            for order in available_orders:
                if hasattr(env, '_can_vehicle_handle_order'):
                    if not env._can_vehicle_handle_order(vehicle, order):
                        continue
                
                profit = self.calculate_assignment_profit(vehicle, order, env)
                profit_matrix.append((profit, vehicle_id, order.order_id))
        
        profit_matrix.sort(key=lambda x: x[0], reverse=True)
        
        assigned_vehicles = set()
        assigned_orders = set()
        
        for profit, vehicle_id, order_id in profit_matrix:
            if vehicle_id not in assigned_vehicles and order_id not in assigned_orders:
                assignments[vehicle_id] = order_id
                assigned_vehicles.add(vehicle_id)
                assigned_orders.add(order_id)
        
        return assignments
    
    def select_actions(self, states, env) -> List[np.ndarray]:
        actions = []
        
        available_orders = [order for order in env.active_orders.values() 
                          if env.order_statuses.get(order.order_id) == OrderStatus.PENDING]
        
        assignments = self.greedy_assignment(available_orders, env.vehicle_states, env)
        
        for i, vehicle in enumerate(env.vehicles):
            vehicle_id = vehicle.vehicle_id
            movement_action = 0.0
            pickup_action = 0.0
            
            if vehicle_id in assignments:
                assigned_order_id = assignments[vehicle_id]
                assigned_order = None
                for order in available_orders:
                    if order.order_id == assigned_order_id:
                        assigned_order = order
                        break
                
                if assigned_order:
                    current_node = vehicle.current_node
                    target_node = assigned_order.start_node
                    
                    if target_node > current_node:
                        movement_action = 0.8
                    elif target_node < current_node:
                        movement_action = -0.8
                    else:
                        movement_action = 0.0
                    
                    pickup_action = 1.0
            else:
                if available_orders:
                    nearest_order = min(available_orders, 
                                      key=lambda o: abs(o.start_node - vehicle.current_node))
                    target_node = nearest_order.start_node
                    
                    if target_node > vehicle.current_node:
                        movement_action = 0.3
                    elif target_node < vehicle.current_node:
                        movement_action = -0.3
                    else:
                        movement_action = 0.0
                else:
                    movement_action = np.random.uniform(-0.2, 0.2)
                
                pickup_action = 0.0
            
            movement_action += np.random.normal(0, 0.05)
            movement_action = np.clip(movement_action, -1.0, 1.0)
            
            actions.append(np.array([movement_action, pickup_action], dtype=np.float32))
        
        return actions


class GreedyBaselineTester:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Initializing Greedy GAP Baseline tester...")
        
        self.episode_rewards = []
        self.episode_pickups = []
        self.episode_deliveries = []
        self.episode_pickup_rates = []
        self.episode_delivery_rates = []
        self.episode_total_orders = []
        
        self._init_environment()
        
        self.greedy_agent = GreedyGAPBaseline(
            num_ground_vehicles=self.config['num_ground_vehicles'],
            num_uavs=self.config['num_uavs']
        )
        
        self.results_dir = os.path.join(current_dir, 'results_greedy_baseline')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _init_environment(self):
        logger.info("Initializing road network environment (greedy mode)...")
        
        env_config = {
            'num_ground_vehicles': self.config['num_ground_vehicles'],
            'num_uavs': self.config['num_uavs'],
            'max_time_steps': self.config['max_time_steps'],
        }
        
        self.env = PureRealRoadNetworkEnvironmentWithConstraints(**env_config)
        
        states = self.env.reset()
        self.state_dim = len(states[0])
        self.action_dim = 2
        self.num_agents = len(states)
        
        logger.info(f"Environment initialized:")
        logger.info(f"  State dimension: {self.state_dim}")
        logger.info(f"  Number of agents: {self.num_agents}")
    
    def run_episode(self, episode: int) -> Dict[str, Any]:
        states = self.env.reset()
        episode_reward = 0.0
        self.env.episode_count = episode
        
        for step in range(self.config['max_time_steps']):
            actions = self.greedy_agent.select_actions(states, self.env)
            
            next_states, rewards, done, info = self.env.step(actions)
            episode_reward += sum(rewards)
            states = next_states
            
            if done:
                break
        
        episode_pickups = self.env.episode_pickups
        episode_deliveries = self.env.episode_deliveries
        total_orders = self.env.total_orders_generated
        
        pickup_rate = (episode_pickups / total_orders) if total_orders > 0 else 0
        delivery_rate = (episode_deliveries / total_orders) if total_orders > 0 else 0
        
        return {
            'episode_reward': episode_reward,
            'episode_pickups': episode_pickups,
            'episode_deliveries': episode_deliveries,
            'total_orders': total_orders,
            'pickup_rate': pickup_rate,
            'delivery_rate': delivery_rate,
        }
    
    def evaluate_baseline(self, num_episodes: int = 100):
        logger.info(f"Starting Greedy GAP Baseline evaluation ({num_episodes} episodes)...")
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_start = time.time()
            
            stats = self.run_episode(episode)
            
            self.episode_rewards.append(stats['episode_reward'])
            self.episode_pickups.append(stats['episode_pickups'])
            self.episode_deliveries.append(stats['episode_deliveries'])
            self.episode_pickup_rates.append(stats['pickup_rate'])
            self.episode_delivery_rates.append(stats['delivery_rate'])
            self.episode_total_orders.append(stats['total_orders'])
            
            if episode % 10 == 0 or episode == num_episodes - 1:
                episode_time = time.time() - episode_start
                
                logger.info(f"Episode {episode:4d} | "
                          f"Reward: {stats['episode_reward']:7.2f} | "
                          f"Pickups: {stats['episode_pickups']:3d}({stats['pickup_rate']:.1%}) | "
                          f"Deliveries: {stats['episode_deliveries']:3d}({stats['delivery_rate']:.1%}) | "
                          f"Total Orders: {stats['total_orders']:3d} | "
                          f"Time: {episode_time:.1f}s")
        
        self._generate_evaluation_report(num_episodes, time.time() - start_time)
        
    def _generate_evaluation_report(self, num_episodes: int, total_time: float):
        logger.info("\n" + "="*60)
        logger.info("Greedy GAP Baseline Evaluation Report")
        logger.info("="*60)
        
        if not self.episode_rewards:
            logger.warning("No evaluation data")
            return
        
        avg_reward = np.mean(self.episode_rewards)
        avg_pickups = np.mean(self.episode_pickups)
        avg_deliveries = np.mean(self.episode_deliveries)
        avg_pickup_rate = np.mean(self.episode_pickup_rates)
        avg_delivery_rate = np.mean(self.episode_delivery_rates)
        avg_total_orders = np.mean(self.episode_total_orders)
        
        logger.info(f"Core Performance Metrics (average, {num_episodes} episodes):")
        logger.info(f"  Average Reward: {avg_reward:.2f}")
        logger.info(f"  Average Pickups: {avg_pickups:.1f} ({avg_pickup_rate:.1%})")
        logger.info(f"  Average Deliveries: {avg_deliveries:.1f} ({avg_delivery_rate:.1%})")
        logger.info(f"  Average Total Orders: {avg_total_orders:.1f}")
        
        if len(self.episode_rewards) >= 10:
            final_avg_reward = np.mean(self.episode_rewards[-10:])
            final_avg_pickups = np.mean(self.episode_pickups[-10:])
            final_avg_deliveries = np.mean(self.episode_deliveries[-10:])
            final_avg_pickup_rate = np.mean(self.episode_pickup_rates[-10:])
            final_avg_delivery_rate = np.mean(self.episode_delivery_rates[-10:])
            
            logger.info(f"Final Performance (last 10 episodes):")
            logger.info(f"  Final Reward: {final_avg_reward:.2f}")
            logger.info(f"  Final Pickups: {final_avg_pickups:.1f} ({final_avg_pickup_rate:.1%})")
            logger.info(f"  Final Deliveries: {final_avg_deliveries:.1f} ({final_avg_delivery_rate:.1%})")
        
        logger.info(f"Total Time: {total_time:.1f}s (average: {total_time/num_episodes:.2f}s/episode)")
        logger.info("="*60)


def main():
    config = {
        'num_ground_vehicles': 5,
        'num_uavs': 2,
        'max_time_steps': 50,
        'num_episodes': 1,
        'debug_mode': True,
    }
    
    print("Greedy GAP Baseline Road Delivery Evaluation")
    print("=" * 60) 
    print(f"Agents: {config['num_ground_vehicles']} carriers + {config['num_uavs']} UAVs")
    print(f"Evaluation episodes: {config['num_episodes']}")
    print(f"Algorithm: Approximate Greedy GAP Solver")
    print(f"Metrics: Pickups, Pickup Rate, Deliveries, Delivery Rate")
    print("=" * 60)
    
    try:
        tester = GreedyBaselineTester(config)
        tester.evaluate_baseline(config['num_episodes'])
        logger.info("Greedy GAP Baseline evaluation completed!")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
