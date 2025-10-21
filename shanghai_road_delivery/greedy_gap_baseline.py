#!/usr/bin/env python3
"""
基于近似贪心算法的GAP问题Baseline
直接替代强化学习算法，在相同环境中测试性能对比

评价指标：
- 成功派送数 (episode_deliveries)  
- 成功派送率 (delivery_rate)
- 接单数 (episode_pickups)
- 接单率 (pickup_rate)
"""

import os
import sys
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'simulator'))

# 导入环境（与RL使用完全相同的环境）
from pure_road_network_environment_with_energy_constraints_centralized import (
    PureRealRoadNetworkEnvironmentWithConstraints, 
    OrderStatus, 
    VehicleState
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GreedyGAPBaseline:
    """基于近似贪心算法的GAP问题baseline"""
    
    def __init__(self, num_ground_vehicles=6, num_uavs=3):
        self.num_ground_vehicles = num_ground_vehicles  
        self.num_uavs = num_uavs
        self.total_vehicles = num_ground_vehicles + num_uavs
        
        # 贪心策略权重参数
        self.distance_weight = 0.4      # 距离权重
        self.capacity_weight = 0.3      # 容量/能量权重  
        self.success_weight = 0.3       # 成功概率权重
        
        logger.info(f"🤖 初始化贪心GAP算法: {num_ground_vehicles}载体 + {num_uavs}UAV")
        
    def calculate_assignment_profit(self, vehicle, order, env) -> float:
        """
        计算车辆-订单分配收益（贪心策略核心）
        """
        profit = 0.0
        
        # 1. 距离收益（距离越近收益越高）
        try:
            distance = abs(vehicle.current_node - order.start_node)
            max_distance = 100.0  
            distance_score = max(0, 1.0 - distance / max_distance)
            profit += self.distance_weight * distance_score
        except:
            profit += self.distance_weight * 0.5
        
        # 2. 容量/能量收益
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
        
        # 3. 成功概率收益
        success_score = 1.0
        if vehicle.vehicle_type == 'uav':
            if hasattr(order, 'weight') and order.weight > 5:
                success_score = 0.1
        
        profit += self.success_weight * success_score
        return profit
    
    def greedy_assignment(self, available_orders, vehicles_status, env) -> Dict[int, int]:
        """贪心分配算法主逻辑"""
        assignments = {}
        
        if not available_orders:
            return assignments
        
        # 计算所有车辆-订单对的收益
        profit_matrix = []
        
        for vehicle in env.vehicles:
            vehicle_id = vehicle.vehicle_id
            vehicle_state = env.vehicle_states.get(vehicle_id)
            
            # 只考虑空闲车辆
            if vehicle_state != VehicleState.IDLE:
                continue
            
            for order in available_orders:
                # 🆕 关键修复：在贪心算法中也要检查车辆约束
                if hasattr(env, '_can_vehicle_handle_order'):
                    if not env._can_vehicle_handle_order(vehicle, order):
                        continue  # 跳过无法处理的订单
                
                profit = self.calculate_assignment_profit(vehicle, order, env)
                profit_matrix.append((profit, vehicle_id, order.order_id))
        
        # 按收益排序（贪心选择）
        profit_matrix.sort(key=lambda x: x[0], reverse=True)
        
        assigned_vehicles = set()
        assigned_orders = set()
        
        # 贪心分配：优先选择收益最高的组合
        for profit, vehicle_id, order_id in profit_matrix:
            if vehicle_id not in assigned_vehicles and order_id not in assigned_orders:
                assignments[vehicle_id] = order_id
                assigned_vehicles.add(vehicle_id)
                assigned_orders.add(order_id)
        
        return assignments
    
    def select_actions(self, states, env) -> np.ndarray:
        """为平台生成贪心动作（单一135维向量）"""
        
        # 获取当前可用订单和车辆
        available_orders = list(env.active_orders.values())
        vehicles = env.vehicles
        
        # 创建9x15的权重矩阵
        weight_matrix = np.zeros((9, 15))
        
        # 简化贪心分配
        available_orders_copy = available_orders[:]
        for i, vehicle in enumerate(vehicles):
            if i >= 9 or len(available_orders_copy) == 0:
                break
                
            # 找到最近的订单
            best_order_idx = None
            best_distance = float('inf')
            
            for j, order in enumerate(available_orders_copy):
                if j >= 15:
                    break
                distance = abs(vehicle.current_node - order.start_node)
                if distance < best_distance:
                    best_distance = distance
                    best_order_idx = j
            
            # 分配最近的订单
            if best_order_idx is not None:
                weight_matrix[i, best_order_idx] = 1.0
                available_orders_copy.pop(best_order_idx)
        
        # 展平为135维向量
        return weight_matrix.flatten()


class GreedyBaselineTester:
    """贪心Baseline测试器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("🧪 初始化贪心GAP Baseline测试器...")
        
        # 统计记录
        self.episode_rewards = []
        self.episode_pickups = []
        self.episode_deliveries = []
        self.episode_pickup_rates = []
        self.episode_delivery_rates = []
        self.episode_total_orders = []
        
        # 初始化环境
        self._init_environment()
        
        # 初始化贪心算法
        self.greedy_agent = GreedyGAPBaseline(
            num_ground_vehicles=self.config['num_ground_vehicles'],
            num_uavs=self.config['num_uavs']
        )
        
        # 结果保存
        self.results_dir = os.path.join(current_dir, 'results_greedy_baseline')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _init_environment(self):
        """初始化环境（与RL训练完全相同）"""
        logger.info("🌍 初始化上海道路网络环境（贪心模式）...")
        
        env_config = {
            'num_ground_vehicles': self.config['num_ground_vehicles'],
            'num_uavs': self.config['num_uavs'],
            'max_time_steps': self.config['max_time_steps'],
            'max_concurrent_orders': self.config.get('max_concurrent_orders', 15),
        }
        
        self.env = PureRealRoadNetworkEnvironmentWithConstraints(**env_config)
        
        # 加载指定天的数据并重置环境
        # day 可能未传入到此方法，使用配置项 init_day 作为初始化日（若配置中没有则为 None）
        # 使用与DDPG相同的测试集: day_28, day_29, day_30
        test_days = [28, 29, 30]
        day = test_days[0]  # 默认使用第一个测试天
        if day is not None:
            self.env.load_road_orders(day)
        states = self.env.reset(day=day)
        # 新环境返回单一状态向量
        self.state_dim = len(states) if not isinstance(states, list) else len(states[0])
        self.action_dim = 2
        self.num_agents = 1  # 现在是单智能体平台
        
        logger.info(f"✅ 环境初始化完成:")
        logger.info(f"   状态维度: {self.state_dim}")
        logger.info(f"   智能体数量: {self.num_agents}")
    
    def run_episode(self, episode: int, day: Optional[int] = None) -> Dict[str, Any]:
        """运行单个episode"""
        # 加载指定天的数据并重置环境
        if day is not None:
            self.env.load_road_orders(day)
        states = self.env.reset(day=day)
        episode_reward = 0.0
        self.env.episode_count = episode
        
        for step in range(self.config['max_time_steps']):
            # 使用贪心算法选择动作
            actions = self.greedy_agent.select_actions(states, self.env)
            
            # 环境执行
            # 环境执行
            next_states, reward, done, info = self.env.step(actions)
            # 新环境返回单个奖励值
            episode_reward += float(reward)
            
            if done:
                break
        
        # 统计
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
        """评估baseline性能"""
        logger.info(f"🚀 开始贪心GAP Baseline评估 ({num_episodes} episodes)...")
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_start = time.time()
            
            # 循环使用测试天数
            test_days = [28, 29, 30]  # 与DDPG相同的测试集
            day = test_days[episode % len(test_days)]
            stats = self.run_episode(episode, day=day)
            
            # 记录统计
            self.episode_rewards.append(stats['episode_reward'])
            self.episode_pickups.append(stats['episode_pickups'])
            self.episode_deliveries.append(stats['episode_deliveries'])
            self.episode_pickup_rates.append(stats['pickup_rate'])
            self.episode_delivery_rates.append(stats['delivery_rate'])
            self.episode_total_orders.append(stats['total_orders'])
            
            # 输出信息
            if episode % 10 == 0 or episode == num_episodes - 1:
                episode_time = time.time() - episode_start
                
                logger.info(f"Episode {episode:4d} | "
                          f"奖励: {stats['episode_reward']:7.2f} | "
                          f"接单: {stats['episode_pickups']:3d}({stats['pickup_rate']:.1%}) | "
                          f"配送: {stats['episode_deliveries']:3d}({stats['delivery_rate']:.1%}) | "
                          f"总订单: {stats['total_orders']:3d} | "
                          f"时间: {episode_time:.1f}s")
        
        # 生成评估报告
        self._generate_evaluation_report(num_episodes, time.time() - start_time)
        
    def _generate_evaluation_report(self, num_episodes: int, total_time: float):
        """生成评估报告"""
        logger.info("\n" + "="*60)
        logger.info("🎯 贪心GAP Baseline 评估报告")
        logger.info("="*60)
        
        if not self.episode_rewards:
            logger.warning("⚠️ 无评估数据")
            return
        
        # 计算统计指标
        avg_reward = np.mean(self.episode_rewards)
        avg_pickups = np.mean(self.episode_pickups)
        avg_deliveries = np.mean(self.episode_deliveries)
        avg_pickup_rate = np.mean(self.episode_pickup_rates)
        avg_delivery_rate = np.mean(self.episode_delivery_rates)
        avg_total_orders = np.mean(self.episode_total_orders)
        
        # 输出核心指标
        logger.info(f"📊 核心性能指标 (平均值, {num_episodes} episodes):")
        logger.info(f"   💰 平均奖励: {avg_reward:.2f}")
        logger.info(f"   📦 平均接单数: {avg_pickups:.1f} ({avg_pickup_rate:.1%})")
        logger.info(f"   🚚 平均配送数: {avg_deliveries:.1f} ({avg_delivery_rate:.1%})")
        logger.info(f"   📋 平均总订单: {avg_total_orders:.1f}")
        
        # 最后10个episodes的性能
        if len(self.episode_rewards) >= 10:
            final_avg_reward = np.mean(self.episode_rewards[-10:])
            final_avg_pickups = np.mean(self.episode_pickups[-10:])
            final_avg_deliveries = np.mean(self.episode_deliveries[-10:])
            final_avg_pickup_rate = np.mean(self.episode_pickup_rates[-10:])
            final_avg_delivery_rate = np.mean(self.episode_delivery_rates[-10:])
            
            logger.info(f"\n📈 最终性能 (最后10 episodes):")
            logger.info(f"   💰 最终奖励: {final_avg_reward:.2f}")
            logger.info(f"   📦 最终接单数: {final_avg_pickups:.1f} ({final_avg_pickup_rate:.1%})")
            logger.info(f"   🚚 最终配送数: {final_avg_deliveries:.1f} ({final_avg_delivery_rate:.1%})")
        
        logger.info(f"\n⏱️ 运行时间: {total_time:.1f}s (平均: {total_time/num_episodes:.2f}s/episode)")
        logger.info("="*60)


def main():
    # 贪心Baseline配置（与RL训练完全相同的环境参数）
    config = {
        'num_ground_vehicles': 6,  # 地面载体数量
        'num_uavs': 3,      # UAV数量  
        'max_time_steps': 120,
        'max_concurrent_orders': 15,  # 最大并发订单数（与DDPG对齐）
        'num_episodes': 1,      # 评估轮数
        'debug_mode': True,       # 调试模式
    }
    
    print("🤖 贪心GAP Baseline 上海道路配送评估")
    print("=" * 60) 
    print(f"🎯 智能体: {config['num_ground_vehicles']}载体 + {config['num_uavs']}UAV")
    print(f"📊 评估轮数: {config['num_episodes']}")
    print(f"🔧 算法: 近似贪心GAP求解")
    print(f"📈 评价指标: 接单数、接单率、配送数、配送率")
    print("=" * 60)
    
    try:
        tester = GreedyBaselineTester(config)
        tester.evaluate_baseline(config['num_episodes'])
        logger.info("✅ 贪心GAP Baseline评估完成!")
        
    except KeyboardInterrupt:
        logger.info("⚠️ 评估被用户中断")
    except Exception as e:
        logger.error(f"❌ 评估过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
