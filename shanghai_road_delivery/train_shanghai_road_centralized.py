import os
import sys
import json
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'simulator'))

from road_maddpg_networks import RoadMADDPG
from pure_road_network_environment_with_energy_constraints_centralized import PureRealRoadNetworkEnvironmentWithConstraints

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CentralizedTrainingManager:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.episode_rewards = []
        self.episode_pickups = []
        self.episode_deliveries = []
        self.episode_pickup_rates = []
        self.episode_delivery_rates = []
        self.episode_total_orders = []
        
        self.max_concurrent_orders = 15
        self.action_dim = self.max_concurrent_orders
        
        self._init_environment()
        self._init_centralized_maddpg()
        
        self.results_dir = os.path.join(current_dir, 'results_centralized_1')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _init_environment(self):
        logger.info("Initializing road network environment...")
        
        env_config = {
            'num_ground_vehicles': self.config['num_ground_vehicles'],
            'num_uavs': self.config['num_uavs'],
            'max_time_steps': self.config['max_time_steps'],
            "max_concurrent_orders": self.max_concurrent_orders,
        }
        
        self.env = PureRealRoadNetworkEnvironmentWithConstraints(**env_config)
        
        states = self.env.reset()
        self.state_dim = len(states[0])
        self.num_agents = len(states)
        
        logger.info(f"Environment initialized:")
        logger.info(f"   State dimension: {self.state_dim}")
        logger.info(f"   Action dimension: {self.action_dim}")
        logger.info(f"   Number of agents: {self.num_agents}")
    
    def _init_centralized_maddpg(self):
        logger.info("Initializing centralized MADDPG networks...")
        
        maddpg_config = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'num_agents': self.num_agents,
            'lr_actor': self.config['lr_actor'],
            'lr_critic': self.config['lr_critic'],
            'gamma': self.config['gamma'],
            'tau': self.config['tau'],
            'batch_size': self.config['batch_size'], 
        }
        
        self.maddpg = RoadMADDPG(**maddpg_config)
        logger.info("Centralized MADDPG initialized")
    
    def train_episode(self, episode: int) -> Dict[str, float]:
        states = self.env.reset()
        episode_reward = 0
        total_pickups = 0
        total_deliveries = 0
        total_orders = 0
        episode_pickups = 0
        episode_deliveries = 0
        
        self.env.episode_count = episode
        
        for step in range(self.config['max_time_steps']):
            actions = self.maddpg.get_actions(states, add_noise=True)
            
            next_states, rewards, done, info = self.env.step(actions)
            
            episode_reward += sum(rewards)
            if 'pickup_count' in info:
                total_pickups += info['pickup_count']
            if 'delivery_count' in info:
                total_deliveries += info['delivery_count']
            if 'total_orders' in info:
                total_orders = info['total_orders']
            
            self.maddpg.store_transition(states, actions, rewards, next_states, done)
            
            if len(self.maddpg.memory) >= self.config['batch_size']:
                if step % self.config.get('train_interval', 1) == 0:
                    self.maddpg.update()
            
            episode_pickups = self.env.episode_pickups
            episode_deliveries = self.env.episode_deliveries

            states = next_states
            
            if done:
                break
            
            if step < 5 and episode % 10 == 0:
                print(f"    Step{step}: reward={sum(rewards):.2f}, active_orders={len(self.env.active_orders)}")
        
        pickup_rate = (episode_pickups / total_orders ) if total_orders > 0 else 0
        delivery_rate = (episode_deliveries / total_orders ) if total_orders > 0 else 0
        
        return {
            'episode_reward': episode_reward,
            'total_pickups': total_pickups,
            'total_deliveries': total_deliveries,
            'episode_pickups': episode_pickups,
            'episode_deliveries': episode_deliveries,
            'total_orders': total_orders,
            'pickup_rate': pickup_rate,
            'delivery_rate': delivery_rate,
            'final_active_orders': len(self.env.active_orders)
        }
    
    def train(self):
        logger.info("Starting centralized MADDPG training...")
        
        start_time = time.time()
        best_reward = -float('inf')
        
        for episode in range(self.config['num_episodes']):
            episode_start = time.time()
            
            stats = self.train_episode(episode)
            
            self.episode_rewards.append(stats['episode_reward'])
            self.episode_pickups.append(stats['episode_pickups'])
            self.episode_deliveries.append(stats['episode_deliveries'])
            self.episode_pickup_rates.append(stats['pickup_rate'])
            self.episode_delivery_rates.append(stats['delivery_rate'])
            self.episode_total_orders.append(stats['total_orders'])
            
            if episode % self.config.get('log_interval', 10) == 0:
                episode_time = time.time() - episode_start
                
                logger.info(f"Episode {episode:4d} | "
                          f"Reward: {stats['episode_reward']:7.2f}  | "
                          f"Pickups: {stats['episode_pickups']:3d}({stats['pickup_rate']:.1%}) "
                          f"Deliveries: {stats['episode_deliveries']:3d}({stats['delivery_rate']:.1%}) "
                          f"Total Orders: {stats['total_orders']:3d}  | "
                          f"Time: {episode_time:.1f}s")
                
                if stats['total_orders'] < 5:
                    logger.warning(f"Order generation anomaly: total_orders={stats['total_orders']}")
            
            if stats['episode_reward'] > best_reward:
                best_reward = stats['episode_reward']
                self.save_model(f'best_centralized_episode_{episode}.pth')
            
            if episode % self.config.get('save_interval', 100) == 0 and episode > 0:
                self.save_model(f'centralized_episode_{episode}.pth')
                self.save_training_curves()
        
        total_time = time.time() - start_time
        logger.info(f"Training completed! Total time: {total_time:.1f}s")
        
        self.save_model('centralized_final.pth')
        self.save_training_curves()
        self.save_training_config()
    
    def save_model(self, filename: str):
        model_path = os.path.join(self.results_dir, filename)
        self.maddpg.save_models(model_path)
        logger.info(f"Model saved: {filename}")
    
    def save_training_curves(self):
        plt.figure(figsize=(20, 12))
        
        plt.subplot(2, 3, 1)
        plt.plot(self.episode_rewards, alpha=0.6, color='blue', label='Episode Reward')
        if len(self.episode_rewards) > 10:
            window = min(50, len(self.episode_rewards) // 4)
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(self.episode_rewards)), moving_avg, 
                    color='red', linewidth=2, label=f'Moving Avg ({window})')
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(self.episode_pickups, alpha=0.7, color='green')
        plt.title('Pickup Count per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Pickups')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.plot(self.episode_deliveries, alpha=0.7, color='orange')
        plt.title('Delivery Count per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Deliveries')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 4)
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Full Match Line')
        plt.title('Order Match Rate per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Match Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 100 else self.episode_rewards
        plt.hist(recent_rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Recent Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 6)
        episodes = range(len(self.episode_rewards))
        
        if len(self.episode_rewards) > 1:
            norm_rewards = np.array(self.episode_rewards) / max(max(self.episode_rewards), 1)
            norm_pickups = np.array(self.episode_pickups) / max(max(self.episode_pickups), 1)
            norm_deliveries = np.array(self.episode_deliveries) / max(max(self.episode_deliveries), 1)
            
            plt.plot(episodes, norm_rewards, alpha=0.7, label='Normalized Rewards')
            plt.plot(episodes, norm_pickups, alpha=0.7, label='Normalized Pickups')
            plt.plot(episodes, norm_deliveries, alpha=0.7, label='Normalized Deliveries')
        
        plt.title('Normalized Metrics Comparison')
        plt.xlabel('Episode')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'centralized_training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training curves saved")
    
    def save_training_config(self):
        config_data = {
            'training_config': self.config,
            'final_stats': {
                'total_episodes': len(self.episode_rewards),
                'best_reward': max(self.episode_rewards) if self.episode_rewards else 0,
                'avg_final_reward': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0,
                'avg_final_pickups': np.mean(self.episode_pickups[-10:]) if len(self.episode_pickups) >= 10 else 0,
                'avg_final_deliveries': np.mean(self.episode_deliveries[-10:]) if len(self.episode_deliveries) >= 10 else 0,
                'avg_final_pickup_rate': np.mean(self.episode_pickup_rates[-10:]) if len(self.episode_pickup_rates) >= 10 else 0,
                'avg_final_delivery_rate': np.mean(self.episode_delivery_rates[-10:]) if len(self.episode_delivery_rates) >= 10 else 0,
                'avg_final_total_orders': np.mean(self.episode_total_orders[-10:]) if len(self.episode_total_orders) >= 10 else 0,
            },
            'training_timestamp': datetime.now().isoformat(),
            'environment_info': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'num_agents': self.num_agents,
                'device': str(self.device)
            }
        }
        
        config_path = os.path.join(self.results_dir, 'centralized_training_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info("Training configuration saved")

def main():
    config = {
        'num_ground_vehicles': 5,
        'num_uavs': 2,
        'max_time_steps': 50,
        'num_episodes': 500,
        'lr_actor': 0.05,
        'lr_critic': 0.05,
        'gamma': 0.95,
        'tau': 0.02,
        'batch_size': 32,
        'train_interval': 1,
        'log_interval': 10,
        'save_interval': 50,
        'debug_mode': True,
    }
    
    print("Centralized MADDPG Road Delivery Training")
    print("=" * 60) 
    print(f"Agents: {config['num_ground_vehicles']} carriers + {config['num_uavs']} UAVs")
    print(f"Training episodes: {config['num_episodes']}")
    print("=" * 60)
    
    try:
        trainer = CentralizedTrainingManager(config)
        trainer.train()
        
        print("\nCentralized MADDPG training completed!")
        print(f"Results saved to: {trainer.results_dir}")
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()