"""
Reward calculator for Pokemon Leaf Green RL training.
Implements sophisticated reward shaping for efficient learning.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from .state_parser import GameState, GameRegion, Pokemon
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    # Major achievements
    badge_reward: float = 100.0
    new_pokemon_reward: float = 10.0
    level_up_reward: float = 5.0
    
    # Efficiency penalties
    step_penalty: float = -0.01
    time_penalty: float = -0.001
    
    # Battle rewards/penalties
    battle_win_reward: float = 20.0
    battle_lose_penalty: float = -10.0
    damage_penalty_scale: float = 0.1
    healing_reward: float = 2.0
    
    # Exploration rewards
    new_area_reward: float = 5.0
    revisit_penalty: float = -0.5
    progress_reward_scale: float = 1.0
    
    # Menu efficiency
    menu_time_penalty: float = -0.05
    text_skip_reward: float = 0.1
    
    # Special achievements
    rare_pokemon_bonus: float = 50.0
    evolution_bonus: float = 15.0
    item_found_reward: float = 3.0
    
    # Money and items
    money_gain_scale: float = 0.001
    money_loss_penalty: float = 0.002


class RewardTracker:
    """Tracks game progress for reward calculation."""
    
    def __init__(self):
        """Initialize reward tracker."""
        self.reset()
    
    def reset(self):
        """Reset tracking state."""
        # Badge tracking
        self.badges_earned = [False] * 8
        self.badge_count = 0
        
        # Pokemon tracking
        self.pokemon_seen = set()
        self.pokemon_caught = set()
        self.max_pokemon_level = 1
        self.party_levels = [0] * 6
        
        # Location tracking
        self.visited_areas = set()
        self.area_visit_count = {}
        self.current_region = GameRegion.UNKNOWN
        self.total_steps = 0
        
        # Battle tracking
        self.battles_won = 0
        self.battles_lost = 0
        self.total_damage_taken = 0
        self.total_damage_dealt = 0
        
        # Progress tracking
        self.money = 0
        self.items_found = 0
        self.evolutions_witnessed = 0
        
        # Time tracking
        self.time_in_menu = 0
        self.time_in_battle = 0
        self.time_in_text = 0
        self.total_game_time = 0
        
        # Efficiency metrics
        self.actions_per_progress = []
        self.last_progress_time = 0
        
        logger.info("Reward tracker reset")


class RewardCalculator:
    """Calculates rewards based on game state changes."""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize reward calculator.
        
        Args:
            config: Reward configuration parameters
        """
        self.config = config or RewardConfig()
        self.tracker = RewardTracker()
        self.last_state: Optional[GameState] = None
        
        # Rare Pokemon IDs (Legendary, Starter, etc.)
        self.rare_pokemon_ids = {
            144, 145, 146, 150, 151,  # Legendary birds + Mewtwo + Mew
            1, 4, 7,  # Starter Pokemon
            147, 148, 149,  # Dratini line
            131, 132,  # Lapras, Ditto
        }
        
        # Evolution chains for bonus detection
        self.evolution_chains = {
            1: 2, 2: 3,      # Bulbasaur line
            4: 5, 5: 6,      # Charmander line
            7: 8, 8: 9,      # Squirtle line
            # Add more evolution chains as needed
        }
        
        logger.info("Reward calculator initialized")
    
    def calculate_reward(
        self,
        current_state: GameState,
        action: int,
        info: Optional[Dict] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward based on state transition.
        
        Args:
            current_state: Current game state
            action: Action taken
            info: Additional information
            
        Returns:
            Tuple[float, Dict[str, float]]: Total reward and reward breakdown
        """
        if self.last_state is None:
            self.last_state = current_state
            return 0.0, {}
        
        reward_breakdown = {}
        total_reward = 0.0
        
        try:
            # Update tracker with current state
            self._update_tracker(current_state)
            
            # Calculate individual reward components
            badge_reward = self._calculate_badge_reward(current_state)
            pokemon_reward = self._calculate_pokemon_reward(current_state)
            exploration_reward = self._calculate_exploration_reward(current_state)
            battle_reward = self._calculate_battle_reward(current_state)
            efficiency_reward = self._calculate_efficiency_reward(current_state, action)
            progress_reward = self._calculate_progress_reward(current_state)
            money_reward = self._calculate_money_reward(current_state)
            
            # Combine rewards
            reward_breakdown = {
                'badge': badge_reward,
                'pokemon': pokemon_reward,
                'exploration': exploration_reward,
                'battle': battle_reward,
                'efficiency': efficiency_reward,
                'progress': progress_reward,
                'money': money_reward
            }
            
            total_reward = sum(reward_breakdown.values())
            
            # Apply reward clipping to prevent extreme values
            total_reward = np.clip(total_reward, -50.0, 200.0)
            
            self.last_state = current_state
            
            return total_reward, reward_breakdown
            
        except Exception as e:
            logger.error(f"Reward calculation failed: {e}")
            return 0.0, {}
    
    def _update_tracker(self, state: GameState):
        """Update internal tracking state."""
        # Update basic counters
        self.tracker.total_steps += 1
        self.tracker.total_game_time = state.game_time
        
        # Update money
        self.tracker.money = state.player.money
        
        # Update location tracking
        region_key = f"{state.current_region.value}_{state.player.map_id}"
        if region_key not in self.tracker.visited_areas:
            self.tracker.visited_areas.add(region_key)
        
        # Update area visit count
        self.tracker.area_visit_count[region_key] = \
            self.tracker.area_visit_count.get(region_key, 0) + 1
        
        # Update current region
        self.tracker.current_region = state.current_region
        
        # Update time tracking
        if state.in_menu:
            self.tracker.time_in_menu += 1
        if state.in_battle:
            self.tracker.time_in_battle += 1
        if state.text_displayed:
            self.tracker.time_in_text += 1
    
    def _calculate_badge_reward(self, state: GameState) -> float:
        """Calculate reward for earning gym badges."""
        reward = 0.0
        
        # Check for new badges
        for i, badge in enumerate(state.player.badges):
            if badge and not self.tracker.badges_earned[i]:
                reward += self.config.badge_reward
                self.tracker.badges_earned[i] = True
                self.tracker.badge_count += 1
                logger.info(f"New badge earned! Total: {self.tracker.badge_count}/8")
        
        return reward
    
    def _calculate_pokemon_reward(self, state: GameState) -> float:
        """Calculate reward for Pokemon-related progress."""
        reward = 0.0
        
        # Check for new Pokemon caught
        for pokemon in state.party_pokemon:
            species_id = pokemon.species_id
            
            if species_id not in self.tracker.pokemon_caught:
                base_reward = self.config.new_pokemon_reward
                
                # Bonus for rare Pokemon
                if species_id in self.rare_pokemon_ids:
                    base_reward += self.config.rare_pokemon_bonus
                    logger.info(f"Rare Pokemon caught: {species_id}")
                
                reward += base_reward
                self.tracker.pokemon_caught.add(species_id)
                logger.info(f"New Pokemon caught: {species_id}")
            
            # Level up rewards
            if pokemon.level > self.tracker.max_pokemon_level:
                levels_gained = pokemon.level - self.tracker.max_pokemon_level
                reward += levels_gained * self.config.level_up_reward
                self.tracker.max_pokemon_level = pokemon.level
                logger.info(f"Pokemon leveled up to {pokemon.level}")
            
            # Evolution detection (simplified)
            if (species_id in self.evolution_chains.values() and
                species_id not in self.tracker.pokemon_seen):
                reward += self.config.evolution_bonus
                self.tracker.evolutions_witnessed += 1
                logger.info(f"Evolution witnessed: {species_id}")
            
            self.tracker.pokemon_seen.add(species_id)
        
        return reward
    
    def _calculate_exploration_reward(self, state: GameState) -> float:
        """Calculate reward for map exploration."""
        reward = 0.0
        
        region_key = f"{state.current_region.value}_{state.player.map_id}"
        
        # Reward for visiting new areas
        if region_key not in self.tracker.visited_areas:
            reward += self.config.new_area_reward
            logger.info(f"New area discovered: {state.current_region.value}")
        
        # Penalty for revisiting areas too frequently
        visit_count = self.tracker.area_visit_count.get(region_key, 0)
        if visit_count > 10:  # Arbitrary threshold
            reward += self.config.revisit_penalty * (visit_count - 10)
        
        # Progress reward based on unique areas visited
        progress_multiplier = len(self.tracker.visited_areas) / 50.0  # Normalize
        reward += self.config.progress_reward_scale * progress_multiplier
        
        return reward
    
    def _calculate_battle_reward(self, state: GameState) -> float:
        """Calculate battle-related rewards and penalties."""
        reward = 0.0
        
        if not state.in_battle or not self.last_state:
            return reward
        
        # HP change detection
        if (state.player.current_pokemon and 
            self.last_state.player.current_pokemon):
            
            current_hp = state.player.current_pokemon.hp
            last_hp = self.last_state.player.current_pokemon.hp
            hp_change = current_hp - last_hp
            
            if hp_change < 0:  # Damage taken
                damage = abs(hp_change)
                reward -= damage * self.config.damage_penalty_scale
                self.tracker.total_damage_taken += damage
            elif hp_change > 0:  # Healing
                reward += self.config.healing_reward
        
        # Battle outcome detection (simplified)
        if self.last_state.in_battle and not state.in_battle:
            # Battle ended - determine outcome based on Pokemon HP
            if (state.party_pokemon and 
                state.party_pokemon[0].hp > 0):
                reward += self.config.battle_win_reward
                self.tracker.battles_won += 1
                logger.info("Battle won!")
            else:
                reward += self.config.battle_lose_penalty
                self.tracker.battles_lost += 1
                logger.info("Battle lost!")
        
        return reward
    
    def _calculate_efficiency_reward(self, state: GameState, action: int) -> float:
        """Calculate efficiency-related rewards and penalties."""
        reward = 0.0
        
        # Basic step penalty to encourage efficiency
        reward += self.config.step_penalty
        
        # Time penalties for being stuck in menus/text
        if state.in_menu:
            reward += self.config.menu_time_penalty
        
        # Small reward for progressing through text quickly
        if self.last_state and self.last_state.text_displayed and not state.text_displayed:
            reward += self.config.text_skip_reward
        
        # Game time penalty (very small)
        reward += self.config.time_penalty
        
        return reward
    
    def _calculate_progress_reward(self, state: GameState) -> float:
        """Calculate overall game progress reward."""
        reward = 0.0
        
        # Progress score based on multiple factors
        badge_progress = sum(state.player.badges) / 8.0
        pokemon_progress = min(len(self.tracker.pokemon_caught) / 151.0, 1.0)
        exploration_progress = min(len(self.tracker.visited_areas) / 50.0, 1.0)
        
        # Weighted progress score
        progress_score = (
            badge_progress * 0.5 +
            pokemon_progress * 0.3 +
            exploration_progress * 0.2
        )
        
        # Small continuous reward for maintaining progress
        reward += progress_score * 0.1
        
        return reward
    
    def _calculate_money_reward(self, state: GameState) -> float:
        """Calculate money-related rewards."""
        reward = 0.0
        
        if self.last_state:
            money_change = state.player.money - self.last_state.player.money
            
            if money_change > 0:  # Gained money
                reward += money_change * self.config.money_gain_scale
            elif money_change < 0:  # Lost money
                reward -= abs(money_change) * self.config.money_loss_penalty
        
        return reward
    
    def get_reward_stats(self) -> Dict[str, float]:
        """Get reward calculation statistics."""
        total_time = max(self.tracker.total_game_time, 1)
        
        stats = {
            'badges_earned': self.tracker.badge_count,
            'pokemon_caught': len(self.tracker.pokemon_caught),
            'areas_explored': len(self.tracker.visited_areas),
            'battles_won': self.tracker.battles_won,
            'battles_lost': self.tracker.battles_lost,
            'win_rate': self.tracker.battles_won / max(self.tracker.battles_won + self.tracker.battles_lost, 1),
            'damage_taken': self.tracker.total_damage_taken,
            'evolutions_seen': self.tracker.evolutions_witnessed,
            'efficiency_score': self.tracker.total_steps / total_time,
            'menu_time_ratio': self.tracker.time_in_menu / total_time,
            'battle_time_ratio': self.tracker.time_in_battle / total_time,
            'money': self.tracker.money,
            'max_pokemon_level': self.tracker.max_pokemon_level
        }
        
        return stats
    
    def reset(self):
        """Reset the reward calculator state."""
        self.tracker.reset()
        self.last_state = None
        logger.info("Reward calculator reset")


class AdaptiveRewardCalculator(RewardCalculator):
    """Adaptive reward calculator that adjusts rewards based on training progress."""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """Initialize adaptive reward calculator."""
        super().__init__(config)
        
        # Adaptive parameters
        self.episode_count = 0
        self.recent_rewards = []
        self.adaptation_window = 100  # Episodes to consider for adaptation
        
        # Adaptive scaling factors
        self.badge_scale = 1.0
        self.exploration_scale = 1.0
        self.efficiency_scale = 1.0
        
    def calculate_reward(
        self,
        current_state: GameState,
        action: int,
        info: Optional[Dict] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate adaptive reward with dynamic scaling."""
        base_reward, breakdown = super().calculate_reward(current_state, action, info)
        
        # Apply adaptive scaling
        if 'badge' in breakdown:
            breakdown['badge'] *= self.badge_scale
        if 'exploration' in breakdown:
            breakdown['exploration'] *= self.exploration_scale
        if 'efficiency' in breakdown:
            breakdown['efficiency'] *= self.efficiency_scale
        
        # Recalculate total
        total_reward = sum(breakdown.values())
        
        # Track for adaptation
        self.recent_rewards.append(total_reward)
        if len(self.recent_rewards) > self.adaptation_window:
            self.recent_rewards.pop(0)
        
        return total_reward, breakdown
    
    def adapt_rewards(self, episode_stats: Dict[str, float]):
        """Adapt reward scaling based on training progress."""
        self.episode_count += 1
        
        if self.episode_count % 50 == 0:  # Adapt every 50 episodes
            # Analyze recent performance
            avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0
            
            # Adjust badge rewards if not earning badges frequently
            if episode_stats.get('badges_earned', 0) < 1:
                self.badge_scale = min(self.badge_scale * 1.1, 2.0)
            
            # Adjust exploration if staying in same areas
            if episode_stats.get('areas_explored', 0) < 5:
                self.exploration_scale = min(self.exploration_scale * 1.1, 2.0)
            
            # Adjust efficiency based on average reward
            if avg_reward < -10:
                self.efficiency_scale = max(self.efficiency_scale * 0.9, 0.5)
            
            logger.info(f"Adapted reward scales: badge={self.badge_scale:.2f}, "
                       f"exploration={self.exploration_scale:.2f}, "
                       f"efficiency={self.efficiency_scale:.2f}")


if __name__ == "__main__":
    # Test the reward calculator
    from .state_parser import PlayerState, Pokemon
    
    # Create test states
    initial_state = GameState(
        player=PlayerState(x=100, y=100, map_id=0, direction=0,
                          money=1000, badges=[False]*8, pokemon_count=1,
                          current_pokemon=Pokemon(species_id=1, level=5, hp=20, max_hp=20,
                                                attack=10, defense=10, speed=10,
                                                special_attack=10, special_defense=10,
                                                experience=100)),
        party_pokemon=[],
        current_region=GameRegion.PALLET_TOWN,
        in_battle=False,
        in_menu=False,
        text_displayed=False,
        game_time=1000
    )
    
    # Test reward calculation
    calculator = RewardCalculator()
    reward, breakdown = calculator.calculate_reward(initial_state, 0)
    
    print(f"Initial reward: {reward}")
    print(f"Breakdown: {breakdown}")
    
    # Simulate badge earned
    badge_state = GameState(
        player=PlayerState(x=100, y=100, map_id=0, direction=0,
                          money=1000, badges=[True] + [False]*7, pokemon_count=1,
                          current_pokemon=initial_state.player.current_pokemon),
        party_pokemon=[],
        current_region=GameRegion.PEWTER_CITY,
        in_battle=False,
        in_menu=False,
        text_displayed=False,
        game_time=2000
    )
    
    reward, breakdown = calculator.calculate_reward(badge_state, 5)
    print(f"\nBadge earned reward: {reward}")
    print(f"Breakdown: {breakdown}")
    
    # Get stats
    stats = calculator.get_reward_stats()
    print(f"\nReward stats: {stats}")
