import numpy as np
import random
from collections import deque
import logging

logger = logging.getLogger(__name__)

class DQNScheduler:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.q_table = {}
        self.training_history = {'rewards': [], 'losses': []}
        
    def _get_state_key(self, state):
        state_discrete = [round(val, 1) for val in state]
        return tuple(state_discrete)
        
    def update_target_model(self):
        pass
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        state_key = self._get_state_key(state)
        
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        if state_key in self.q_table:
            return np.argmax(self.q_table[state_key])
        else:
            self.q_table[state_key] = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
            
        minibatch = random.sample(self.memory, self.batch_size)
        loss = 0
        
        for state, action, reward, next_state, done in minibatch:
            state_key = self._get_state_key(state)
            next_state_key = self._get_state_key(next_state)
            
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)
            
            current_q = self.q_table[state_key][action]
            
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * np.max(self.q_table[next_state_key])
            
            self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
            loss += (target_q - current_q) ** 2
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss / self.batch_size
        
    def get_training_history(self):
        if not hasattr(self, 'training_history') or self.training_history is None:
            self.training_history = {'rewards': [], 'losses': []}
        return self.training_history

    def train(self, env, num_episodes, max_time=None, max_steps_per_episode=500):
        import time
        
        self.training_history = {'rewards': [], 'losses': []}
        start_time = time.time()
        
        for episode in range(num_episodes):
            if max_time and (time.time() - start_time) > max_time:
                logger.warning(f"Training stopped after {episode} episodes due to time limit")
                break
                
            state = env.reset()
            total_reward = 0
            done = False
            losses = []
            step_count = 0
            
            while not done and step_count < max_steps_per_episode:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step_count += 1
                loss = self.replay()
                if loss > 0:
                    losses.append(loss)
                if max_time and (time.time() - start_time) > max_time:
                    logger.warning(f"Training episode interrupted due to time limit after {step_count} steps")
                    done = True
                    break
            
            episode_avg_loss = np.mean(losses) if losses else 0
            self.training_history['rewards'].append(total_reward)
            self.training_history['losses'].append(episode_avg_loss)
            
            logger.info(f"Episode: {episode+1}/{num_episodes}, Reward: {total_reward:.2f}, "
                        f"Loss: {episode_avg_loss:.6f}, Epsilon: {self.epsilon:.4f}")
            
        return self.training_history
        
    def save(self, filepath):
        np.save(filepath, self.q_table)
        
    def load(self, filepath):
        self.q_table = np.load(filepath, allow_pickle=True).item()
