import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from models import Task, CPUCore

logger = logging.getLogger(__name__)

class CPUEnvironment:
    def __init__(self, num_cpus=4, workload=None, 
                 reward_latency_weight=0.7, reward_throughput_weight=0.3,
                 max_steps=1000):
        self.num_cpus = num_cpus
        self.cpus = [CPUCore(id=i) for i in range(num_cpus)]
        self.tasks = workload if workload else []
        self.current_time = 0.0
        self.time_step = 1.0  
        self.reward_latency_weight = reward_latency_weight
        self.reward_throughput_weight = reward_throughput_weight
        self.max_steps = max_steps
        self.step_count = 0
        self.completed_tasks_history = []
   
        self.observation_space_size = num_cpus * 2 + 3 
        self.action_space_size = (len(self.tasks) + 1) * num_cpus  
        
    def reset(self) -> np.ndarray:
        self.current_time = 0.0
        self.step_count = 0
        self.completed_tasks_history = []
        
        for cpu in self.cpus:
            cpu.current_task = None
            cpu.idle_time = 0.0
            cpu.busy_time = 0.0
       
        for task in self.tasks:
            task.remaining_time = task.execution_time
            task.start_time = None
            task.completion_time = None
            task.assigned_cpu = None
            task.is_completed = False
            task.is_running = False
            
        return self.get_state()
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_count += 1
        prev_completed_tasks = len(self.completed_tasks_history)
        
        if action != self.get_no_op_action():
            task_id, cpu_id = self.decode_action(action)
            self._assign_task_to_cpu(task_id, cpu_id)
      
        self._advance_time(self.time_step)
      
        for cpu in self.cpus:
            if not cpu.is_idle() and cpu.current_task.remaining_time <= 0:
                completed_task = cpu.release_task(self.current_time)
                if completed_task:
                    self.completed_tasks_history.append(completed_task)
     
        reward = self._calculate_reward(prev_completed_tasks)
        new_state = self.get_state()
        done = self._is_episode_done()
   
        info = {
            'current_time': self.current_time,
            'completed_tasks': len(self.completed_tasks_history),
            'avg_latency': self.get_average_latency()
        }
        
        return new_state, reward, done, info
    
    def get_state(self) -> np.ndarray:
        state = []
        
        for cpu in self.cpus:
            state.append(0 if cpu.is_idle() else 1)
   
        for cpu in self.cpus:
            state.append(cpu.get_utilization())
            
        waiting_tasks = sum(1 for task in self.tasks 
                           if not task.is_completed and not task.is_running)
        state.append(waiting_tasks / max(len(self.tasks), 1))
   
        completed_tasks = sum(1 for task in self.tasks if task.is_completed)
        state.append(completed_tasks / max(len(self.tasks), 1))
    
        if waiting_tasks > 0:
            avg_priority = np.mean([task.priority for task in self.tasks 
                                   if not task.is_completed and not task.is_running])
            state.append(avg_priority)
        else:
            state.append(0)
            
        return np.array(state, dtype=np.float32)
    
    def encode_action(self, task_id: int, cpu_id: int) -> int:
        return (task_id + 1) * self.num_cpus + cpu_id
        
    def decode_action(self, action: int) -> Tuple[int, int]:
        cpu_id = action % self.num_cpus
        task_id = action // self.num_cpus - 1  
        return task_id, cpu_id
        
    def get_no_op_action(self) -> int:
        return 0
        
    def _assign_task_to_cpu(self, task_id: int, cpu_id: int) -> bool:
        if task_id < 0 or task_id >= len(self.tasks):
            logger.warning(f"Invalid task_id: {task_id}")
            return False
            
        if cpu_id < 0 or cpu_id >= self.num_cpus:
            logger.warning(f"Invalid cpu_id: {cpu_id}")
            return False
            
        task = self.tasks[task_id]
        cpu = self.cpus[cpu_id]
        
        if task.is_completed or task.is_running:
            return False
          
        if not cpu.is_idle():
            return False
        
        cpu.assign_task(task, self.current_time)
        return True
        
    def _advance_time(self, time_delta: float) -> None:
        self.current_time += time_delta
        for cpu in self.cpus:
            cpu.update(time_delta)
            
    def _calculate_reward(self, prev_completed_tasks: int) -> float:
        completed_tasks_reward = len(self.completed_tasks_history) - prev_completed_tasks
        
        avg_latency = self.get_average_latency()
        latency_penalty = 0
        if avg_latency > 0:
            max_latency = sum(task.execution_time for task in self.tasks)
            normalized_latency = min(avg_latency / max_latency, 1)
            latency_penalty = -normalized_latency
  
        throughput = self.get_throughput()
        throughput_reward = throughput / len(self.tasks) if self.tasks else 0
 
        reward = (self.reward_latency_weight * latency_penalty + 
                 self.reward_throughput_weight * throughput_reward + 
                 completed_tasks_reward)
        
        return reward
        
    def _is_episode_done(self) -> bool:
        if all(task.is_completed for task in self.tasks):
            return True
  
        if self.step_count >= self.max_steps:
            return True
            
        return False
        
    def get_average_latency(self) -> float:
        completed_tasks = [task for task in self.tasks if task.is_completed]
        if not completed_tasks:
            return 0.0
            
        total_latency = sum(task.calculate_latency() for task in completed_tasks)
        return total_latency / len(completed_tasks)
        
    def get_throughput(self) -> float:
        if self.current_time == 0:
            return 0.0
            
        return len(self.completed_tasks_history) / self.current_time
        
    def get_cpu_utilization(self) -> float:
        return np.mean([cpu.get_utilization() for cpu in self.cpus])
        
    def get_completed_tasks(self) -> int:
        return len(self.completed_tasks_history)
