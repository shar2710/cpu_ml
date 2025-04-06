import numpy as np
import logging
from typing import List, Dict, Any
from models import Task

logger = logging.getLogger(__name__)

class WorkloadGenerator:
    
    def __init__(self, workload_type='trading'):
        self.workload_type = workload_type
        
    def generate_workload(self, num_tasks=100) -> List[Task]:
        if self.workload_type == 'trading':
            return self._generate_trading_workload(num_tasks)
        elif self.workload_type == 'realtime':
            return self._generate_realtime_workload(num_tasks)
        elif self.workload_type == 'mixed':
            return self._generate_mixed_workload(num_tasks)
        else:
            logger.warning(f"Unknown workload type: {self.workload_type}, defaulting to trading")
            return self._generate_trading_workload(num_tasks)
            
    def _generate_trading_workload(self, num_tasks=100) -> List[Task]:
        tasks = []
        arrival_rate = 5.0  
        
        for i in range(num_tasks):
            arrival_time = np.random.exponential(1.0 / arrival_rate) * i
        
            task_type = np.random.choice(['trade', 'analysis'], p=[0.8, 0.2])
            
            if task_type == 'trade':
                execution_time = np.random.uniform(0.1, 2.0)
                priority = np.random.uniform(0.7, 1.0)
                memory_requirement = np.random.uniform(10, 100)
            else:
                execution_time = np.random.uniform(5.0, 15.0)
                priority = np.random.uniform(0.3, 0.6)
                memory_requirement = np.random.uniform(100, 500)
                
            num_cores = 4 
            has_affinity = np.random.choice([True, False], p=[0.3, 0.7])
            if has_affinity:
                cache_affinity = [np.random.randint(0, num_cores)]
            else:
                cache_affinity = []
                
            has_dependencies = np.random.choice([True, False], p=[0.1, 0.9])
            if has_dependencies and i > 0:
                num_dependencies = np.random.randint(1, min(3, i))
                dependencies = np.random.choice(range(i), size=num_dependencies, replace=False).tolist()
            else:
                dependencies = []
                
            task = Task(
                id=i,
                arrival_time=arrival_time,
                execution_time=execution_time,
                remaining_time=execution_time,
                priority=priority,
                memory_requirement=memory_requirement,
                cache_affinity=cache_affinity,
                dependencies=dependencies
            )
            
            tasks.append(task)
            
        tasks.sort(key=lambda x: x.arrival_time)
        
        for i, task in enumerate(tasks):
            task.id = i
            
        return tasks
        
    def _generate_realtime_workload(self, num_tasks=100) -> List[Task]:
        tasks = []
        
        arrival_rate = 10.0  
        
        for i in range(num_tasks):
            arrival_time = i / arrival_rate
            
            task_type = np.random.choice(['periodic', 'aperiodic'], p=[0.7, 0.3])
            
            if task_type == 'periodic':
                execution_time = np.random.uniform(0.5, 3.0)
                priority = np.random.uniform(0.6, 0.9)
                memory_requirement = np.random.uniform(50, 200)
            else:
                execution_time = np.random.uniform(1.0, 10.0)
                priority = np.random.uniform(0.3, 0.8)
                memory_requirement = np.random.uniform(100, 300)
                
            num_cores = 4  
            has_affinity = np.random.choice([True, False], p=[0.5, 0.5])
            if has_affinity:
                cache_affinity = [np.random.randint(0, num_cores)]
            else:
                cache_affinity = []
                
            has_dependencies = np.random.choice([True, False], p=[0.3, 0.7])
            if has_dependencies and i > 0:
                num_dependencies = np.random.randint(1, min(3, i))
                dependencies = np.random.choice(range(i), size=num_dependencies, replace=False).tolist()
            else:
                dependencies = []
                
            task = Task(
                id=i,
                arrival_time=arrival_time,
                execution_time=execution_time,
                remaining_time=execution_time,
                priority=priority,
                memory_requirement=memory_requirement,
                cache_affinity=cache_affinity,
                dependencies=dependencies
            )
            
            tasks.append(task)
      
        tasks.sort(key=lambda x: x.arrival_time)
        
        for i, task in enumerate(tasks):
            task.id = i
            
        return tasks
        
    def _generate_mixed_workload(self, num_tasks=100) -> List[Task]:
        trading_count = int(num_tasks * 0.6)
        realtime_count = num_tasks - trading_count
        
        trading_tasks = self._generate_trading_workload(trading_count)
        realtime_tasks = self._generate_realtime_workload(realtime_count)
    
        mixed_tasks = trading_tasks + realtime_tasks
        mixed_tasks.sort(key=lambda x: x.arrival_time)
     
        for i, task in enumerate(mixed_tasks):
            task.id = i
            
        return mixed_tasks
