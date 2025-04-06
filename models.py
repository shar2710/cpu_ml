from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time

@dataclass
class Task:
    id: int
    arrival_time: float  
    execution_time: float  
    priority: float  
    memory_requirement: float  
    cache_affinity: List[int]  
    dependencies: List[int]  
    remaining_time: float  
    start_time: Optional[float] = None  
    completion_time: Optional[float] = None  
    assigned_cpu: Optional[int] = None  
    is_completed: bool = False
    is_running: bool = False
    
    def calculate_latency(self) -> float:
        if self.completion_time is None:
            return 0.0
        return self.completion_time - self.arrival_time
    
    def calculate_response_time(self) -> float:
        if self.start_time is None:
            return 0.0
        return self.start_time - self.arrival_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'arrival_time': self.arrival_time,
            'execution_time': self.execution_time,
            'remaining_time': self.remaining_time,
            'priority': self.priority,
            'memory_requirement': self.memory_requirement,
            'start_time': self.start_time,
            'completion_time': self.completion_time,
            'assigned_cpu': self.assigned_cpu,
            'is_completed': self.is_completed,
            'is_running': self.is_running,
            'latency': self.calculate_latency() if self.is_completed else None
        }

@dataclass
class CPUCore:
    id: int
    current_task: Optional[Task] = None
    idle_time: float = 0.0
    busy_time: float = 0.0
    
    def is_idle(self) -> bool:
        return self.current_task is None
    
    def assign_task(self, task: Task, current_time: float) -> None:
        if not self.is_idle():
            raise RuntimeError(f"Cannot assign task to CPU {self.id} as it's already running task {self.current_task.id}")
        
        self.current_task = task
        task.assigned_cpu = self.id
        task.is_running = True
        
        if task.start_time is None:
            task.start_time = current_time
    
    def release_task(self, current_time: float) -> Optional[Task]:
        if self.is_idle():
            return None
            
        task = self.current_task
        task.is_running = False
        task.assigned_cpu = None
        
        if task.remaining_time <= 0:
            task.is_completed = True
            task.completion_time = current_time
        
        self.current_task = None
        return task
    
    def update(self, time_delta: float) -> None:
        if self.is_idle():
            self.idle_time += time_delta
        else:
            self.busy_time += time_delta
            self.current_task.remaining_time -= time_delta
    
    def get_utilization(self) -> float:
        total_time = self.idle_time + self.busy_time
        if total_time == 0:
            return 0.0
        return self.busy_time / total_time
