import numpy as np
from environment.cpu_env import CPUEnvironment

class TraditionalScheduler:
    def select_action(self, env):
        raise NotImplementedError("This method should be implemented by subclasses")

class FCFSScheduler(TraditionalScheduler):
    def select_action(self, env):
        idle_cpus = [cpu_id for cpu_id, cpu in enumerate(env.cpus) if cpu.is_idle()]
        if not idle_cpus:
            return env.get_no_op_action()
        
        waiting_tasks = [task for task in env.tasks if not task.is_completed and not task.is_running]
        if not waiting_tasks:
            return env.get_no_op_action()
        
        waiting_tasks.sort(key=lambda task: task.arrival_time)
        
        task_id = waiting_tasks[0].id
        cpu_id = idle_cpus[0]
        
        return env.encode_action(task_id, cpu_id)

class SJFScheduler(TraditionalScheduler):
    def select_action(self, env):
        idle_cpus = [cpu_id for cpu_id, cpu in enumerate(env.cpus) if cpu.is_idle()]
        if not idle_cpus:
            return env.get_no_op_action()
        
        waiting_tasks = [task for task in env.tasks if not task.is_completed and not task.is_running]
        if not waiting_tasks:
            return env.get_no_op_action()
        
        waiting_tasks.sort(key=lambda task: task.remaining_time)
        
        task_id = waiting_tasks[0].id
        cpu_id = idle_cpus[0]
        
        return env.encode_action(task_id, cpu_id)

class RoundRobinScheduler(TraditionalScheduler):
    def __init__(self, time_quantum=2):
        self.time_quantum = time_quantum
        self.task_queue = []
        self.last_scheduled_task = -1
        
    def select_action(self, env):
        idle_cpus = [cpu_id for cpu_id, cpu in enumerate(env.cpus) if cpu.is_idle()]
        if not idle_cpus:
            return env.get_no_op_action()
        
        waiting_tasks = [task for task in env.tasks if not task.is_completed and not task.is_running]
        if not waiting_tasks:
            return env.get_no_op_action()
        
        if not self.task_queue:
            waiting_tasks.sort(key=lambda task: task.arrival_time)
            self.task_queue = [task.id for task in waiting_tasks]
        
        task_id = None
        for _ in range(len(self.task_queue)):
            candidate_id = self.task_queue.pop(0)
            task = next((t for t in env.tasks if t.id == candidate_id), None)
            
            if task and not task.is_completed and not task.is_running:
                task_id = candidate_id
                self.task_queue.append(task_id)
                break
                
        if task_id is None and waiting_tasks:
            task_id = waiting_tasks[0].id
            self.task_queue.append(task_id)
            
        if task_id is None:
            return env.get_no_op_action()
            
        cpu_id = idle_cpus[0]
        
        return env.encode_action(task_id, cpu_id)
