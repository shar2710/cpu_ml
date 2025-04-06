import numpy as np
import pandas as pd
import random
from collections import deque

class Process:
    def __init__(self, pid, arrival_time, burst_time, priority=0):
        self.pid = pid  
        self.arrival_time = arrival_time  
        self.burst_time = burst_time  
        self.remaining_time = burst_time  
        self.priority = priority  
        self.completion_time = 0  
        self.waiting_time = 0
        self.turnaround_time = 0

    def __repr__(self):
        return f"Process({self.pid}, AT={self.arrival_time}, BT={self.burst_time}, P={self.priority})"

def generate_processes(n=5):
    processes = []
    for i in range(n):
        arrival_time = random.randint(0, 10)  
        burst_time = random.randint(1, 10)  
        priority = random.randint(1, 5)  
        processes.append(Process(i, arrival_time, burst_time, priority))
    return sorted(processes, key=lambda p: p.arrival_time)  

def fcfs_scheduling(processes):
    time = 0
    for process in processes:
        if time < process.arrival_time:
            time = process.arrival_time
        process.completion_time = time + process.burst_time
        process.turnaround_time = process.completion_time - process.arrival_time
        process.waiting_time = process.turnaround_time - process.burst_time
        time = process.completion_time
    return processes

def sjf_scheduling(processes):
    time = 0
    completed = []
    ready_queue = []
    
    while len(completed) < len(processes):
        ready_queue += [p for p in processes if p.arrival_time <= time and p not in completed]
        ready_queue.sort(key=lambda p: p.burst_time)  
        if ready_queue:
            process = ready_queue.pop(0)
            process.completion_time = time + process.burst_time
            process.turnaround_time = process.completion_time - process.arrival_time
            process.waiting_time = process.turnaround_time - process.burst_time
            time = process.completion_time
            completed.append(process)
        else:
            time += 1  
    return completed

def round_robin_scheduling(processes, quantum=2):
    time = 0
    queue = deque(sorted(processes, key=lambda p: p.arrival_time))
    completed = []

    while queue:
        process = queue.popleft()
        if time < process.arrival_time:
            time = process.arrival_time
        execution_time = min(process.remaining_time, quantum)
        process.remaining_time -= execution_time
        time += execution_time

        if process.remaining_time == 0:
            process.completion_time = time
            process.turnaround_time = process.completion_time - process.arrival_time
            process.waiting_time = process.turnaround_time - process.burst_time
            completed.append(process)
        else:
            queue.append(process)  
    return completed


def print_results(processes, algorithm_name):
    print(f"\n--- {algorithm_name} Scheduling Results ---")
    print("PID | AT  | BT  | CT  | TAT | WT")
    for p in processes:
        print(f"{p.pid:3} | {p.arrival_time:2}  | {p.burst_time:2}  | {p.completion_time:2}  | {p.turnaround_time:2}  | {p.waiting_time:2}")

if __name__ == "__main__":
    processes = generate_processes(5)
    print_results(fcfs_scheduling(processes.copy()), "FCFS")
    print_results(sjf_scheduling(processes.copy()), "SJF")
    print_results(round_robin_scheduling(processes.copy(), quantum=2), "Round Robin")
