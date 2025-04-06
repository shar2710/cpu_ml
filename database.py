from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
import datetime

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    simulations = db.relationship('SimulationConfig', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class SimulationConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    num_cpus = db.Column(db.Integer, default=4)
    num_tasks = db.Column(db.Integer, default=100)
    workload_type = db.Column(db.String(20), default='trading')
    reward_latency_weight = db.Column(db.Float, default=0.7)
    reward_throughput_weight = db.Column(db.Float, default=0.3)
    max_steps = db.Column(db.Integer, default=1000)
    learning_rate = db.Column(db.Float, default=0.001)
    gamma = db.Column(db.Float, default=0.99)
    epsilon = db.Column(db.Float, default=1.0)
    epsilon_min = db.Column(db.Float, default=0.01)
    epsilon_decay = db.Column(db.Float, default=0.995)
    memory_size = db.Column(db.Integer, default=10000)
    batch_size = db.Column(db.Integer, default=64)
    time_quantum = db.Column(db.Integer, default=2)  #for round robin
    results = db.relationship('SimulationResult', backref='config', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<SimulationConfig {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'num_cpus': self.num_cpus,
            'num_tasks': self.num_tasks,
            'workload_type': self.workload_type,
            'reward_latency_weight': self.reward_latency_weight,
            'reward_throughput_weight': self.reward_throughput_weight,
            'max_steps': self.max_steps,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'time_quantum': self.time_quantum
        }

class SimulationResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    config_id = db.Column(db.Integer, db.ForeignKey('simulation_config.id'), nullable=False)
    scheduler_type = db.Column(db.String(20), nullable=False) 
    run_date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    avg_latency = db.Column(db.Float)
    throughput = db.Column(db.Float)
    cpu_utilization = db.Column(db.Float)
    total_reward = db.Column(db.Float)
    completed_tasks = db.Column(db.Integer)
    training_history = db.Column(db.JSON)
    
    def __repr__(self):
        return f'<SimulationResult {self.id} {self.scheduler_type}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'config_id': self.config_id,
            'scheduler_type': self.scheduler_type,
            'run_date': self.run_date.isoformat() if self.run_date else None,
            'avg_latency': self.avg_latency,
            'throughput': self.throughput,
            'cpu_utilization': self.cpu_utilization,
            'total_reward': self.total_reward,
            'completed_tasks': self.completed_tasks,
            'training_history': self.training_history
        }