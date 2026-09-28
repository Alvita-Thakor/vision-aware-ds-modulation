from dataclasses import dataclass
from src.entities.position import Position

@dataclass
class Metrics:
    convergence_time : float
    path_length: float
    final_position : Position
    goal_reached : bool 
