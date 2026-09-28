from dataclasses import dataclass,field
from src.entities.position import Position
@dataclass
class Obstacle:
    position:Position = field(default_factory=lambda : Position(1.0,1.0))
    radius: float = 1.0