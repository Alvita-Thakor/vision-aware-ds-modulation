from dataclasses import dataclass,field
from src.entities.position import Position
from src.entities.velocity import Velocity

@dataclass # we are not freezing here as robot cant be immutable
class Robot:
    position:Position = field(default_factory=lambda: Position(0.0,0.0))
    velocity:Velocity = field(default_factory=lambda: Velocity(0.0,0.0))
    orientation: float = 0.0
    radius: float = 0.2