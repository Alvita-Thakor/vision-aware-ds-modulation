from dataclasses import dataclass,field
from src.entities.position import Position

@dataclass(frozen=True)
class Goal:
    position : Position = field(default_factory=lambda: Position(5.0,5.0))
    