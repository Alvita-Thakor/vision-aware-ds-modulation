from dataclasses import dataclass
from math import hypot

@dataclass(frozen=True)
class Velocity:
    vx:float
    vy:float

    def magnitude(self)->float:
        return hypot(self.vx,self.vy)

    def as_tuple(self)->tuple[float,float]:
        return (self.vx,self.vy)