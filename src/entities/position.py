from dataclasses import dataclass
from math import hypot
#dataclass generates automatically without using constrcutor init/eq/repr
@dataclass(frozen=True)   #postion is now immutable
class Position:
    x:float
    y:float
    #forward reference - we declared postion before we completing it , so this tells python to treat as text for now
    def distance_to(self,other:"Position")->float:
        return hypot(self.x-other.x,self.y-other.y)

    def as_tuple(self) -> tuple[float,float]:
        return (self.x,self.y)