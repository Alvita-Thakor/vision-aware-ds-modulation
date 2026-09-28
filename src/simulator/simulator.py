from dataclasses import dataclass, field
from src.entities.obstacle import Obstacle
from src.entities.goal import Goal
from src.entities.robot import Robot
from src.entities.position import Position
from src.controller.controller import Controller


@dataclass
class Simulator:
    world_width: float = 10.0
    world_height: float = 10.0
    robot: Robot = field(default_factory=lambda: Robot(position=Position(3.0, 5.0)))
    goal: Goal = field(default_factory=lambda: Goal(position=Position(5.0, 5.0)))
    is_running: bool = False
    collision: bool = False
    current_time: float = 0.0
    dt: float = 0.05
    obstacles: list[Obstacle] = field(default_factory=list)        # perceived (noisy) — controller input
    true_obstacles: list[Obstacle] = field(default_factory=list)   # ground truth — collision checks
    controller: Controller = field(default_factory=Controller)
    clearances : list = field(default_factory=list)
    convergence_time : float | None = None

    def step(self):
        if self.is_running:
            self.robot.velocity = self.controller.compute_velocity(
                self.robot.position, self.goal.position, self.obstacles, self.robot.radius
            )
            new_x = self.robot.position.x + self.robot.velocity.vx * self.dt
            new_y = self.robot.position.y + self.robot.velocity.vy * self.dt
            new_position = Position(new_x, new_y)

            for i, obstacle in enumerate(self.true_obstacles):
                distance = Position.distance_to(new_position, obstacle.position)
                self.clearances[i] = min(self.clearances[i],distance - self.robot.radius - obstacle.radius)

            if not self.is_valid_position(new_position):
                if self.check_collision(new_position):
                    self.collision = True
                self.is_running = False
            else:
                self.robot.position = new_position
                if Position.distance_to(self.robot.position, self.goal.position) <= self.controller.goal_tolerance:
                    self.convergence_time = self.current_time
                    self.is_running = False
                self.current_time += self.dt

    def is_valid_position(self, position: Position) -> bool:
        r = self.robot.radius

        if not (r <= position.x <= self.world_width - r and r <= position.y <= self.world_height - r):
            return False

        for obstacle in self.true_obstacles:      # <-- ground truth, not self.obstacles
            distance = Position.distance_to(position, obstacle.position)
            if distance <= obstacle.radius + r:
                return False

        return True

    def start(self):
        self.is_running = True

    def check_collision(self,position:Position) -> bool:
        r = self.robot.radius
        for obstacle in self.true_obstacles:   
            distance = Position.distance_to(position, obstacle.position)
            if distance <= obstacle.radius + r:
                return True
        
        return False