from datetime import datetime
from src.entities.obstacle import Obstacle
from src.entities.goal import  Goal
from src.entities.robot import Robot
from src.entities.position import Position
from src.entities.velocity import Velocity
from src.simulator.simulator import Simulator
from src.vision.visualize import Visualizer
from src.controller.controller import Controller
from src.metrics.metrics import Metrics
from math import hypot

def main() -> None:
    print(f"Started at : {datetime.now()}")
    robot=Robot(position=Position(1,4.08),velocity=Velocity(1,0))
    goal=Goal(position=Position(5,4))
    obstacle1=Obstacle(position=Position(3,4))
    obstacle2=Obstacle(position=Position(4,3))
    controller=Controller()

    sim=Simulator(robot=robot,goal=goal,obstacles=[obstacle1,obstacle2],controller=controller)
    sim.start()
    graph=Visualizer(simulator=sim)
    for _ in range(200):
        if sim.is_running==False:
            break
        sim.step()
        graph.record_position()
        speed = hypot(sim.robot.velocity.vx, sim.robot.velocity.vy)
        print(speed)


    goal_reached = (Position.distance_to(sim.robot.position, sim.goal.position)<= controller.goal_tolerance)
    metric=Metrics(convergence_time=sim.current_time,path_length=graph.calculate_path_length(),final_position=sim.robot.position,goal_reached=goal_reached)

    graph.plot()
    print(metric)
    


if __name__ == "__main__":
    main()