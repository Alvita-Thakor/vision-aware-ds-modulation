import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from src.entities.position import Position

class Visualizer:
    def __init__(self,simulator):
        self.simulator = simulator
        self.trajectory=[]

    def plot(self):
        fig , ax = plt.subplots()
        ax.set_xlim(0,self.simulator.world_width)
        ax.set_ylim(0,self.simulator.world_height)
        ax.plot(self.simulator.robot.position.x,self.simulator.robot.position.y,"o")
        ax.plot(self.simulator.goal.position.x,self.simulator.goal.position.y,"o")

        for obstacle in self.simulator.obstacles:
            obs=Circle((obstacle.position.x,obstacle.position.y),obstacle.radius)
            ax.add_patch(obs)

        x_values = [point[0] for point in self.trajectory]
        y_values = [point[1] for point in self.trajectory]
        ax.plot(x_values,y_values)
        plt.show()

    def record_position(self):
        self.trajectory.append((self.simulator.robot.position.x,self.simulator.robot.position.y))

    def calculate_path_length(self):
        if len(self.trajectory)< 2:
            return 0
        total_distance=0
        
        for i in range(1, len(self.trajectory)):
            current_coordinate=Position(*self.trajectory[i-1])
            new_coordinate=Position(*self.trajectory[i])
            total_distance+=Position.distance_to(current_coordinate,new_coordinate)
        return total_distance

        