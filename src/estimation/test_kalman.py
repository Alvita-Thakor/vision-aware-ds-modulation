from src.estimation.kalman_filter import Kalman_filter
import numpy as np
import matplotlib.pyplot as plt

kf = Kalman_filter(dt=0.05)
measurements = []

raw_positions=[]
filtered_positions=[]
actual_positions=[]

for _ in range(30):
    x = 5 +np.random.normal(0,0.5)
    y= 5 + np.random.normal(0,0.5)
    measurements.append((x,y))

for each in measurements:
    raw_positions.append(each)
    kf.predict()
    kf.update(each)
    print(each)
    print("Filtered",kf.get_position())
    filtered_positions.append(kf.get_position())
    actual_positions.append((5,5))

raw_x=[position[0] for position in raw_positions]
raw_y=[position[1] for position in raw_positions]
filtered_x=[position[0] for position in filtered_positions]
filtered_y=[position[1] for position in filtered_positions]
actual_x=[position[0] for position in actual_positions]
actual_y=[position[1] for position in actual_positions]

plt.scatter(raw_x,raw_y,label="raw")
plt.plot(filtered_x,filtered_y,label="filtered")
plt.scatter(actual_x,actual_y,label="actual")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()