import numpy as np

class Kalman_filter:
    def __init__(self,dt=0.05):
        self.dt=dt

        self.state = np.zeros((4,1))

        self.P = np.eye(4)

        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.R = np.array([[0.01,0],[0,0.01]])
        q=0.5
        self.Q= np.array([[dt**3/3,0,dt**2/2,0],[0,dt**3/3,0,dt**2/2],[dt**2/2,0,dt,0],[0,dt**2/2,0,dt]])*q

        self.initialized = False

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def get_position(self):
        return (float(self.state[0,0]),float(self.state[1,0]))

    def update(self,measurement):
        measurement_matrix = np.array([[measurement[0]],[measurement[1]]])
        predicted_measurement = self.H @ self.state
        residual = measurement_matrix - predicted_measurement

        S= self.H @ self.P @ self.H.T + self.R
        K= self.P @ self.H.T @ np.linalg.inv(S)

        self.state= self.state + K @ residual # correct state

        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P  # correct uncertainty

    def initialize(self,measurement):
        self.state =  np.array([[measurement[0]],[measurement[1]],[0],[0]], dtype=float)
        self.initialized = True