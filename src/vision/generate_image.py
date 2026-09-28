import cv2
import numpy as np
import pandas as pd
import math
from collections import deque
from src.estimation.kalman_filter import Kalman_filter

from src.vision.vision import (
    detect_obstacle,
    pixel_to_world,
    pixel_radius_to_world
)

from src.entities.obstacle import Obstacle
from src.entities.position import Position

from src.simulator.simulator import Simulator
from src.vision.visualize import Visualizer


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================
MODE = "ground_truth"

IMAGE_WIDTH = 500
IMAGE_HEIGHT = 500

WORLD_WIDTH = 10.0
WORLD_HEIGHT = 10.0

TRUE_OBSTACLE_RADIUS_PIXEL = 40

FORCE_TOTAL_OCCLUSION_WINDOW = False

# ------------------------------------------------------------
# Perception noise
# ------------------------------------------------------------

PIXEL_NOISE_STD = 5.0
RADIUS_NOISE_STD= 0.0

# ------------------------------------------------------------
# Blur
#
# 0 = no blur
# Larger values = stronger blur
#
# Gaussian kernel must be odd.
# ------------------------------------------------------------

BLUR_KERNEL_SIZE = 0

# ------------------------------------------------------------
# Occlusion
#
# Percentage of obstacle diameter that is covered.
#
# 0.0 = no occlusion
# 0.25 = 25% of diameter
# 0.50 = 50% of diameter
# ------------------------------------------------------------

OCCLUSION_PERCENTAGE = 0.0

# ------------------------------------------------------------
# Latency
#
# Delay in milliseconds.
# Simulator dt = 0.05 s = 50 ms.
# ------------------------------------------------------------

LATENCY_MS = 0

# ------------------------------------------------------------
# Camera frame rate
#
# Simulator runs at 20 FPS because:
#
# dt = 0.05 s
#
# FPS = 1 / 0.05 = 20
#
# Set:
#
# 20 → every simulation step
# 10 → every 2 steps
# 5  → every 4 steps
# ------------------------------------------------------------

CAMERA_FPS = 20
LIGHTING_FACTOR=0.7
# ------------------------------------------------------------
# Random seed
#
# None → different noise every run
# 42   → reproducible experiment
# ------------------------------------------------------------

RNG_SEED = 10

# ------------------------------------------------------------
# Maximum number of simulation steps
# ------------------------------------------------------------

MAX_STEPS = 200


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def apply_lighting(image, lighting_factor):
    lighting_factor = max(0.0, lighting_factor)

    lit_image = image.astype(np.float32) * lighting_factor
    lit_image = np.clip(lit_image, 0, 255)

    return lit_image.astype(np.uint8)

def apply_blur(image, kernel_size):
    """
    Apply Gaussian blur to the camera image.

    kernel_size = 0 means no blur.
    """

    if kernel_size <= 0:
        return image

    # Gaussian kernel must be odd.
    if kernel_size % 2 == 0:
        raise ValueError(
            "BLUR_KERNEL_SIZE must be an odd number."
        )

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )


def apply_occlusion(image, center_x, center_y, obstacle_radius, occlusion_percentage):
    if occlusion_percentage <= 0:
        return image

    occlusion_percentage = min(occlusion_percentage, 1.0)

    diameter = 2 * obstacle_radius
    occlusion_width = int(diameter * occlusion_percentage)
    
    if occlusion_width <= 0:
        return image

    # Cover the right side of the obstacle.
    x1 = int(center_x)
    x2 = int(center_x + occlusion_width)

    y1 = int(center_y - obstacle_radius)
    y2 = int(center_y + obstacle_radius)

    # Keep rectangle inside the image.
    x1 = max(0, x1)
    x2 = min(image.shape[1] - 1, x2)
    y1 = max(0, y1)
    y2 = min(image.shape[0] - 1, y2)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        -1
    )

    return image


def calculate_latency_frames(latency_ms,dt):

    latency_seconds = latency_ms / 1000.0
    latency_frames = int(round(latency_seconds / dt))

    return max(latency_frames,0)


def calculate_vision_interval(camera_fps,dt):
    if camera_fps <= 0:
        raise ValueError("CAMERA_FPS must be greater than 0.")

    simulator_fps = 1.0 / dt
    interval = int(round(simulator_fps / camera_fps))
    return max(interval,1)


# ============================================================
# RANDOM NUMBER GENERATOR
# ============================================================
def run_experiment(RNG_SEED,mode,PIXEL_NOISE_STD,BLUR_KERNEL_SIZE,OCCLUSION_PERCENTAGE,LATENCY_MS,CAMERA_FPS):
    rng = np.random.default_rng(RNG_SEED)


    # ============================================================
    # SIMULATOR
    # ============================================================

    kf=Kalman_filter(dt=0.05)
    simulator = Simulator()

    visualizer = Visualizer(
        simulator
    )

    simulator.start()


    # ============================================================
    # DERIVED EXPERIMENT PARAMETERS
    # ============================================================

    latency_frames = calculate_latency_frames(
        LATENCY_MS,
        simulator.dt
    )

    vision_interval = calculate_vision_interval(
        CAMERA_FPS,
        simulator.dt
    )

    latency_buffer = deque(
        maxlen=latency_frames + 1
    )

    raw_positions = []
    filtered_positions = []
    filtered_true_positions = []
    true_positions = []
    raw_true_positions = []
    occlusion_true_positions = []
    occlusion_predicted_positions = []


    # ============================================================
    # INITIAL OBSTACLE POSITION
    # ============================================================

    obstacle_x_pixel = 230
    obstacle_y_pixel = 250


    # ============================================================
    # STATE VARIABLES
    # ============================================================

    steps = 0

    last_valid_estimate = None

    last_camera_estimate = []

    estimated_positions = []
    perception_true_positions = []

    robot_collided = False

    # ============================================================
    # MAIN SIMULATION LOOP
    # ============================================================

        
    while (
        simulator.is_running
        and steps < MAX_STEPS
    ):

        # ========================================================
        # 1. MOVE THE TRUE OBSTACLE
        # ========================================================

        obstacle_y_pixel -= 2 


        # ========================================================
        # 2. CREATE CLEAN CAMERA IMAGE
        # ========================================================

        image = np.ones(
            (
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
                3
            ),
            dtype=np.uint8
        ) * 255
        
        if FORCE_TOTAL_OCCLUSION_WINDOW and (15<=steps<20):
            pass
        else:
            cv2.circle(image,(obstacle_x_pixel,obstacle_y_pixel),TRUE_OBSTACLE_RADIUS_PIXEL,(0, 0, 0),-1)
        # ========================================================
        # 3. GROUND TRUTH
        #
        # This is the REAL physical obstacle.
        #
        # It must NEVER receive noise, blur, latency, etc.
        # ========================================================

        true_x_world, true_y_world = pixel_to_world(
            obstacle_x_pixel,
            obstacle_y_pixel,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            WORLD_WIDTH,
            WORLD_HEIGHT
        )

        true_radius_world = pixel_radius_to_world(
            TRUE_OBSTACLE_RADIUS_PIXEL,
            IMAGE_WIDTH,
            WORLD_WIDTH
        )

        true_obstacle = Obstacle(position=Position(true_x_world,true_y_world),radius=true_radius_world)
        simulator.true_obstacles = [true_obstacle]
        if not simulator.clearances:
            simulator.clearances = [float("inf")] * len(simulator.true_obstacles)
        
        if mode == "ground_truth":
            simulator.obstacles = [true_obstacle]

        #true_positions.append((true_obstacle.position.x,true_obstacle.position.y))

        # ========================================================
        # 4. CAMERA FRAME
        #
        # Only update perception when a camera frame arrives.
        # ========================================================

        if mode != "ground_truth" and steps % vision_interval == 0:

            camera_image = image.copy()


            # ----------------------------------------------------
            # Blur
            # ----------------------------------------------------
            camera_image = apply_lighting(camera_image, LIGHTING_FACTOR)

            camera_image = apply_blur(
                camera_image,
                BLUR_KERNEL_SIZE
            )


            # ----------------------------------------------------
            # Occlusion
            # ----------------------------------------------------

            camera_image = apply_occlusion(
                camera_image,
                obstacle_x_pixel,
                obstacle_y_pixel,
                TRUE_OBSTACLE_RADIUS_PIXEL,
                OCCLUSION_PERCENTAGE
            )


            # ----------------------------------------------------
            # OpenCV detection
            # ----------------------------------------------------

            current_obstacles = detect_obstacle(
                camera_image,
                pixel_noise_std=PIXEL_NOISE_STD,
                radius_noise_std=RADIUS_NOISE_STD,
                rng=rng
            )

            true_positions.append((true_obstacle.position.x,true_obstacle.position.y))
            # ----------------------------------------------------
            # Handle failed detection
            # ----------------------------------------------------

            if current_obstacles:
                detected_x=current_obstacles[0].position.x
                detected_y=current_obstacles[0].position.y
                raw_true_positions.append((true_obstacle.position.x,true_obstacle.position.y))
                measurement=(detected_x,detected_y)
                print(
    "STEP:", steps,
    "| DETECTED:",
    (round(detected_x, 3), round(detected_y, 3)),
    "| RADIUS:",
    round(current_obstacles[0].radius, 3),
    "| TRUE:",
    (true_obstacle.position.x, true_obstacle.position.y)
)
                raw_positions.append(measurement)
                
                if mode == "noisy":
                    last_camera_estimate = current_obstacles

                else :
                    if not kf.initialized:
                        kf.initialize(measurement=measurement)
                    
                    else:
                        kf.predict()
                        kf.update(measurement=measurement)
                
                    filtered_position = kf.get_position()
                    filtered_positions.append(filtered_position)
                    filtered_true_positions.append((true_obstacle.position.x, true_obstacle.position.y))
                    filtered_obstacle = Obstacle(Position(filtered_position[0],filtered_position[1]),current_obstacles[0].radius)
                    last_camera_estimate = [filtered_obstacle]

                last_valid_estimate = current_obstacles

            elif last_valid_estimate is not None:

                # handle missing detection 
                if mode == "noisy":
                    last_camera_estimate = last_valid_estimate

                else :
                    kf.predict()
                    predicted_position = kf.get_position()
                    print(
    "       KF:",
    (round(filtered_position[0], 3), round(filtered_position[1], 3)),
    "| KF STATE:",
    (round(kf.state[2, 0], 3), round(kf.state[3, 0], 3))
)

                    occlusion_true_positions.append((true_obstacle.position.x, true_obstacle.position.y))

                    occlusion_predicted_positions.append(predicted_position)

                    #print("Step:", steps,"True:", (true_obstacle.position.x, true_obstacle.position.y),"Predicted:", predicted_position,"vx:",kf.state[2,0],"vy:",kf.state[3,0])
                    predicted_obstacle = Obstacle(Position(predicted_position[0],predicted_position[1]),last_valid_estimate[0].radius)
                    filtered_positions.append(predicted_position)
                    filtered_true_positions.append((true_obstacle.position.x, true_obstacle.position.y))
                    last_camera_estimate = [predicted_obstacle]

            else:

                # Nothing has ever been detected.
                last_camera_estimate = []


        # ========================================================
        # 5. IF CAMERA DID NOT UPDATE, KEEP LAST MEASUREMENT
        # ========================================================

        if last_camera_estimate is None:

            current_obstacles = []

        else:

            current_obstacles = last_camera_estimate


        # ========================================================
        # 6. LATENCY BUFFER
        # ========================================================

        latency_buffer.append(last_camera_estimate)

        delayed_obstacles = latency_buffer[0]


        # ========================================================
        # 7. PERCEIVED OBSTACLE
        #
        # This is what the controller BELIEVES exists.
        #
        # It may contain:
        #
        # - noise
        # - blur effects
        # - occlusion effects
        # - frame-rate delay
        # - latency
        #
        # It is NOT used for collision checking.
        # ========================================================

        if mode == "ground_truth":
            simulator.obstacles = [true_obstacle]

        else:
            simulator.obstacles = delayed_obstacles


        # ========================================================
        # 8. STORE PERCEPTION ESTIMATE
        # ========================================================

        if delayed_obstacles:

            estimated_positions.append((delayed_obstacles[0].position.x,delayed_obstacles[0].position.y))
            perception_true_positions.append((true_obstacle.position.x,true_obstacle.position.y))

        else:
            estimated_positions.append(None)


        # ========================================================
        # 9. SIMULATOR STEP
        #
        # Controller uses simulator.obstacles.
        #
        # Collision checking uses simulator.true_obstacles.
        # ========================================================

        simulator.step()
        true_normal = simulator.controller.compute_normal(
    simulator.robot.position,
    true_obstacle
)

        estimated_normal = None

        if last_camera_estimate:
            estimated_normal = simulator.controller.compute_normal(
        simulator.robot.position,
        last_camera_estimate[0]
    )

        print(
    "NORMALS:",
    "| TRUE:", tuple(round(x, 3) for x in true_normal),
    "| EST:", (
        tuple(round(x, 3) for x in estimated_normal)
        if estimated_normal else None
    )
)
        print(
    "STEP:", steps,
    "| ROBOT:",
    (
        round(simulator.robot.position.x, 3),
        round(simulator.robot.position.y, 3)
    ),
    "| OBSTACLE EST:",
    (
        round(last_camera_estimate[0].position.x, 3),
        round(last_camera_estimate[0].position.y, 3)
    ) if last_camera_estimate else None,
    "| TRUE OBSTACLE:",
    (
        round(true_obstacle.position.x, 3),
        round(true_obstacle.position.y, 3)
    )
)

        # ========================================================
        # 10. CHECK WHETHER SIMULATION STOPPED DUE TO COLLISION
        # ========================================================

        if not simulator.is_running:

            distance_to_goal = Position.distance_to(simulator.robot.position,simulator.goal.position)

            if distance_to_goal > simulator.controller.goal_tolerance:

                robot_collided = True


        # ========================================================
        # 11. RECORD ROBOT TRAJECTORY
        # ========================================================

        visualizer.record_position()

        steps += 1

    raw_errors = []
    filtered_errors = []

    for i in range(len(raw_positions)):
        true_position = Position(*raw_true_positions[i])
        raw_position = Position(*raw_positions[i])

        raw_error = Position.distance_to(true_position, raw_position)
        raw_errors.append(raw_error)

    for i in range(len(filtered_positions)):
        true_position = Position(*filtered_true_positions[i])
        flt_position = Position(*filtered_positions[i])

        filtered_error = Position.distance_to(true_position, flt_position)
        filtered_errors.append(filtered_error)
    if raw_errors:
        raw_rmse = np.sqrt(np.mean(np.array(raw_errors)**2))
        #print("Raw RMSE:", raw_rmse)

    if mode == "kalman" and filtered_errors:
        filtered_rmse = np.sqrt(np.mean(np.array(filtered_errors) ** 2))
        occlusion_errors = []

        for i in range(len(occlusion_predicted_positions)):
            true_position = Position(*occlusion_true_positions[i])
            predicted_position = Position(*occlusion_predicted_positions[i])

            error = Position.distance_to(true_position, predicted_position)
            occlusion_errors.append(error)

        if occlusion_errors:
            occlusion_rmse = np.sqrt(np.mean(np.array(occlusion_errors) ** 2))
        else:
            occlusion_rmse = None
        #print("Filtered RMSE:", filtered_rmse)
        #print("Occlusion Prediction RMSE:", occlusion_rmse)

    # ============================================================
    # VISUALIZATION
    # ============================================================

    #visualizer.plot()


    # ============================================================
    # PATH LENGTH
    # ============================================================

    path_length = visualizer.calculate_path_length()


    # ============================================================
    # GOAL STATUS
    # ============================================================

    distance_to_goal = Position.distance_to(
        simulator.robot.position,
        simulator.goal.position
    )

    goal_reached = (
        distance_to_goal
        <= simulator.controller.goal_tolerance
    )


    # ============================================================
    # PERCEPTION ERROR
    # ============================================================

    total_error = 0.0

    valid_count = 0

    for true_pos, estimated_pos in zip(perception_true_positions,estimated_positions):

        if estimated_pos is None:
            continue

        dx = (true_pos[0]- estimated_pos[0])
        dy = (true_pos[1]- estimated_pos[1])
        error = math.hypot(dx,dy)

        total_error += error
        valid_count += 1


    if valid_count > 0:

        average_perception_error = (
            total_error
            / valid_count
        )

    else:

        average_perception_error = float("nan")

    results = {
        "mode" : mode,
        "pixel_noise_std" : PIXEL_NOISE_STD,
        "blur_kernel_size" : BLUR_KERNEL_SIZE,
        "occlusion_percentage" : OCCLUSION_PERCENTAGE , 
        "latency_ms" : LATENCY_MS,
        "camera_fps" : CAMERA_FPS,
        "rng_seed" : RNG_SEED,
        "goal_reached" : goal_reached,
        "collision" : robot_collided,
        "minimum_clearance" : min(simulator.clearances),
        "convergence_time" : simulator.convergence_time,
        "path_length" : path_length,
    }

    if mode == "ground_truth":
        results["localization_rmse"] = None

    elif mode == "noisy":
        results["localization_rmse"] = float(raw_rmse)

    else:  # kalman
        results["localization_rmse"] = float(filtered_rmse)

    return results


# ============================================================
# RESULTS
# ============================================================

cv2.waitKey(0)

cv2.destroyAllWindows()

def run_multiple_experiments(mode,BLUR_KERNEL_SIZE,PIXEL_NOISE_STD,OCCLUSION_PERCENTAGE,LATENCY_MS,CAMERA_FPS):
    results=[]
    noise_levels = [ 0,5, 10]
    blur_levels = [0,3,5,7,9]
    fps_levels = [2,4,5,10,20]
    latency_levels = [0,50,100,150,200]
    occlusion_levels = [0,0.25,0.5]
    seeds = range(1, 101)
    modes = ["ground_truth", "noisy", "kalman"]

    for latency in latency_levels:
            for Mode in modes:
                for seed in seeds:
                    result = run_experiment(RNG_SEED=seed,mode =  Mode,BLUR_KERNEL_SIZE=BLUR_KERNEL_SIZE,PIXEL_NOISE_STD=PIXEL_NOISE_STD,OCCLUSION_PERCENTAGE=OCCLUSION_PERCENTAGE,LATENCY_MS=latency,CAMERA_FPS=CAMERA_FPS)
                    results.append(result)

    return results

result = run_multiple_experiments(mode = "noisy",BLUR_KERNEL_SIZE=0,PIXEL_NOISE_STD=5,OCCLUSION_PERCENTAGE=0,LATENCY_MS=0,CAMERA_FPS=20)
df = pd.DataFrame(result)
df.to_csv("latency_sweep_new2.csv",index=False)
print(df.shape)
'''
for seed in [2, 3, 4]:

    print("\n" + "=" * 70)
    print(f"SEED = {seed}")
    print("=" * 70)

    for occlusion in [0.10, 0.25, 0.40, 0.50]:

        print(f"\n--- OCCLUSION = {occlusion * 100:.0f}% ---")

        result = run_experiment(
            RNG_SEED=seed,
            mode="kalman",
            PIXEL_NOISE_STD=5,
            BLUR_KERNEL_SIZE=0,
            OCCLUSION_PERCENTAGE=occlusion,
            LATENCY_MS=0,
            CAMERA_FPS=20
        )

        print(
            "RESULT:",
            "collision =", result["collision"],
            "| clearance =", result["minimum_clearance"],
            "| RMSE =", result["localization_rmse"],
            "| path =", result["path_length"]
        )
'''


#df= pd.DataFrame(results)
#df.to_csv("fps_sweep_new2.csv",index=False)

#print(results)