import cv2
import numpy as np

from src.entities.obstacle import Obstacle
from src.entities.position import Position


def pixel_to_world(
    x_pixel,
    y_pixel,
    image_width,
    image_height,
    world_width,
    world_height
):
    x_world = (x_pixel / image_width) * world_width

    # Image y increases downward.
    # Simulator y increases upward.
    y_world = (1 - (y_pixel / image_height)) * world_height

    return x_world, y_world


def pixel_radius_to_world(
    radius_pixel,
    image_width,
    world_width
):
    return (radius_pixel / image_width) * world_width


def add_measurement_noise(
    x_pixel,
    y_pixel,
    noise_std,
    rng
):
    """
    Add Gaussian measurement noise in PIXEL coordinates.

    noise_std:
        Standard deviation of pixel measurement noise.
    """

    noisy_x = x_pixel + rng.normal(0, noise_std)
    noisy_y = y_pixel + rng.normal(0, noise_std)

    return noisy_x, noisy_y


def detect_obstacle(
    image,
    pixel_noise_std=0.0,
    radius_noise_std=0.0,
    rng=None
):
    """
    Detect black obstacles in a white image.

    Returns:
        list[Obstacle]
    """

    if rng is None:
        rng = np.random.default_rng()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    obstacles = []

    image_height, image_width = gray.shape

    for contour in contours:

        M = cv2.moments(contour)

        if M["m00"] == 0:
            continue

        # ---------------------------------------------
        # Centroid in pixel coordinates
        # ---------------------------------------------

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # ---------------------------------------------
        # Add measurement noise in PIXEL space
        # ---------------------------------------------

        cx, cy = add_measurement_noise(
            cx,
            cy,
            pixel_noise_std,
            rng
        )

        # ---------------------------------------------
        # Estimate radius
        # ---------------------------------------------

        (_, _), radius = cv2.minEnclosingCircle(contour)

        radius = radius + rng.normal(0, radius_noise_std)
        radius = max(radius, 0.0)
    
        # ---------------------------------------------
        # Pixel → world conversion
        # ---------------------------------------------

        x_world, y_world = pixel_to_world(
            cx,
            cy,
            image_width,
            image_height,
            10.0,
            10.0
        )

        radius_world = pixel_radius_to_world(
            radius,
            image_width,
            10.0
        )

        obstacle = Obstacle(
            position=Position(
                x_world,
                y_world
            ),
            radius=radius_world
        )
        obstacles.append(obstacle)

    return obstacles