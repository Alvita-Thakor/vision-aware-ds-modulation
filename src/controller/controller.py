from dataclasses import dataclass
from math import hypot, atan2, sin, cos
import numpy as np

from src.entities.position import Position
from src.entities.velocity import Velocity


@dataclass
class Controller:
    k: float = 5.0
    influence_distance: float = 2.0
    goal_tolerance: float = 0.5
    max_speed: float = 2.0
    tangent_bias: float =0.2

    def compute_velocity(
        self,
        robot_position,
        goal_position,
        obstacles,
        robot_radius
    ):
        # --------------------------------------------------
        # 1. Original DS towards the goal
        # --------------------------------------------------
        distance = Position.distance_to(
            robot_position,
            goal_position
        )

        if distance <= self.goal_tolerance:
            return Velocity(0, 0)

        vx = (goal_position.x - robot_position.x) * self.k
        vy = (goal_position.y - robot_position.y) * self.k

        ds_velocity = Velocity(vx, vy)

        # --------------------------------------------------
        # 2. Compute individual modulation for each obstacle
        # --------------------------------------------------
        gammas = []
        modulated_velocities = []

        for obstacle in obstacles:

            obstacle_distance = self.compute_obstacle_distance(
                robot_position,
                obstacle,
                robot_radius
            )

            # Ignore obstacles outside influence region
            if obstacle_distance > self.influence_distance:
                continue

            gamma = self.compute_gamma(
                robot_position,
                obstacle,
                robot_radius
            )

            gammas.append(gamma)

            # Normal / reference direction
            normal = self.compute_normal(
                robot_position,
                obstacle
            )

            # Tangent direction
            tangent = self.compute_tangent(normal)

            # Basis matrix
            E = self.build_basis(
                normal,
                tangent
            )

            # Eigenvalues from Huber formulation
            eigenvalues = self.compute_eigenvalues(gamma)

            # Modulation matrix
            M = self.build_modulation_matrix(
                E,
                eigenvalues
            )

            # Each obstacle acts on the ORIGINAL DS
            modulated_velocity = self.apply_modulation(
                M,
                ds_velocity
            )

            # --------------------------------------------------
            # Practical symmetry breaking for exact head-on case
            # --------------------------------------------------
            tangential_velocity = (
                modulated_velocity.vx * tangent[0]
                + modulated_velocity.vy * tangent[1]
            )

            if abs(tangential_velocity) < 1e-6:
                modulated_velocity = Velocity(
                    modulated_velocity.vx
                    + self.tangent_bias * tangent[0],

                    modulated_velocity.vy
                    + self.tangent_bias * tangent[1]
                )

            modulated_velocities.append(
                modulated_velocity
            )

        # --------------------------------------------------
        # 3. No nearby obstacles
        # --------------------------------------------------
        if not gammas:
            return self.limit_speed(ds_velocity)

        # --------------------------------------------------
        # 4. Calculate Huber obstacle weights
        # --------------------------------------------------
        weights = self.compute_weights(gammas)

        final_vx = sum(w * v.vx for w, v in zip(weights, modulated_velocities))
        final_vy = sum(w * v.vy for w, v in zip(weights, modulated_velocities))
        final_velocity = Velocity(final_vx, final_vy)


        # --------------------------------------------------
        # 8. Apply final speed constraint
        # --------------------------------------------------
        return self.limit_speed(final_velocity)

    # ======================================================
    # OBSTACLE DISTANCE
    # ======================================================

    def compute_obstacle_distance(
        self,
        robot_position,
        obstacle,
        robot_radius
    ):
        center_distance = Position.distance_to(
            robot_position,
            obstacle.position
        )

        effective_radius = (
            obstacle.radius + robot_radius
        )

        return center_distance - effective_radius

    # ======================================================
    # GAMMA
    # ======================================================

    def compute_gamma(
        self,
        robot_position,
        obstacle,
        robot_radius
    ):
        center_distance = Position.distance_to(
            robot_position,
            obstacle.position
        )

        effective_radius = (
            obstacle.radius + robot_radius
        )

        gamma = (
            center_distance / effective_radius
        ) ** 2

        # Keep gamma >= 1 for points on/outside
        # the effective obstacle boundary.
        return max(gamma, 1.0)

    # ======================================================
    # NORMAL / REFERENCE DIRECTION
    # ======================================================

    def compute_normal(
        self,
        robot_position,
        obstacle
    ):
        dx = (
            robot_position.x
            - obstacle.position.x
        )

        dy = (
            robot_position.y
            - obstacle.position.y
        )

        distance = hypot(dx, dy)

        # This should not normally happen because the
        # simulator prevents the robot from entering an obstacle.
        if distance < 1e-9:
            return (1.0, 0.0)

        nx = dx / distance
        ny = dy / distance

        return (nx, ny)

    # ======================================================
    # TANGENT
    # ======================================================

    def compute_tangent(self, normal):
        tx = -normal[1]
        ty = normal[0]

        return (tx, ty)

    # ======================================================
    # BASIS MATRIX
    # ======================================================

    def build_basis(self, normal, tangent):
        return [
            [normal[0], tangent[0]],
            [normal[1], tangent[1]]
        ]

    # ======================================================
    # EIGENVALUES
    # Huber et al.:
    #
    # lambda_r = 1 - 1/Gamma
    # lambda_e = 1 + 1/Gamma
    # ======================================================

    def compute_eigenvalues(self, gamma):
        lambda_r = 1.0 - (1.0 / gamma)
        lambda_e = 1.0 + (1.0 / gamma)

        return (lambda_r, lambda_e)

    # ======================================================
    # MODULATION MATRIX
    # M = E D E^(-1)
    # ======================================================

    def build_modulation_matrix(
        self,
        E,
        eigenvalues
    ):
        E = np.array(E, dtype=float)

        D = np.diag(
            eigenvalues
        )

        E_inverse = np.linalg.inv(E)

        M = E @ D @ E_inverse

        return M

    # ======================================================
    # APPLY MODULATION
    # ======================================================

    def apply_modulation(
        self,
        M,
        velocity
    ):
        v = np.array([
            velocity.vx,
            velocity.vy
        ])

        modulated_v = M @ v

        return Velocity(
            float(modulated_v[0]),
            float(modulated_v[1])
        )

    # ======================================================
    # MULTI-OBSTACLE WEIGHTS
    #
    # Huber Eq. (12):
    #
    # w_o =
    # product_{i != o}(Gamma_i^-1)
    # --------------------------------
    # sum_k product_{i != k}(Gamma_i^-1)
    # ======================================================

    def compute_weights(self, gammas):

        inverse_gammas = [ gamma - 1.0  for gamma in gammas]

        raw_weights = []

        for i in range(len(gammas)):

            product = 1.0

            for j in range(len(gammas)):

                if i != j:
                    product *= inverse_gammas[j]

            raw_weights.append(product)

        total_weight = sum(raw_weights)

        if total_weight == 0:
            return [
                1.0 / len(gammas)
                for _ in gammas
            ]

        return [
            weight / total_weight
            for weight in raw_weights
        ]

    # ======================================================
    # MAX SPEED
    # ======================================================

    def limit_speed(self, velocity):

        speed = hypot(
            velocity.vx,
            velocity.vy
        )

        if speed > self.max_speed:

            vx = (
                self.max_speed
                * velocity.vx
                / speed
            )

            vy = (
                self.max_speed
                * velocity.vy
                / speed
            )

            return Velocity(vx, vy)

        return velocity