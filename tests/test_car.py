"""
Unit tests for the Car class.
"""

import pytest
import math
import numpy as np
from src.environment.car import Car


class TestCarInitialization:
    """Test car initialization and default values."""

    def test_car_creation(self):
        """Car should initialize with correct default values."""
        car = Car(x=100, y=200, angle=0)
        assert car.x == 100
        assert car.y == 200
        assert car.angle == 0
        assert car.velocity == 0
        assert car.alive is True

    def test_car_custom_params(self):
        """Car should accept custom physics parameters."""
        car = Car(
            x=50,
            y=50,
            angle=math.pi / 4,
            max_velocity=15.0,
            acceleration=1.0,
        )
        assert car.max_velocity == 15.0
        assert car.acceleration_rate == 1.0
        assert car.angle == math.pi / 4


class TestCarPhysics:
    """Test car physics and movement."""

    def test_acceleration_straight(self):
        """Car should accelerate when going straight."""
        car = Car(x=100, y=100, angle=0)
        initial_velocity = car.velocity

        car.update(Car.STRAIGHT)

        assert car.velocity > initial_velocity
        # Velocity = acceleration * friction (0.5 * 0.98 = 0.49)
        expected_velocity = car.acceleration_rate * car.friction
        assert abs(car.velocity - expected_velocity) < 0.01

    def test_velocity_increases_over_time(self):
        """Velocity should increase with repeated acceleration."""
        car = Car(x=100, y=100, angle=0)

        for _ in range(10):
            car.update(Car.STRAIGHT)

        assert car.velocity > car.acceleration_rate

    def test_max_velocity_limit(self):
        """Velocity should not exceed max_velocity."""
        car = Car(x=100, y=100, angle=0, max_velocity=5.0, acceleration=1.0)

        # Accelerate way beyond max
        for _ in range(100):
            car.update(Car.STRAIGHT)

        assert car.velocity <= car.max_velocity

    def test_friction_decay(self):
        """Velocity should decay due to friction when not accelerating."""
        car = Car(x=100, y=100, angle=0, friction=0.9, acceleration=0.0)
        car.velocity = 10.0  # Set initial velocity

        car.update(Car.STRAIGHT)

        # Velocity should decrease
        assert car.velocity < 10.0
        assert car.velocity == 10.0 * 0.9

    def test_position_updates_with_movement(self):
        """Position should change based on velocity and angle."""
        car = Car(x=100, y=100, angle=0, acceleration=2.0, friction=1.0)

        initial_x = car.x
        initial_y = car.y

        car.update(Car.STRAIGHT)

        # With angle=0 (pointing right), x should increase
        assert car.x > initial_x
        # y should remain approximately the same (might have tiny floating point difference)
        assert abs(car.y - initial_y) < 0.01


class TestCarSteering:
    """Test car steering mechanics."""

    def test_turn_left(self):
        """Car angle should decrease when turning left."""
        car = Car(x=100, y=100, angle=0)
        initial_angle = car.angle

        car.update(Car.TURN_LEFT)

        assert car.angle < initial_angle
        assert car.angle == initial_angle - car.turn_rate

    def test_turn_right(self):
        """Car angle should increase when turning right."""
        car = Car(x=100, y=100, angle=0)
        initial_angle = car.angle

        car.update(Car.TURN_RIGHT)

        assert car.angle > initial_angle
        assert car.angle == initial_angle + car.turn_rate

    def test_straight_no_angle_change(self):
        """Angle should not change when going straight."""
        car = Car(x=100, y=100, angle=math.pi / 4)
        initial_angle = car.angle

        car.update(Car.STRAIGHT)

        assert car.angle == initial_angle

    def test_continuous_turning(self):
        """Car should continue turning with repeated turn actions."""
        car = Car(x=100, y=100, angle=0, turn_rate=0.1)

        for _ in range(10):
            car.update(Car.TURN_RIGHT)

        expected_angle = 10 * 0.1
        assert abs(car.angle - expected_angle) < 0.01


class TestCarCollision:
    """Test car collision detection helpers."""

    def test_get_corners_returns_four_points(self):
        """get_corners should return 4 corner points."""
        car = Car(x=100, y=100, angle=0)
        corners = car.get_corners()

        assert len(corners) == 4
        assert all(isinstance(corner, tuple) for corner in corners)
        assert all(len(corner) == 2 for corner in corners)

    def test_corners_symmetric_at_zero_angle(self):
        """At angle=0, corners should be symmetric around center."""
        car = Car(x=100, y=100, angle=0, width=40, height=60)
        corners = car.get_corners()

        # Extract x and y coordinates
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]

        # Center should be at (100, 100)
        assert abs(sum(xs) / 4 - 100) < 0.01
        assert abs(sum(ys) / 4 - 100) < 0.01

    def test_corners_rotate_with_car(self):
        """Corners should rotate when car rotates."""
        car1 = Car(x=100, y=100, angle=0)
        car2 = Car(x=100, y=100, angle=math.pi / 2)

        corners1 = car1.get_corners()
        corners2 = car2.get_corners()

        # Corners should be different when car is rotated
        assert corners1 != corners2

    def test_get_front_center(self):
        """get_front_center should return point at front of car."""
        car = Car(x=100, y=100, angle=0, height=50)
        front = car.get_front_center()

        # At angle=0, front should be to the right
        assert front[0] > car.x
        assert abs(front[1] - car.y) < 0.01
        assert abs(front[0] - (car.x + 25)) < 0.01  # height/2 = 25


class TestCarState:
    """Test car state management."""

    def test_reset_restores_initial_state(self):
        """reset() should restore car to initial position."""
        car = Car(x=100, y=200, angle=math.pi / 4)

        # Move the car
        for _ in range(10):
            car.update(Car.TURN_RIGHT)

        # Reset
        car.reset()

        assert car.x == 100
        assert car.y == 200
        assert car.angle == math.pi / 4
        assert car.velocity == 0
        assert car.alive is True

    def test_kill_stops_car(self):
        """kill() should set alive=False and stop movement."""
        car = Car(x=100, y=100, angle=0)
        car.velocity = 5.0

        car.kill()

        assert car.alive is False
        assert car.velocity == 0

    def test_dead_car_doesnt_update(self):
        """Dead car should not update when update() is called."""
        car = Car(x=100, y=100, angle=0)
        car.kill()

        initial_x = car.x
        initial_y = car.y

        car.update(Car.STRAIGHT)

        # Position should not change
        assert car.x == initial_x
        assert car.y == initial_y

    def test_get_state_vector(self):
        """get_state_vector should return correct numpy array."""
        car = Car(x=100, y=200, angle=1.5)
        car.velocity = 3.5

        state = car.get_state_vector()

        assert isinstance(state, np.ndarray)
        assert state.shape == (4,)
        assert state[0] == 100  # x
        assert state[1] == 200  # y
        assert state[2] == 1.5  # angle
        assert state[3] == 3.5  # velocity


class TestCarRepr:
    """Test car string representation."""

    def test_repr(self):
        """__repr__ should return informative string."""
        car = Car(x=100, y=200, angle=0)
        repr_str = repr(car)

        assert "Car" in repr_str
        assert "100" in repr_str
        assert "200" in repr_str
        assert "alive=True" in repr_str
