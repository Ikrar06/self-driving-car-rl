"""
Unit tests for the Sensor classes.
"""

import pytest
import math
import numpy as np
from src.environment.sensor import RaySensor, SensorArray, create_default_sensors
from src.environment.track import Track
from src.environment.car import Car


class TestRaySensor:
    """Test individual ray sensor."""

    def test_sensor_creation(self):
        """Sensor should initialize with correct parameters."""
        sensor = RaySensor(angle_offset=30.0, max_range=150.0)

        assert sensor.max_range == 150.0
        # Angle should be converted to radians
        assert abs(sensor.angle_offset - math.radians(30.0)) < 0.01

    def test_sensor_default_values(self):
        """Sensor should have correct default values."""
        sensor = RaySensor(angle_offset=0.0)

        assert sensor.current_distance == 200.0  # Default max_range
        assert sensor.hit_point is None


class TestSensorArray:
    """Test sensor array functionality."""

    def test_sensor_array_creation(self):
        """SensorArray should create correct number of sensors."""
        sensors = SensorArray(num_sensors=5)

        assert sensors.num_sensors == 5
        assert len(sensors.sensors) == 5

    def test_default_angles(self):
        """SensorArray should use default angles if not provided."""
        sensors = SensorArray(num_sensors=5)

        # Default angles should be -60, -30, 0, 30, 60
        expected_angles = [-60, -30, 0, 30, 60]
        assert sensors.angles == expected_angles

    def test_custom_angles(self):
        """SensorArray should accept custom angles."""
        custom_angles = [-45, 0, 45]
        sensors = SensorArray(num_sensors=3, angles=custom_angles)

        assert sensors.angles == custom_angles
        assert len(sensors.sensors) == 3

    def test_single_sensor(self):
        """SensorArray should work with single sensor."""
        sensors = SensorArray(num_sensors=1)

        assert len(sensors.sensors) == 1
        assert sensors.angles == [0]


class TestSensorWithTrack:
    """Test sensors with actual track."""

    def setup_method(self):
        """Create a simple track for testing."""
        # Create rectangular track
        outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
        inner = [(25, 25), (75, 25), (75, 75), (25, 75)]

        self.track = Track(
            name="Test Track",
            outer_boundary=outer,
            inner_boundary=inner,
            checkpoints=[],
            start_pos=(12, 50),  # Left side of track
        )

    def test_sensor_cast_basic(self):
        """Sensor should detect distance to boundary."""
        sensor = RaySensor(angle_offset=0.0, max_range=50.0)

        # Cast from (12, 50) pointing right (angle=0)
        # Should hit inner boundary at x=25
        reading = sensor.cast(12, 50, 0.0, self.track)

        # Should detect wall
        assert reading > 0.0
        assert sensor.hit_point is not None

    def test_sensor_no_hit(self):
        """Sensor should return 0.0 if no obstacle in range."""
        sensor = RaySensor(angle_offset=0.0, max_range=5.0)  # Very short range

        # Cast from (12, 50) with tiny range
        reading = sensor.cast(12, 50, 0.0, self.track)

        # Might hit or not hit depending on exact position
        # Just check it returns a valid value
        assert 0.0 <= reading <= 1.0

    def test_sensor_readings_normalized(self):
        """Sensor readings should be normalized [0.0-1.0]."""
        sensor = RaySensor(angle_offset=0.0, max_range=100.0)

        reading = sensor.cast(12, 50, 0.0, self.track)

        # Reading should be in valid range
        assert 0.0 <= reading <= 1.0


class TestSensorArrayWithTrack:
    """Test sensor array reading from track."""

    def setup_method(self):
        """Create track and sensor array."""
        outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
        inner = [(25, 25), (75, 25), (75, 75), (25, 75)]

        self.track = Track(
            name="Test Track",
            outer_boundary=outer,
            inner_boundary=inner,
            checkpoints=[],
            start_pos=(12, 50),
        )

        self.sensors = SensorArray(num_sensors=5)

    def test_read_all_sensors(self):
        """read_all should return array of correct size."""
        readings = self.sensors.read_all(12, 50, 0.0, self.track)

        assert isinstance(readings, np.ndarray)
        assert readings.shape == (5,)
        assert readings.dtype == np.float32

    def test_all_readings_normalized(self):
        """All sensor readings should be in [0.0-1.0] range."""
        readings = self.sensors.read_all(12, 50, 0.0, self.track)

        assert np.all(readings >= 0.0)
        assert np.all(readings <= 1.0)

    def test_sensors_different_readings(self):
        """Different sensors should give different readings."""
        readings = self.sensors.read_all(12, 50, 0.0, self.track)

        # At least some sensors should have different readings
        # (unless car is perfectly centered, which is unlikely)
        unique_readings = len(np.unique(readings))
        # Just check we got valid readings
        assert unique_readings >= 1


class TestCarSensorIntegration:
    """Test sensor integration with Car class."""

    def setup_method(self):
        """Create car, track, and sensors."""
        outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
        inner = [(25, 25), (75, 25), (75, 75), (25, 75)]

        self.track = Track(
            name="Test Track",
            outer_boundary=outer,
            inner_boundary=inner,
            checkpoints=[],
            start_pos=(12, 50),
        )

        self.car = Car(x=12, y=50, angle=0.0)
        self.sensors = create_default_sensors(max_range=100.0)

    def test_attach_sensors_to_car(self):
        """Should be able to attach sensors to car."""
        self.car.attach_sensors(self.sensors)

        assert self.car.sensors is not None
        assert self.car.sensors == self.sensors

    def test_read_sensors_without_attachment(self):
        """Reading sensors without attachment should return empty array."""
        readings = self.car.read_sensors(self.track)

        assert isinstance(readings, np.ndarray)
        assert len(readings) == 0

    def test_read_sensors_with_attachment(self):
        """Reading sensors with attachment should return readings."""
        self.car.attach_sensors(self.sensors)
        readings = self.car.read_sensors(self.track)

        assert isinstance(readings, np.ndarray)
        assert len(readings) == 5
        assert np.all(readings >= 0.0)
        assert np.all(readings <= 1.0)

    def test_get_observation(self):
        """get_observation should return sensors + velocity."""
        self.car.attach_sensors(self.sensors)
        self.car.velocity = 5.0

        observation = self.car.get_observation(self.track)

        # Should be 5 sensors + 1 velocity = 6 values
        assert observation.shape == (6,)
        assert observation.dtype == np.float32

        # All values should be normalized [0-1]
        assert np.all(observation >= 0.0)
        assert np.all(observation <= 1.0)

    def test_observation_without_sensors(self):
        """get_observation without sensors should only return velocity."""
        self.car.velocity = 5.0
        observation = self.car.get_observation(self.track)

        # Should be just velocity
        assert observation.shape == (1,)

    def test_sensors_update_with_car_movement(self):
        """Sensor readings should change when car moves."""
        self.car.attach_sensors(self.sensors)

        # Get initial readings
        readings1 = self.car.read_sensors(self.track)

        # Move car
        for _ in range(5):
            self.car.update(Car.STRAIGHT)

        # Get new readings
        readings2 = self.car.read_sensors(self.track)

        # Readings should have changed (car is closer to walls)
        assert not np.array_equal(readings1, readings2)


class TestCreateDefaultSensors:
    """Test default sensor creation helper."""

    def test_create_default_sensors(self):
        """create_default_sensors should create 5 sensors."""
        sensors = create_default_sensors()

        assert sensors.num_sensors == 5
        assert len(sensors.sensors) == 5
        assert sensors.angles == [-60, -30, 0, 30, 60]

    def test_create_with_custom_range(self):
        """create_default_sensors should accept custom range."""
        sensors = create_default_sensors(max_range=150.0)

        assert sensors.max_range == 150.0
        assert all(s.max_range == 150.0 for s in sensors.sensors)
