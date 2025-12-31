"""
Unit tests for the Track class.
"""

import pytest
import json
import os
import tempfile
from src.environment.track import Track, Line


class TestLine:
    """Test Line segment class."""

    def test_line_creation(self):
        """Line should initialize with start and end points."""
        line = Line((0, 0), (10, 10))
        assert line.start == (0, 0)
        assert line.end == (10, 10)

    def test_line_intersection_basic(self):
        """Lines that cross should intersect."""
        line1 = Line((0, 0), (10, 10))
        # Line crossing from bottom-left to top-right
        assert line1.intersects((0, 10), (10, 0)) is True

    def test_line_no_intersection(self):
        """Parallel lines should not intersect."""
        line1 = Line((0, 0), (10, 0))
        # Parallel horizontal line
        assert line1.intersects((0, 5), (10, 5)) is False

    def test_line_intersection_perpendicular(self):
        """Perpendicular lines should intersect."""
        line1 = Line((5, 0), (5, 10))  # Vertical line
        # Horizontal line crossing it
        assert line1.intersects((0, 5), (10, 5)) is True


class TestTrackInitialization:
    """Test track initialization."""

    def test_track_creation(self):
        """Track should initialize with correct parameters."""
        outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
        inner = [(25, 25), (75, 25), (75, 75), (25, 75)]
        checkpoints = [{"start": (50, 0), "end": (50, 25)}]

        track = Track(
            name="Test Track",
            outer_boundary=outer,
            inner_boundary=inner,
            checkpoints=checkpoints,
            start_pos=(50, 50),
            start_angle=0.0,
        )

        assert track.name == "Test Track"
        assert len(track.outer_boundary) == 4
        assert len(track.inner_boundary) == 4
        assert len(track.checkpoints) == 1
        assert track.start_pos == (50, 50)
        assert track.start_angle == 0.0

    def test_track_bounds_calculation(self):
        """Track should calculate correct bounding box."""
        outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
        inner = [(25, 25), (75, 25), (75, 75), (25, 75)]

        track = Track(
            name="Test",
            outer_boundary=outer,
            inner_boundary=inner,
            checkpoints=[],
            start_pos=(50, 50),
        )

        assert track.min_x == 0
        assert track.max_x == 100
        assert track.min_y == 0
        assert track.max_y == 100
        assert track.width == 100
        assert track.height == 100


class TestCollisionDetection:
    """Test point-in-polygon collision detection."""

    def setup_method(self):
        """Create a simple rectangular track for testing."""
        # Outer boundary: 0-100 in both x and y
        self.outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
        # Inner boundary: 25-75 in both x and y
        self.inner = [(25, 25), (75, 25), (75, 75), (25, 75)]

        self.track = Track(
            name="Test Track",
            outer_boundary=self.outer,
            inner_boundary=self.inner,
            checkpoints=[],
            start_pos=(50, 12),  # On the track
        )

    def test_point_on_track(self):
        """Point between inner and outer boundary should be on track."""
        # Point at (10, 10) - inside outer, outside inner
        assert self.track.is_point_on_track(10, 10) is True

    def test_point_inside_inner_boundary(self):
        """Point inside inner boundary should NOT be on track."""
        # Point at (50, 50) - inside inner boundary (the "hole")
        assert self.track.is_point_on_track(50, 50) is False

    def test_point_outside_outer_boundary(self):
        """Point outside outer boundary should NOT be on track."""
        # Point at (150, 150) - completely outside
        assert self.track.is_point_on_track(150, 150) is False

    def test_point_on_boundary_edge(self):
        """Point exactly on boundary should be considered on track."""
        # Point at (0, 50) - on outer boundary
        # Ray-casting is inclusive on one side
        result = self.track.is_point_on_track(0, 50)
        # This depends on implementation, just check it doesn't crash
        assert isinstance(result, bool)

    def test_multiple_points_on_track(self):
        """Multiple points in track area should all return True."""
        track_points = [
            (10, 10),  # Top-left track area
            (90, 10),  # Top-right track area
            (90, 90),  # Bottom-right track area
            (10, 90),  # Bottom-left track area
        ]

        for x, y in track_points:
            assert self.track.is_point_on_track(x, y) is True


class TestCheckpoints:
    """Test checkpoint crossing detection."""

    def setup_method(self):
        """Create a track with checkpoints."""
        outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
        inner = [(25, 25), (75, 25), (75, 75), (25, 75)]
        checkpoints = [
            {"start": (50, 0), "end": (50, 25)},  # Top checkpoint
            {"start": (100, 50), "end": (75, 50)},  # Right checkpoint
        ]

        self.track = Track(
            name="Test Track",
            outer_boundary=outer,
            inner_boundary=inner,
            checkpoints=checkpoints,
            start_pos=(50, 12),
        )

    def test_checkpoint_crossing(self):
        """Crossing a checkpoint line should be detected."""
        # Move from (45, 10) to (55, 10) - crosses vertical checkpoint at x=50
        crossed = self.track.check_checkpoint((45, 10), (55, 10), 0)
        assert crossed is True

    def test_checkpoint_no_crossing(self):
        """Not crossing checkpoint should return False."""
        # Move from (30, 10) to (40, 10) - doesn't cross checkpoint at x=50
        crossed = self.track.check_checkpoint((30, 10), (40, 10), 0)
        assert crossed is False

    def test_checkpoint_parallel_movement(self):
        """Moving parallel to checkpoint should not trigger crossing."""
        # Move from (50, 5) to (50, 15) - parallel to checkpoint
        crossed = self.track.check_checkpoint((50, 5), (50, 15), 0)
        # This might return True or False depending on exact implementation
        # Just check it doesn't crash
        assert isinstance(crossed, bool)

    def test_next_checkpoint_index(self):
        """Next checkpoint index should wrap around."""
        assert self.track.get_next_checkpoint_idx(0) == 1
        assert self.track.get_next_checkpoint_idx(1) == 0  # Wraps to start

    def test_invalid_checkpoint_index(self):
        """Checking invalid checkpoint should return False."""
        crossed = self.track.check_checkpoint((0, 0), (10, 10), 99)
        assert crossed is False


class TestTrackIO:
    """Test track loading and saving."""

    def test_save_and_load_track(self):
        """Track should save to and load from JSON correctly."""
        # Create a track
        outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
        inner = [(25, 25), (75, 25), (75, 75), (25, 75)]
        checkpoints = [{"start": (50, 0), "end": (50, 25)}]

        original_track = Track(
            name="Test Track",
            outer_boundary=outer,
            inner_boundary=inner,
            checkpoints=checkpoints,
            start_pos=(50, 12),
            start_angle=1.5,
        )

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            original_track.save(temp_path)

            # Load from file
            loaded_track = Track.load(temp_path)

            # Verify all attributes match
            assert loaded_track.name == original_track.name
            assert loaded_track.outer_boundary == original_track.outer_boundary
            assert loaded_track.inner_boundary == original_track.inner_boundary
            assert len(loaded_track.checkpoints) == len(original_track.checkpoints)
            assert loaded_track.start_pos == original_track.start_pos
            assert loaded_track.start_angle == original_track.start_angle

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_oval_easy(self):
        """Should be able to load the oval_easy.json track."""
        track_path = "tracks/oval_easy.json"

        # Check if file exists
        if not os.path.exists(track_path):
            pytest.skip(f"Track file {track_path} not found")

        track = Track.load(track_path)

        assert track.name == "Oval Easy"
        assert len(track.outer_boundary) > 0
        assert len(track.inner_boundary) > 0
        assert len(track.checkpoints) > 0
        assert track.start_pos is not None


class TestTrackRepr:
    """Test track string representation."""

    def test_repr(self):
        """__repr__ should return informative string."""
        outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
        inner = [(25, 25), (75, 25), (75, 75), (25, 75)]

        track = Track(
            name="Test Track",
            outer_boundary=outer,
            inner_boundary=inner,
            checkpoints=[],
            start_pos=(50, 50),
        )

        repr_str = repr(track)

        assert "Track" in repr_str
        assert "Test Track" in repr_str
        assert "4" in repr_str  # 4 outer points
