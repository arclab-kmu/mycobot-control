"""The only place in this package where ROS 2 may be imported.

Empty for now. The vendor's ROS 2 nodes live in ~/manipulation/mycobot_ros2
and talk to the same serial port; they are not a dependency of this package.
Whatever grows here must import ``rclpy`` lazily, and nothing outside this
directory may import from it. ``tests/test_core_has_no_ros.py`` enforces that.
"""


def require_ros():
    """Import rclpy or fail with a message that names what to source."""
    try:
        import rclpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "rclpy is not importable. Source a ROS 2 environment first, e.g.\n"
            "    source /opt/ros/humble/setup.bash"
        ) from exc
    return rclpy
