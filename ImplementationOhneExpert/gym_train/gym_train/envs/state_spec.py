STATE_VECTOR_FEATURES = [
    "last_action",
    "speed",
    "distance_to_destination",
    "acceleration",
    "time_error",
    "jerk",
    "distance_covered",
    "current_gradient",
    "distance_next_gradient",
    "next_gradient",
]

# 27 segments x 2 columns (value, length) = 54 features
SPEED_LIMIT_FEATURES = [
    f"speedlimit_{i}_{attr}"
    for i in range(27)
    for attr in ["value", "length"]
]

ALL_FEATURES = STATE_VECTOR_FEATURES + SPEED_LIMIT_FEATURES
