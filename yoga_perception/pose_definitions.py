"""Iyengar yoga pose definitions with alignment rules.

Each pose is defined as a dict with:
    - name: English name
    - sanskrit: Sanskrit name
    - rules: list of alignment rules, each with:
        - check: string key for the measurement to evaluate
        - condition: lambda taking the measurement value, returns True if aligned
        - cue: the Iyengar-style verbal cue if misaligned
        - priority: 1 (critical) to 3 (refinement)
"""

POSES = {}


def register_pose(pose_def):
    """Register a pose definition."""
    POSES[pose_def["name"]] = pose_def
    return pose_def


# ---------------------------------------------------------------------------
# Tadasana (Mountain Pose)
# ---------------------------------------------------------------------------
register_pose({
    "name": "tadasana",
    "sanskrit": "Tadasana",
    "english": "Mountain Pose",
    "required_parts": [
        "torso_front", "left_upper_leg_front", "right_upper_leg_front",
        "left_foot", "right_foot",
    ],
    "rules": [
        {
            "check": "torso_vertical",
            "condition": lambda v: abs(v) < 5,
            "cue": "Stand tall. Align your torso vertically — lift through the crown of your head.",
            "priority": 1,
        },
        {
            "check": "shoulder_level",
            "condition": lambda v: abs(v) < 15,
            "cue": "Level your shoulders. Roll them back and draw the shoulder blades down your back.",
            "priority": 1,
        },
        {
            "check": "hip_level",
            "condition": lambda v: abs(v) < 15,
            "cue": "Level your hips. Distribute your weight evenly on both feet.",
            "priority": 1,
        },
        {
            "check": "left_leg_angle",
            "condition": lambda v: v > 165,
            "cue": "Straighten your left leg. Engage your quadriceps and lift your kneecap.",
            "priority": 2,
        },
        {
            "check": "right_leg_angle",
            "condition": lambda v: v > 165,
            "cue": "Straighten your right leg. Engage your quadriceps and lift your kneecap.",
            "priority": 2,
        },
        {
            "check": "left_arm_vertical",
            "condition": lambda v: abs(v) < 10,
            "cue": "Let your left arm hang naturally alongside your body, fingers pointing down.",
            "priority": 3,
        },
        {
            "check": "right_arm_vertical",
            "condition": lambda v: abs(v) < 10,
            "cue": "Let your right arm hang naturally alongside your body, fingers pointing down.",
            "priority": 3,
        },
    ],
})


# ---------------------------------------------------------------------------
# Trikonasana (Triangle Pose)
# ---------------------------------------------------------------------------
register_pose({
    "name": "trikonasana",
    "sanskrit": "Trikonasana",
    "english": "Triangle Pose",
    "required_parts": [
        "torso_front", "left_upper_arm_front", "right_upper_arm_front",
        "left_upper_leg_front", "right_upper_leg_front",
    ],
    "rules": [
        {
            "check": "shoulder_level",
            "condition": lambda v: True,  # shoulders will NOT be level — that's correct
            "cue": None,
            "priority": 3,
        },
        {
            "check": "left_arm_vertical",
            "condition": lambda v: True,  # evaluated contextually below
            "cue": None,
            "priority": 3,
        },
        {
            "check": "right_arm_vertical",
            "condition": lambda v: True,
            "cue": None,
            "priority": 3,
        },
        {
            "check": "arms_aligned",
            "condition": lambda v: v is True,
            "cue": "Stack your arms — extend both arms in one vertical line from floor to ceiling.",
            "priority": 1,
        },
        {
            "check": "torso_vertical",
            "condition": lambda v: abs(v) > 30,
            "cue": "Extend your torso sideways over the front leg. Do not collapse forward.",
            "priority": 1,
        },
        {
            "check": "front_knee_over_ankle",
            "condition": lambda v: v is True,
            "cue": "Keep your front leg straight. Do not bend the knee.",
            "priority": 1,
        },
        {
            "check": "hip_level",
            "condition": lambda v: abs(v) < 30,
            "cue": "Open your hips. Rotate your top hip back so both hips face forward.",
            "priority": 2,
        },
    ],
})


# ---------------------------------------------------------------------------
# Virabhadrasana II (Warrior II)
# ---------------------------------------------------------------------------
register_pose({
    "name": "virabhadrasana_ii",
    "sanskrit": "Virabhadrasana II",
    "english": "Warrior II",
    "required_parts": [
        "torso_front", "left_upper_arm_front", "right_upper_arm_front",
        "left_upper_leg_front", "right_upper_leg_front",
    ],
    "rules": [
        {
            "check": "torso_vertical",
            "condition": lambda v: abs(v) < 8,
            "cue": "Keep your torso upright and centered. Do not lean toward the front leg.",
            "priority": 1,
        },
        {
            "check": "shoulder_level",
            "condition": lambda v: abs(v) < 15,
            "cue": "Level your shoulders. Draw them down away from your ears.",
            "priority": 1,
        },
        {
            "check": "front_knee_angle",
            "condition": lambda v: 80 <= v <= 100,
            "cue": "Bend your front knee to 90 degrees. Align the knee directly over the ankle.",
            "priority": 1,
        },
        {
            "check": "back_leg_angle",
            "condition": lambda v: v > 165,
            "cue": "Straighten your back leg fully. Press through the outer edge of your back foot.",
            "priority": 1,
        },
        {
            "check": "arms_horizontal",
            "condition": lambda v: v is True,
            "cue": "Extend both arms parallel to the floor. Reach actively through your fingertips.",
            "priority": 2,
        },
        {
            "check": "hip_level",
            "condition": lambda v: abs(v) < 15,
            "cue": "Level your hips. Do not let the front hip drop.",
            "priority": 2,
        },
    ],
})


# ---------------------------------------------------------------------------
# Vrksasana (Tree Pose)
# ---------------------------------------------------------------------------
register_pose({
    "name": "vrksasana",
    "sanskrit": "Vrksasana",
    "english": "Tree Pose",
    "required_parts": [
        "torso_front", "left_upper_leg_front", "right_upper_leg_front",
    ],
    "rules": [
        {
            "check": "torso_vertical",
            "condition": lambda v: abs(v) < 5,
            "cue": "Stand tall. Lengthen your spine upward through the crown of your head.",
            "priority": 1,
        },
        {
            "check": "shoulder_level",
            "condition": lambda v: abs(v) < 12,
            "cue": "Level your shoulders. Do not hike one shoulder up.",
            "priority": 1,
        },
        {
            "check": "hip_level",
            "condition": lambda v: abs(v) < 15,
            "cue": "Keep your hips level. Press your standing hip inward to stay centered.",
            "priority": 1,
        },
        {
            "check": "standing_leg_angle",
            "condition": lambda v: v > 170,
            "cue": "Straighten your standing leg completely. Engage the thigh muscles and lift the kneecap.",
            "priority": 1,
        },
    ],
})


# ---------------------------------------------------------------------------
# Adho Mukha Svanasana (Downward-Facing Dog)
# ---------------------------------------------------------------------------
register_pose({
    "name": "adho_mukha_svanasana",
    "sanskrit": "Adho Mukha Svanasana",
    "english": "Downward-Facing Dog",
    "required_parts": [
        "torso_front", "left_upper_arm_front", "right_upper_arm_front",
        "left_upper_leg_front", "right_upper_leg_front",
        "left_hand", "right_hand", "left_foot", "right_foot",
    ],
    "rules": [
        {
            "check": "left_arm_angle",
            "condition": lambda v: v > 165,
            "cue": "Straighten your left arm fully. Rotate your outer upper arm inward.",
            "priority": 1,
        },
        {
            "check": "right_arm_angle",
            "condition": lambda v: v > 165,
            "cue": "Straighten your right arm fully. Rotate your outer upper arm inward.",
            "priority": 1,
        },
        {
            "check": "left_leg_angle",
            "condition": lambda v: v > 155,
            "cue": "Work toward straightening your left leg. Press your left heel toward the floor.",
            "priority": 2,
        },
        {
            "check": "right_leg_angle",
            "condition": lambda v: v > 155,
            "cue": "Work toward straightening your right leg. Press your right heel toward the floor.",
            "priority": 2,
        },
        {
            "check": "shoulder_level",
            "condition": lambda v: abs(v) < 15,
            "cue": "Even out your shoulders. Distribute weight equally between both hands.",
            "priority": 2,
        },
    ],
})


def get_pose(name):
    """Look up a pose by name."""
    if name not in POSES:
        available = ", ".join(POSES.keys())
        raise ValueError(f"Unknown pose '{name}'. Available: {available}")
    return POSES[name]


def list_poses():
    """Return list of registered pose names."""
    return list(POSES.keys())
