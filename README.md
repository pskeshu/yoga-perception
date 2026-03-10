# yoga_perception

Yoga alignment analysis powered by DensePose. Inspired by the precision and alignment-focused practice of Iyengar yoga, this tool uses computer vision to detect body positioning and provide verbal alignment cues — much like an attentive yoga teacher would.

## How it works

1. **DensePose inference** maps every visible pixel of the body to a 3D surface model
2. **Pose analysis** computes joint angles, body part orientations, and relative positions
3. **Cue engine** compares measurements against alignment rules and generates Iyengar-style verbal cues

## Supported poses

| Pose | Sanskrit | Key checks |
|------|----------|------------|
| Mountain Pose | Tadasana | Torso vertical, shoulders level, legs straight |
| Triangle Pose | Trikonasana | Arms stacked, torso lateral extension, hips open |
| Warrior II | Virabhadrasana II | Front knee 90°, torso upright, arms horizontal |
| Tree Pose | Vrksasana | Standing leg straight, hips level, torso vertical |
| Downward-Facing Dog | Adho Mukha Svanasana | Arms straight, legs working toward straight |

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### DensePose model

Download the DensePose model files into the `models/` directory:

```bash
# Config
wget https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml -O models/densepose_rcnn_R_50_FPN_s1x.yaml

# Weights
wget https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl -O models/model_final_162be9.pkl
```

## Usage

```bash
python -m yoga_perception.main photo.jpg --pose tadasana
```

Example output:
```
Pose: Mountain Pose (Tadasana)
Alignment Score: 71.4%

Alignment cues:
  *** Stand tall. Align your torso vertically — lift through the crown of your head.
  ** Straighten your left leg. Engage your quadriceps and lift your kneecap.
```

Save annotated image:
```bash
python -m yoga_perception.main photo.jpg --pose virabhadrasana_ii --output result.jpg
```

## Project structure

```
yoga_perception/
├── yoga_perception/
│   ├── __init__.py
│   ├── main.py                 # CLI entry point
│   ├── densepose_inference.py  # DensePose model loading and inference
│   ├── pose_definitions.py     # Iyengar alignment rules per pose
│   └── cue_engine.py           # Compares poses to rules, generates cues
├── models/                     # DensePose model files (not tracked)
├── poses/                      # Reference pose data
├── output/                     # Output images (not tracked)
├── requirements.txt
└── README.md
```

## Adding new poses

Define alignment rules in `yoga_perception/pose_definitions.py`:

```python
register_pose({
    "name": "your_pose",
    "sanskrit": "Sanskrit Name",
    "english": "English Name",
    "required_parts": ["torso_front", ...],
    "rules": [
        {
            "check": "torso_vertical",
            "condition": lambda v: abs(v) < 5,
            "cue": "Your alignment cue here.",
            "priority": 1,  # 1=critical, 2=important, 3=refinement
        },
    ],
})
```
