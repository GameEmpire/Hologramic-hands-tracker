# Ologra — 3D Hand Control

Ologra is an experimental 3D hand-control project that uses a webcam to track hand movements and interact with virtual cubes in real time.

## Features

- Webcam-based hand tracking using MediaPipe
- Pinch gestures to grab and move cubes
- Grab offsets for smoother object manipulation
- One Euro filtering to reduce cursor jitter
- Two-hand interaction to scale and rotate cubes
- Basic physics with throwing, gravity, damping, and bouncing
- Depth calibration
- Floor grid, shadows, hover highlights, and a hand-tracking preview

<img width="598.5" height="472.5" alt="Untitled design" src="https://github.com/user-attachments/assets/0f181682-b714-4a7d-b796-1be5026dd138" />
<img width="598.5" height="4722.5" alt="Untitled design (1)" src="https://github.com/user-attachments/assets/9fa19ef1-8ae4-4f41-ac32-64336df6950a" />


## Controls

| Key | Action |
|---|---|
| C | Calibrate depth |
| R | Reset cubes |
| ESC | Quit |

Pinch your thumb and index finger together to grab a cube. Use two hands on the same cube to scale and rotate it.

## Requirements

- Python 3
- OpenCV
- MediaPipe
- Pygame
- A working webcam

## Installation

Install the dependencies:

```bash
python3 -m pip install opencv-python mediapipe pygame
```

Run the project:

```bash
python3 Hand tracker.py
```

Camera availability and compatibility depend on your hardware and software environment.

## AI Disclosure

Most of the code in this project was generated with AI assistance. The project was tested, and development involved setting up the environment and troubleshooting dependencies.

This repository documents an AI-assisted programming experiment.

## Limitations

The 3D scene is rendered in Pygame using perspective projection. It is a hologram-style interaction demo, not a true optical hologram.

## Project Status

Experimental project. Features have been tested in the development environment.

## License

MIT license. Can be used commercially.
