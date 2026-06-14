# Mirror Neuron Training

Mirror Neuron Training is a web-based drawing imitation simulator. You draw a stroke on the left grid, and an AI learner tries to reproduce the same motor pattern on the right grid.

The project is built with Three.js and Vite. It is designed as a visual sandbox for exploring imitation learning, trajectory matching, and reward/evaluation design for drawing behavior.

## What It Does

- Draw on a high-resolution grid with mouse or pointer input.
- Automatically trains when you release the pointer.
- Shows the exact target data on the AI side as an amber overlay.
- Shows the AI's copied stroke in green.
- Visualizes a large swarm of candidate agents using GPU instancing.
- Scores the AI with segment-aware evaluation instead of simple point-by-point matching.
- Lets the AI keep learning across attempts or train more on the current drawing.

## How It Works

The human drawing is captured as a raw stroke path. The app densifies the path while drawing so fast pointer movement still becomes a usable trajectory.

When training starts, the stroke is normalized into panel coordinates and resampled by arc length. That resampled path is the training target. The same target is shown on the AI panel, so the overlay represents the exact data the learner receives.

The current learner is a small browser-side neural network that maps progress along the stroke to an `(x, z)` drawing coordinate:

```text
progress t -> predicted x,z
```

The model is intentionally lightweight so the browser can train and visualize quickly. It is a prototype learner, not a production ML backend.

## Evaluation

The score is based on segment-aware path comparison. The app compares the human target and AI output using:

- Coverage: how much of the human path was copied.
- Precision: how much of the AI path belongs near the human path.
- Curve matching: how well local bends and turns line up.
- Order matching: whether the path sequence is followed.
- Length gate: a harsh multiplier based on total stroke length.

The length gate is intentionally strict:

```text
100% length match -> full score multiplier
99% length match  -> still strong
80% or worse      -> zero multiplier
```

This prevents a short partial copy from getting a high score just because some local curves look correct.

## Swarm Visualization

The app can display thousands of candidate agents efficiently. It uses a single `THREE.InstancedMesh` where each visible trail segment is an instance.

For example:

```text
1,200 agents x 32 segments = 38,400 trail segments
```

Those trail segments are rendered as one instanced mesh instead of thousands of separate Three.js objects.

The current swarm is a visualization layer generated from the same target samples. It is not yet training 1,200 separate neural networks. That would be the next step, likely using Web Workers, WebGPU, or a batched tensor backend.

## Controls

- Draw on the left grid.
- Release to train automatically.
- Keep learning: reuse the current model across attempts.
- Train more: run extra training on the current drawing.
- Reset: clear the drawing and start fresh.

## Development

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

Build for production:

```bash
npm run build
```

## Tech Stack

- Three.js
- Vite
- Browser-side JavaScript neural net prototype
- GPU instancing for high-volume agent visualization

## Project Direction

Useful next steps:

- Move model training into a Web Worker.
- Add true batched multi-agent training.
- Add WebGPU or TensorFlow.js acceleration.
- Support multiple strokes instead of a single continuous stroke.
- Add export/import for drawings and training runs.
- Add richer reward tuning controls.
