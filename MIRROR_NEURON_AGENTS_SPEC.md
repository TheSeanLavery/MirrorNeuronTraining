# Mirror-Neuron Compression Spec (Weights-Only Runtime)

## Summary in plain language

You already have a great real-time multi-agent loop and a neat local persistence path. Right now the saved record still carries a lot of drawing data (even before policy-only replay), so the saved size can grow with longer strokes.  
The goal is: **final rendering should be reproducible from model weights only**, and persistence should store a **minimal compact model payload** instead of large stroke arrays.

## Current status (what exists today)

- Target line is already forced to render above other lines:
  - `targetLine.renderOrder = 120`
  - `targetLine.material.depthTest = false`
- Model now supports compact serialization through:
  - int8 quantization
  - shared global scale
  - shape `{ v, i, h, o, s, b }`
- Training runs now:
  1. train a large teacher network
  2. distill into a compact student (`h=12`)
  3. keep compact model in state and persisted payload
- Restore path:
  - loads compact model
  - rebuilds target overlay from training samples / raw stroke
  - redraws AI from model on load (no separate ai line in storage)

## Spec sheet: tiny metadata-first storage

### Storage key
- `RUN_STORAGE_KEY = "mirror-neuron-training:run:v1"`

### Required persisted fields
1. `rawStroke`: retained for UI replay context
2. `model`: full model payload  
   - `inputSize`
   - `hiddenSize`
   - `outputSize`
   - `w1`: input → hidden weights
   - `b1`: hidden biases
   - `w2`: hidden → output weights
   - `b2`: output biases
3. minimal session metadata (run id, epochs, compact loss, etc.)
4. Optional: support loading legacy compact payloads via `deserializeCompactModel`.

### Prohibited (or optional) persisted large payloads
- Prefer **omit**: `trainingSamples`, `aiStroke`, `trainStroke`
- Keep only if you intentionally need exact historical artifact replay

## Recommended implementation target

1. **Distill hardening**
   - Keep teacher→student output matching as default for every completed draw:
     - teacher: full model
     - student: `StrokeNet(12)`
     - targets: teacher output for same feature `t` inputs
   - Continue to support fallback to teacher model if distill returns empty samples.

2. **Weights-only regeneration on restore**
   - If no saved ai line exists, generate AI line from compact model via feature sweep (`REPLAY_POINTS`).
   - Evaluate with raw stroke only for UI score; no target path needed at restore.

3. **Persistence audit**
- After each run, assert:
  - saved payload has full `model` object and parseable weights
  - no `aiStroke` in record (or explicitly accepted as legacy fallback only)

4. **Target rendering guarantee**
   - Keep target line on top:
     - `targetLine.renderOrder > aiLine.renderOrder` (already satisfied by setting `120`)
     - depth test disabled for target line

## Acceptance criteria

- On full-page reload:
  - restored raw stroke is visible
  - target line renders above all candidates
  - AI line regenerates from compact weights without reading saved `aiStroke`
- Persisted record is compact:
  - `record.model.b.length` remains small for typical strokes
  - no hard dependency on saved target/AI line for final AI generation
- Inference path:
  - runtime final draw uses compact model only

## Suggested follow-up (phase 2)

- Replace random compact model init with explicit deterministic seed for reproducible distill re-runs.
- Add a compactness budget metric in UI (e.g. bytes used by model payload).
- Add a migration path for legacy payloads (`serializeModel`) so older runs still open.
