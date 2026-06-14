import { expect, test } from "@playwright/test";

test("training persistence, target overlay priority, and replayable run restore", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => {
    localStorage.removeItem("mirror-neuron-training:run:v1");
    localStorage.removeItem("mirror-neuron-training:history:v1");
  });
  await page.goto("/");
  await page.reload();
  await expect(page.locator("#stage")).toBeVisible();

  const layerState = await page.evaluate(() => window.__mntDebug.getState());
  expect(layerState.targetRenderOrder).toBeGreaterThan(layerState.swarmRenderOrder);
  expect(layerState.targetDepthTest).toBe(false);
  expect(layerState.targetDepthWrite).toBe(false);
  expect(layerState.renderFrame).toBeGreaterThan(0);

  const strokeCanvasPath = await page.evaluate(() => {
    const points = [];
    for (let i = 0; i < 16; i += 1) {
      const t = i / 15;
      const leftWorld = window.__mntDebug.leftFromNorm(-0.92 + t * 0.7, -0.8 + t * 1.45);
      const canvasPoint = window.__mntDebug.projectToCanvas(leftWorld.x, leftWorld.y, leftWorld.z);
      points.push({
        x: Math.max(12, Math.min(canvasPoint.x, window.innerWidth - 12)),
        y: Math.max(12, Math.min(canvasPoint.y, window.innerHeight - 12))
      });
    }
    return points;
  });

  await page.mouse.move(strokeCanvasPath[0].x, strokeCanvasPath[0].y);
  await page.mouse.down();
  for (let i = 1; i < strokeCanvasPath.length; i += 1) {
    await page.mouse.move(strokeCanvasPath[i].x, strokeCanvasPath[i].y);
    await page.waitForTimeout(6);
  }
  await page.mouse.up();

  await page.waitForFunction(() => window.__mntDebug.getState().rawStrokeCount > 4, null, {
    timeout: 15000
  });
  await page.waitForFunction(
    () => {
      const state = window.__mntDebug.getState();
      return state.isTraining || state.aiStrokeCount > 0 || state.evaluation !== null;
    },
    null,
    { timeout: 15000 }
  );

  let previousProgress = -1;
  let sawMotion = false;

  for (let i = 0; i < 40; i += 1) {
    const current = await page.evaluate(() => window.__mntDebug.getState());
    if (current.swarmProgress > previousProgress + 0.001) {
      sawMotion = true;
    }
    previousProgress = current.swarmProgress;
    if (!current.isTraining && current.aiStrokeCount > 0) break;
    await page.waitForTimeout(50);
  }

  expect(sawMotion).toBeTruthy();
  const finalTrainingState = await page.waitForFunction(
    () => {
      const state = window.__mntDebug.getState();
      return !state.isTraining && state.aiStrokeCount > 0 && state.swarmProgress >= 1;
    },
    null,
    { timeout: 120000 }
  );
  expect(finalTrainingState).not.toBeNull();
  const completedState = await page.evaluate(() => window.__mntDebug.getState());
  expect(completedState.swarmProgress).toBe(1);
  expect(completedState.bestAiStrokeCount).toBeGreaterThan(0);
  expect(completedState.repeatBest).toBe(true);

  const persisted = await page.evaluate(() => localStorage.getItem("mirror-neuron-training:run:v1"));
  expect(persisted).toBeTruthy();

  const persistedRecord = JSON.parse(persisted);
  expect(persistedRecord.model).toBeTruthy();
  expect(persistedRecord.model.v).toBe(1);
  expect(typeof persistedRecord.model.b).toBe("string");
  expect(persistedRecord.model.b.length).toBeGreaterThan(0);
  expect(persistedRecord.compactModelMeta).toBeTruthy();
  expect(persistedRecord.compactModelMeta.h).toBe(12);
  expect(persistedRecord.rawStroke.length).toBeGreaterThan(4);
  expect(persistedRecord.aiStroke).toBeUndefined();

  await page.reload();
  await page.waitForLoadState("networkidle");
  const restoredState = await page.evaluate(() => window.__mntDebug.getState());
  expect(restoredState.isRestoredRun).toBe(true);
  expect(restoredState.rawStrokeCount).toBeGreaterThan(0);
  expect(restoredState.targetLinePointCount).toBeGreaterThan(0);
  expect(restoredState.sampleCount).toBeGreaterThan(0);
  expect(restoredState.compactModelMeta).toBeTruthy();
  expect(restoredState.compactModelMeta.bytes).toBeGreaterThan(0);

  let replayStarts = 0;
  let previous = false;
  for (let i = 0; i < 90; i += 1) {
    const current = await page.evaluate(() => window.__mntDebug.getState());
    if (current.isReplaying && !previous) replayStarts += 1;
    previous = current.isReplaying;
    if (replayStarts >= 2 && current.bestAiStrokeCount > 0) break;
    await page.waitForTimeout(80);
  }

  expect(replayStarts).toBeGreaterThanOrEqual(2);
});
