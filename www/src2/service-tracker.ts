/**
 * Object Tracking Service
 *
 * Provides object tracking functionality for the MTG Vision application.
 * This file includes:
 * - ObjectTracker: A wrapper around the Norfair tracking library
 * - TrackedObjectState: Manages state for tracked objects including embeddings and matching
 *
 * The tracking system maintains object identity across video frames, enabling:
 * - Consistent object IDs over time
 * - Tracking of object state (bounding boxes, embeddings)
 * - Card identification persistence
 * - Prioritization for embedding generation
 */

import * as tf from "@tensorflow/tfjs";
import { Point, Tracker, TrackedObject, TrackerOptions } from "./norfair";
import { BBox, ObjectEmbeddingInfo, TrackerConfig } from "./types";

/**
 * Wrapper for Norfair tracker to handle object tracking
 */
export class ObjectTracker {
  private tracker: Tracker;

  constructor(config: TrackerConfig) {
    this.tracker = new Tracker(config);
    console.log("Norfair tracker init");
  }

  /**
   * Updates the tracker with new detections
   * Returns object IDs assigned to each detection
   */
  update(detections: { point: Point }[]): number[] {
    return this.tracker.update(detections.map((d) => d.point));
  }

  /**
   * Gets the current tracked objects
   */
  getTrackedObjects(): TrackedObject[] {
    return this.tracker.trackedObjects;
  }
}

/**
 * Manages state for tracked objects including embeddings and card matching
 */
export class TrackedObjectState {
  private state = new Map<number, ObjectEmbeddingInfo>();

  /**
   * Updates the list of tracked objects based on tracker results
   */
  updateTrackedObjects(
    trackedObjects: TrackedObject[],
    timestamp: number,
  ): void {
    const currentIds = new Set<number>();
    trackedObjects.forEach((obj) => {
      if (obj.id === -1) return;
      currentIds.add(obj.id);
      if (!this.state.has(obj.id)) {
        this.state.set(obj.id, {
          id: obj.id,
          hasBeenEmbedded: false,
          lastEmbeddingTime: null,
          creationTime: timestamp,
          embedding: null,
          lastKnownBboxVideo: null,
          matchId: null,
          lastMatchTime: null,
          cardData: null,
        });
      }
    });
    this.cleanupDeadObjects(currentIds); // Cleanup immediately after update
  }

  /**
   * Removes tracked objects that are no longer active
   */
  cleanupDeadObjects(activeIds: Set<number>): void {
    const deadIds: number[] = [];
    for (const id of this.state.keys()) {
      if (!activeIds.has(id)) {
        deadIds.push(id);
      }
    }
    deadIds.forEach((id) => {
      this.state.get(id)?.embedding?.dispose();
      this.state.delete(id);
    });
  }

  /**
   * Updates the last known bounding box for a tracked object
   */
  updateLastKnownBbox(id: number, bbox: BBox | null): void {
    const info = this.state.get(id);
    if (info) {
      info.lastKnownBboxVideo = bbox;
    }
  }

  /**
   * Selects an object for embedding generation based on priority
   */
  selectObjectForEmbedding(): { id: number; bboxVideo: BBox } | null {
    const candidates = Array.from(this.state.values()).filter(
      (info) => info.lastKnownBboxVideo,
    );
    const unembedded = candidates
      .filter((i) => !i.hasBeenEmbedded)
      .sort((a, b) => a.creationTime - b.creationTime);
    if (unembedded.length) {
      return {
        id: unembedded[0].id,
        bboxVideo: unembedded[0].lastKnownBboxVideo!,
      };
    }
    const embedded = candidates
      .filter((i) => i.hasBeenEmbedded && i.lastEmbeddingTime)
      .sort((a, b) => a.lastEmbeddingTime! - b.lastEmbeddingTime!);
    if (embedded.length) {
      return { id: embedded[0].id, bboxVideo: embedded[0].lastKnownBboxVideo! };
    }
    const fallback = candidates.filter((i) => i.hasBeenEmbedded);
    if (fallback.length) {
      return { id: fallback[0].id, bboxVideo: fallback[0].lastKnownBboxVideo! };
    }
    return null;
  }

  /**
   * Gets object info by ID
   */
  get(id: number): ObjectEmbeddingInfo | null {
    return this.state.get(id) || null;
  }

  /**
   * Gets embedding tensor for an object
   */
  getEmbedding(id: number): tf.Tensor | null {
    return this.state.get(id)?.embedding || null;
  }

  /**
   * Updates the embedding for a tracked object
   */
  updateEmbedding(id: number, embedding: tf.Tensor | null): void {
    const info = this.state.get(id);
    if (info) {
      info.embedding?.dispose();
      info.embedding = embedding;
      if (embedding) {
        info.hasBeenEmbedded = true;
        info.lastEmbeddingTime = Date.now();
      }
    } else {
      console.warn(`Update embed for non-existent ID ${id}`);
      embedding?.dispose();
    }
  }

  /**
   * Updates the matching card ID for an object
   */
  updateMatchingId(uid: string) {
    const info = this.state.get(parseInt(uid));
    if (info) {
      info.matchId = uid;
      info.lastMatchTime = Date.now();
    } else {
      console.warn(`Update match ID for non-existent ID ${uid}`);
    }
  }

  /**
   * Disposes all embeddings and clears state
   */
  disposeAllEmbeddings(): void {
    this.state.forEach((info) => info.embedding?.dispose());
    this.state.clear();
  }

  /**
   * Updates card data for a tracked object
   */
  updateCardData(id: number, cardData: ObjectEmbeddingInfo['cardData'] | null): void {
    const info = this.state.get(id);
    if (info) {
      info.cardData = cardData;
    }
  }
}
