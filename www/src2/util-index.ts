/**
 * Vector Index Utilities
 *
 * Provides utilities for loading and searching a vector index for card identification.
 * This file includes:
 * - Functions to load vector index metadata and binary vector data
 * - Vector similarity search capabilities using the HNSW nearest neighbor algorithm
 * - Index state management (loading, error handling)
 * - Quantization and vector transformation for efficient embedding comparison
 *
 * The vector index allows for real-time matching of card embeddings against a database
 * of known card embeddings for identification.
 */

import { HNSW } from "./hnsw/main";
import { Meta } from "./types";
import {
  cosineSimilarity,
  cosineSimilarityPreNorm,
  makeDecode,
  makeDecodeSimilarity,
  makeDecodeSimilarityPreNorm,
  makeLinearForward
} from "./util-tensor";

// Global index state
let index: HNSW;
let indexMeta: Meta;
let indexProcessVec: (vec: number[]) => number[];
let indexLoadState = "N/A";

/**
 * Loads the vector index for card identification
 */
export async function loadIndex() {
  if (index || indexMeta) {
    console.warn("Index already loaded.");
    return;
  }

  console.log("Loading index...");
  indexLoadState = "Fetching Data...";

  const metaFile = "/index/index_meta.json";
  const vecFile = "/index/index_vecs.bin";

  // load metaFile
  console.log("load meta...");
  const metaResponse = await fetch(metaFile);
  const meta: Meta = await metaResponse.json();

  // load index
  console.log("load vecs...");
  const vecsResponse = await fetch(vecFile);
  const vecsBuffer = await vecsResponse.arrayBuffer();
  const vecs = new Uint8Array(vecsBuffer);

  const getVec = (i: number) => {
    const byteStart = i * vecQuantBytes;
    const byteEnd = byteStart + vecQuantBytes;
    const vec = vecs.slice(byteStart, byteEnd);
    return vec;
  };

  // check
  const numVecs = meta.ids.length;
  const vecQuantSize = meta.quantize.out_dim;
  const bytesPerNumQuant = { float32: 4, uint8: 1 }[meta.quantize.out_dtype];
  const vecQuantBytes = vecQuantSize * bytesPerNumQuant;
  const totalBytes = numVecs * vecQuantBytes;

  if (vecs.length !== totalBytes) {
    throw new Error(
      `Index vecs size mismatch: ${vecs.length} != ${totalBytes}`,
    );
  }
  if (meta.quantize.out_dtype !== "uint8") {
    throw new Error(
      `Index quantize out_dtype not yet supported: ${meta.quantize.out_dtype}, expected uint8`,
    );
  }
  if (meta.chain.length !== 1) {
    throw new Error(
      `Index chain length not yet supported: ${meta.chain.length}, expected 1`,
    );
  }

  const decode = makeDecode(meta.quantize);
  const decodeSimilarity = makeDecodeSimilarity(meta.quantize);
  const decodeSimilarityPreNorm = makeDecodeSimilarityPreNorm(meta.quantize);

  const quantizedCosineSimilarity = (a: number[], b: number[]) => {
    const aIsRef = a.length == 1;
    const bIsRef = b.length == 1;
    if (aIsRef && bIsRef) {
      return decodeSimilarity(getVec(a[0]), getVec(b[0]));
    }
    const _a = aIsRef ? decode(getVec(a[0])) : a;
    const _b = bIsRef ? decode(getVec(b[0])) : b;
    return cosineSimilarity(_a, _b);
  };

  const quantizedCosineSimilarityPreNormalized = (a: number[], b: number[]) => {
    const aIsRef = a.length == 1;
    const bIsRef = b.length == 1;
    if (aIsRef && bIsRef) {
      return decodeSimilarityPreNorm(getVec(a[0]), getVec(b[0]));
    }
    const _a = aIsRef ? decode(getVec(a[0])) : a;
    const _b = bIsRef ? decode(getVec(b[0])) : b;
    return cosineSimilarityPreNorm(_a, _b);
  };

  // create index
  indexLoadState = "Creating Index...";
  const idx = new HNSW(16, 100, vecQuantSize, quantizedCosineSimilarity);

  // add vectors to index
  console.log("add vecs...");

  const time = performance.now();
  for (let i = 0; i < meta.ids.length; i++) {
    await idx.addPoint(meta.ids[i], [i]);
  }
  const time2 = performance.now();

  console.log("Loaded index in", time2 - time, "ms");
  meta.ids = null;
  indexMeta = meta;
  index = idx;
  indexProcessVec = makeLinearForward(meta.chain[0]);
  indexLoadState = "Success";

  return index;
}

/**
 * Get the current index state
 */
export function getIndexState() {
  return {
    index,
    indexMeta,
    indexProcessVec,
    indexLoadState
  };
}

/**
 * Search the vector index for matching cards
 */
export async function searchIndex(embedding: number[], topK: number = 1) {
  const { index, indexProcessVec } = getIndexState();

  if (!indexProcessVec || !index) {
    console.warn("Index not loaded for search");
    return null;
  }

  const compressedEmbedding = indexProcessVec(embedding);
  return index.searchKNN(compressedEmbedding, topK);
}
