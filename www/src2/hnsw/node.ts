export const NULL_KEY = null;

export type Key = string;

export type Vec = Float32Array | number[];

export class Node {
  id: Key;
  level: number;
  vector: Vec;
  neighbors: Key[][];

  constructor(id: Key, vector: Vec, level: number, M: number) {
    this.id = id;
    this.vector = vector;
    this.level = level;
    this.neighbors = Array.from({ length: level + 1 }, () =>
      new Array(M).fill(NULL_KEY),
    );
  }
}
