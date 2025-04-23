import { PriorityQueue } from './pqueue';
import { Node, Vec, Key, NULL_KEY } from './node';
import { dotProduct, cosineSimilarity, euclideanSimilarity } from './similarity';

type Metric = 'cosine' | 'euclidean' | 'dot';
type SimilarityFn = (a: Vec, b: Vec) => number;

export class HNSW {
  metric: Metric; // Metric to use
  similarityFunction: (a: Vec, b: Vec) => number;
  // d: number | null = null; // Dimension of the vectors
  M: number; // Max number of neighbors
  efConstruction: number; // Max number of nodes to visit during construction
  levelMax: number; // Max level of the graph
  entryPointId: Key | typeof NULL_KEY; // Id of the entry point
  nodes: Map<Key, Node>; // Map of nodes
  probs: number[]; // Probabilities for the levels

  constructor(M = 16, efConstruction = 200, d: number | null = null, metric: Metric | SimilarityFn = 'cosine') {
    this.metric = metric as Metric;
    // this.d = d;
    this.M = M;
    this.efConstruction = efConstruction;
    this.entryPointId = NULL_KEY;
    this.nodes = new Map<Key, Node>();
    this.probs = this.set_probs(M, 1 / Math.log(M));
    this.levelMax = this.probs.length - 1;
    // Set the similarity function based on the metric
    // more similar is higher
    // less similar is lower
    if (typeof metric === 'string') {
      this.similarityFunction = this.getMetric(metric as Metric);
    } else {
      this.similarityFunction = metric;
    }
  }

  private getMetric(metric: Metric): (a: Vec, b: Vec) => number {
    // if (metric === 'dot') {
    //   return dotProduct; // same as cosine IFF input vectors are normalized
    if (metric === 'cosine') {
      return cosineSimilarity;
    } else if (metric === 'euclidean') {
      return euclideanSimilarity;
    } else {
      throw new Error('Invalid metric');
    }
  }

  private set_probs(M: number, levelMult: number): number[] {
    let level = 0;
    const probs = [];
    while (true) {
      const prob = Math.exp(-level / levelMult) * (1 - Math.exp(-1 / levelMult));
      if (prob < 1e-9) break;
      probs.push(prob);
      level++;
    }
    return probs;
  }

  private selectLevel(): number {
    let r = Math.random();
    this.probs.forEach((p, i) => {
      if (r < p) {
        return i;
      }
      r -= p;
    });
    return this.probs.length - 1;
  }

  private async addNodeToGraph(node: Node) {
    if (this.entryPointId === NULL_KEY) {
      this.entryPointId = node.id;
      return;
    }

    let currentNode = this.nodes.get(this.entryPointId)!;
    let closestNode = currentNode;

    for (let level = this.levelMax; level >= 0; level--) {
      while (true) {
        let nextNode = null;
        let maxSimilarity = -Infinity;

        for (const neighborId of currentNode.neighbors[level]) {
          if (neighborId === NULL_KEY) break;

          const neighborNode = this.nodes.get(neighborId)!;
          const similarity = this.similarityFunction(node.vector, neighborNode.vector);
          if (similarity > maxSimilarity) {
            maxSimilarity = similarity;
            nextNode = neighborNode;
          }
        }

        if (nextNode && maxSimilarity > this.similarityFunction(node.vector, closestNode.vector)) {
          currentNode = nextNode;
          closestNode = currentNode;
        } else {
          break;
        }
      }
    }

    const closestLevel = Math.min(node.level, closestNode.level);
    for (let level = 0; level <= closestLevel; level++) {
      // Add new neighbor to closestNode's neighbors
      closestNode.neighbors[level] = closestNode.neighbors[level].filter((id) => id !== NULL_KEY);
      closestNode.neighbors[level].push(node.id);
      // If the number of neighbors exceeds M, remove the farthest one
      if (closestNode.neighbors[level].length > this.M) {
        closestNode.neighbors[level].pop();
      }

      // Add new neighbor to node's neighbors
      node.neighbors[level] = node.neighbors[level].filter((id) => id !== NULL_KEY);
      node.neighbors[level].push(closestNode.id);
      // If the number of neighbors exceeds M, remove the farthest one
      if (node.neighbors[level].length > this.M) {
        node.neighbors[level].pop();
      }
    }
  }

  async addPoint(id: Key, vector: Vec) {
    // if (this.d !== null && vector.length !== this.d) {
    //   throw new Error('All vectors must be of the same dimension');
    // }
    // this.d = vector.length;

    const node = new Node(id, vector, this.selectLevel(), this.M);
    this.nodes.set(id, node);
    this.levelMax = Math.max(this.levelMax, node.level);

    await this.addNodeToGraph(node);
  }

  searchKNN(query: Vec, k: number): { id: Key; score: number }[] {
    if (this.entryPointId === NULL_KEY) {
      throw new Error('Index not built yet');
    }

    // Check if there's only one node in the graph
    if (this.nodes.size === 1) {
      const onlyNode = this.nodes.get(this.entryPointId)!;
      const similarity = this.similarityFunction(onlyNode.vector, query);
      return [{ id: this.entryPointId, score: similarity }];
    }

    const result: { id: Key; score: number }[] = [];
    const visited: Set<Key> = new Set<Key>();

    const candidates = new PriorityQueue<Key>((a, b) => {
      const aNode = this.nodes.get(a)!;
      const bNode = this.nodes.get(b)!;
      return this.similarityFunction(query, bNode.vector) - this.similarityFunction(query, aNode.vector);
    });

    candidates.push(this.entryPointId);
    let level = this.levelMax;

    while (!candidates.isEmpty() && result.length < k) {
      const currentId = candidates.pop()!;
      if (visited.has(currentId)) continue;

      visited.add(currentId);

      const currentNode = this.nodes.get(currentId)!;
      const similarity = this.similarityFunction(currentNode.vector, query);

      if (similarity > 0) {
        result.push({ id: currentId, score: similarity });
      }

      if (currentNode.level === 0) {
        continue;
      }

      level = Math.min(level, currentNode.level - 1);

      for (let i = level; i >= 0; i--) {
        const neighbors = currentNode.neighbors[i];
        for (const neighborId of neighbors) {
          if (!visited.has(neighborId)) {
            candidates.push(neighborId);
          }
        }
      }
    }

    // ordered, highest score first, and slice just in case
    const orderedResults = result.sort((a, b) => b.score - a.score);
    return orderedResults.slice(0, k);
  }

  async buildIndex(data: { id: Key; vector: Vec }[]) {
    // Clear existing index
    this.nodes.clear();
    this.levelMax = 0;
    this.entryPointId = NULL_KEY;

    // Add points to the index
    for (const item of data) {
      await this.addPoint(item.id, item.vector);
    }
  }

  toJSON() {
    const entries = Array.from(this.nodes.entries());
    return {
      M: this.M,
      efConstruction: this.efConstruction,
      levelMax: this.levelMax,
      entryPointId: this.entryPointId,
      nodes: entries.map(([id, node]) => {
        return [
          id,
          {
            id: node.id,
            level: node.level,
            vector: Array.from(node.vector),
            neighbors: node.neighbors.map((level) => Array.from(level)),
          },
        ];
      }),
    };
  }

  static fromJSON(json: any): HNSW {
    const hnsw = new HNSW(json.M, json.efConstruction);
    hnsw.levelMax = json.levelMax;
    hnsw.entryPointId = json.entryPointId;
    hnsw.nodes = new Map(
      json.nodes.map(([id, node]: [number, any]) => {
        return [
          id,
          {
            ...node,
            vector: new Float32Array(node.vector),
          },
        ];
      }),
    );
    return hnsw;
  }
}
