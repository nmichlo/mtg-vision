// Note: Similarity functions
export function dotProduct(a: Float32Array | number[], b: Float32Array | number[]): number {
  let dP = 0.0;
  for (let i = 0; i < a.length; i++) {
    dP += a[i] * b[i];
  }
  return dP;
}

export function cosineSimilarity(a: Float32Array | number[], b: Float32Array | number[]): number {
  let AA = 0.0;
  let BB = 0.0;
  let AB = 0.0;
  for (let i = 0; i < a.length; i++) {
    const A = a[i];
    const B = b[i];
    AA += A * A;
    BB += B * B;
    AB += A * B;
  }
  return AB / (Math.sqrt(AA) * Math.sqrt(BB));
}

function euclideanDistance(a: Float32Array | number[], b: Float32Array | number[]): number {
  let sum = 0.0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return Math.sqrt(sum);
}

export function euclideanSimilarity(a: Float32Array | number[], b: Float32Array | number[]): number {
  return 1 / (1 + euclideanDistance(a, b));
}
