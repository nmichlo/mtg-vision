// --- Single File Port of tracker-pp ---

// --- Basic Matrix/Vector Utilities ---
// Note: This is a minimal implementation for the specific matrices used.
// A proper matrix library would be more robust.

export type Vector = number[];
export type Matrix = number[][];
export type Point = [number, number]; // Equivalent to C++ Point (1x2 Matrix treated as coords)

function createMatrix(rows: number, cols: number, fill: number = 0): Matrix {
    return Array(rows).fill(0).map(() => Array(cols).fill(fill));
}

function createVector(size: number, fill: number = 0): Vector {
    return Array(size).fill(fill);
}

function transpose(matrix: Matrix): Matrix {
    if (matrix.length === 0) return [];
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result = createMatrix(cols, rows);
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

function multiplyMatrices(A: Matrix, B: Matrix): Matrix {
    const rowsA = A.length;
    const colsA = A[0]?.length ?? 0;
    const rowsB = B.length;
    const colsB = B[0]?.length ?? 0;

    if (colsA !== rowsB) {
        throw new Error(`Matrix multiplication dimension mismatch: ${colsA} !== ${rowsB}`);
    }

    const C = createMatrix(rowsA, colsB);
    for (let i = 0; i < rowsA; i++) {
        for (let j = 0; j < colsB; j++) {
            for (let k = 0; k < colsA; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

function multiplyMatrixVector(A: Matrix, x: Vector): Vector {
    const rowsA = A.length;
    const colsA = A[0]?.length ?? 0;

    if (colsA !== x.length) {
        throw new Error(`Matrix-vector multiplication dimension mismatch: ${colsA} !== ${x.length}`);
    }

    const y = createVector(rowsA);
    for (let i = 0; i < rowsA; i++) {
        for (let j = 0; j < colsA; j++) {
            y[i] += A[i][j] * x[j];
        }
    }
    return y;
}

function addMatrices(A: Matrix, B: Matrix): Matrix {
    const rows = A.length;
    const cols = A[0]?.length ?? 0;
    if (rows !== B.length || cols !== B[0]?.length) {
        throw new Error("Matrix addition dimension mismatch.");
    }
    const C = createMatrix(rows, cols);
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

function subtractMatrices(A: Matrix, B: Matrix): Matrix {
     const rows = A.length;
    const cols = A[0]?.length ?? 0;
    if (rows !== B.length || cols !== B[0]?.length) {
        throw new Error("Matrix subtraction dimension mismatch.");
    }
    const C = createMatrix(rows, cols);
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

function addVectors(a: Vector, b: Vector): Vector {
     if (a.length !== b.length) {
        throw new Error("Vector addition dimension mismatch.");
    }
    return a.map((val, i) => val + b[i]);
}

function subtractVectors(a: Vector, b: Vector): Vector {
    if (a.length !== b.length) {
        throw new Error("Vector subtraction dimension mismatch.");
    }
    return a.map((val, i) => val - b[i]);
}

function matrixIdentity(size: number): Matrix {
    const I = createMatrix(size, size);
    for (let i = 0; i < size; i++) {
        I[i][i] = 1;
    }
    return I;
}

// Simple 2x2 matrix inverse (specific for Kalman S matrix)
function inverse2x2(matrix: Matrix): Matrix {
    if (matrix.length !== 2 || matrix[0].length !== 2) {
        throw new Error("Inverse calculation only implemented for 2x2 matrices.");
    }
    const [[a, b], [c, d]] = matrix;
    const det = a * d - b * c;
    if (det === 0) {
        throw new Error("Matrix is singular and cannot be inverted.");
    }
    const invDet = 1 / det;
    return [
        [d * invDet, -b * invDet],
        [-c * invDet, a * invDet]
    ];
}

// --- Kalman Filter ---
// Based on kalman.h

// Constants
const _R_KF = 4.0;
const _Q_KF = 0.1;
const _P_KF = 10.0;
const _dt_KF = 1.0;

const F_KF: Matrix = [ // Transition Matrix (4x4)
    [1, 0, _dt_KF, 0],
    [0, 1, 0, _dt_KF],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
];

const H_KF: Matrix = [ // Measurement Matrix (2x4)
    [1, 0, 0, 0],
    [0, 1, 0, 0]
];

const Q_KF: Matrix = [ // Process Uncertainty (4x4)
    [0, 0, 0, 0], // Assuming Q affects velocity only
    [0, 0, 0, 0],
    [0, 0, _Q_KF, 0],
    [0, 0, 0, _Q_KF]
];

const R_KF: Matrix = [ // Measurement Uncertainty (2x2)
    [_R_KF, 0],
    [0, _R_KF]
];

const I_KF: Matrix = matrixIdentity(4); // Identity Matrix (4x4)
const H_KF_T = transpose(H_KF); // Transpose of H
const F_KF_T = transpose(F_KF); // Transpose of F

class KalmanFilter {
    // State vector [x, y, vx, vy]
    public x: Vector;
    // Estimate uncertainty covariance (4x4)
    public P: Matrix;
    // Kalman Gain (4x2)
    private K: Matrix;
    // Measurement residual (2x1)
    private y: Vector;
    // Measurement prediction covariance (2x2)
    private S: Matrix;

    constructor(initialDetection: Point) {
        this.x = createVector(4); // Initialize [0, 0, 0, 0]
        this.P = matrixIdentity(4); // Initialize P as Identity
        this.K = createMatrix(4, 2);
        this.y = createVector(2);
        this.S = createMatrix(2, 2);

        // Set initial position variance
        this.P[0][0] = _P_KF; // Variance for x (use P_KF for initial uncertainty)
        this.P[1][1] = _P_KF; // Variance for y
        this.P[2][2] = _P_KF; // Variance for vx
        this.P[3][3] = _P_KF; // Variance for vy

        // Set initial state from detection
        this.x[0] = initialDetection[0];
        this.x[1] = initialDetection[1];
        // Initial velocity is 0
    }

    predict(): void {
        // x = F * x
        this.x = multiplyMatrixVector(F_KF, this.x);
        // P = F * P * F^T + Q
        const F_P = multiplyMatrices(F_KF, this.P);
        const F_P_FT = multiplyMatrices(F_P, F_KF_T);
        this.P = addMatrices(F_P_FT, Q_KF);
    }

    update(z: Point): void {
        // y = z' - H * x (measurement residual)
        const H_x = multiplyMatrixVector(H_KF, this.x);
        const zVec: Vector = [z[0], z[1]]; // Convert Point to Vector
        this.y = subtractVectors(zVec, H_x);

        // S = H * P * H^T + R
        const P_HT = multiplyMatrices(this.P, H_KF_T);
        const H_P_HT = multiplyMatrices(H_KF, P_HT);
        this.S = addMatrices(H_P_HT, R_KF);

        // K = P * H^T * S^-1
        const S_inv = inverse2x2(this.S);
        this.K = multiplyMatrices(P_HT, S_inv);

        // x = x + K * y
        const K_y = multiplyMatrixVector(this.K, this.y);
        this.x = addVectors(this.x, K_y);

        // P = (I - K * H) * P
        const K_H = multiplyMatrices(this.K, H_KF);
        const I_KH = subtractMatrices(I_KF, K_H);
        this.P = multiplyMatrices(I_KH, this.P);

        // Joseph form for P update (more stable but complex):
        // P = (I - K * H) * P * (I - K * H)^T + K * R * K^T
        // const I_KH_T = transpose(I_KH);
        // const P_stable = multiplyMatrices(multiplyMatrices(I_KH, this.P), I_KH_T);
        // const KRKT = multiplyMatrices(multiplyMatrices(this.K, R_KF), transpose(this.K));
        // this.P = addMatrices(P_stable, KRKT);
    }
}

// --- Tracked Object ---
// Based on tracked_object.h

export class TrackedObject {
    public hitInertiaMin: number;
    public hitInertiaMax: number;
    public initDelay: number;
    public initialHitCount: number;
    public hitCounter: number;
    public id: number; // -1 if initializing
    public filter: KalmanFilter;

    private isInitializingFlag: boolean;
    private detectedAtLeastOnce: boolean;

    constructor(
        initialDetection: Point,
        hitInertiaMin: number,
        hitInertiaMax: number,
        initDelay: number,
        initialHitCount: number,
        period: number
    ) {
        this.hitInertiaMin = hitInertiaMin;
        this.hitInertiaMax = hitInertiaMax;
        this.initDelay = initDelay;
        this.initialHitCount = initialHitCount;
        // Start hit counter above min + period boost
        this.hitCounter = this.hitInertiaMin + period;
        this.id = -1; // Start as initializing
        this.filter = new KalmanFilter(initialDetection);
        this.isInitializingFlag = true;
        this.detectedAtLeastOnce = false;
    }

    trackerStep(): void {
        this.hitCounter -= 1;
        this.filter.predict();
    }

    isInitializing(): boolean {
        // Check if we should transition from initializing to initialized
        if (this.isInitializingFlag && this.hitCounter > this.hitInertiaMin + this.initDelay) {
            this.isInitializingFlag = false;
            // Reset hit counter to a stable value upon initialization
            this.hitCounter = this.initialHitCount;
        }
        return this.isInitializingFlag;
    }

    // Check if the object should still be kept alive
    hasInertia(): boolean {
        return this.hitCounter >= this.hitInertiaMin;
    }

    // Get the estimated position [x, y]
    estimate(): Point {
        // Extract position (first 2 elements) from state vector
        return [this.filter.x[0], this.filter.x[1]];
    }

    hit(detection: Point, period: number = 1): void {
        // Increase hit counter, capped at max
        if (this.hitCounter < this.hitInertiaMax) {
            // Boost by 2 * period for a hit
            this.hitCounter = Math.min(this.hitCounter + 2 * period, this.hitInertiaMax);
        }

        // Update the Kalman filter with the new detection
        this.filter.update(detection);

        // If this is the first *real* detection, reset velocity estimate to 0
        // to prevent jumpy estimates if initial detection was noisy
        if (!this.detectedAtLeastOnce) {
            this.detectedAtLeastOnce = true;
            this.filter.x[2] = 0; // Reset vx
            this.filter.x[3] = 0; // Reset vy
        }
    }
}

// --- Detection Struct ---
// Based on tracker.h

export interface Detection {
    point: Point; // The detected coordinates [x, y]
    id: number;   // The original index of the detection in the input array
}


// --- Tracker Class ---
// Based on tracker.h

export interface TrackerOptions {
    distanceThreshold: number;
    hitInertiaMin?: number;
    hitInertiaMax?: number;
    initDelay?: number; // If -1, calculated from inertia min/max
    initialHitCount?: number; // If -1, calculated from inertia max
}

export class Tracker {
    private trackedObjects: TrackedObject[] = [];
    private distThreshold: number;
    private hitInertiaMin: number;
    private hitInertiaMax: number;
    private nextID: number = 0;
    private initDelay: number;
    private initialHitCount: number;
    private period: number = 1; // Store last period for new object creation

    constructor(options: TrackerOptions) {
        this.distThreshold = options.distanceThreshold;
        this.hitInertiaMin = options.hitInertiaMin ?? 10;
        this.hitInertiaMax = options.hitInertiaMax ?? 25;

        if (this.hitInertiaMin >= this.hitInertiaMax) {
            throw new Error("hitInertiaMin must be less than hitInertiaMax");
        }

        // Calculate default initDelay if not provided
        if (options.initDelay !== undefined && options.initDelay >= 0) {
            this.initDelay = options.initDelay;
        } else {
            this.initDelay = Math.floor((this.hitInertiaMax - this.hitInertiaMin) / 2);
        }

         // Calculate default initialHitCount if not provided
        if (options.initialHitCount !== undefined && options.initialHitCount >= 0) {
            this.initialHitCount = options.initialHitCount;
        } else {
             // Default to mid-point or max inertia? C++ uses max/2
             this.initialHitCount = Math.floor(this.hitInertiaMax / 2);
        }

        if (this.initDelay < 0) {
             console.warn("Calculated initDelay is negative, setting to 0.");
             this.initDelay = 0;
        }
         if (this.initialHitCount <= this.hitInertiaMin + this.initDelay) {
              console.warn(`initialHitCount (${this.initialHitCount}) should ideally be > hitInertiaMin (${this.hitInertiaMin}) + initDelay (${this.initDelay}) to avoid immediate initialization.`);
         }
    }

    /**
     * Updates tracker state with new detections.
     * @param detections Array of points [[x1, y1], [x2, y2], ...].
     * @param period Number of frames since last update.
     * @returns Array of numbers, same length as input detections.
     * Each number is the assigned TrackedObject ID, or -1 if no match/new object.
     */
    update(detections: Point[], period: number = 1): number[] {
        this.period = period;

        // 1. Create Detection objects with original indices
        let currentDetections: Detection[] = detections.map((p, i) => ({ point: p, id: i }));

        // 2. Remove dead tracked objects (those that lost inertia)
        this.trackedObjects = this.trackedObjects.filter(obj => obj.hasInertia());

        // 3. Predict step for all remaining tracked objects
        this.trackedObjects.forEach(obj => obj.trackerStep());

        // 4. Separate objects into initializing and initialized groups
        const initializingObjs: TrackedObject[] = [];
        const initializedObjs: TrackedObject[] = [];
        this.trackedObjects.forEach(obj => {
            if (obj.isInitializing()) {
                initializingObjs.push(obj);
            } else {
                initializedObjs.push(obj);
            }
        });

        // 5. Match detections to *initialized* objects
        //    `updateObjectInPlace` modifies `currentDetections` by removing matched ones.
        const matchedPairsInitialized = this.updateObjectInPlace(initializedObjs, currentDetections);

        // 6. Match remaining detections to *initializing* objects
        const matchedPairsInitializing = this.updateObjectInPlace(initializingObjs, currentDetections);

        // 7. Create new TrackedObjects for any remaining unmatched detections
        currentDetections.forEach(det => {
            this.trackedObjects.push(
                new TrackedObject(
                    det.point,
                    this.hitInertiaMin,
                    this.hitInertiaMax,
                    this.initDelay,
                    this.initialHitCount,
                    this.period // Use stored period
                )
            );
        });

        // 8. Assign IDs to newly initialized objects
        this.trackedObjects.forEach(obj => {
            // If it's not initializing anymore AND doesn't have an ID yet
            if (!obj.isInitializing() && obj.id === -1) {
                obj.id = this.nextID++;
            }
        });

        // 9. Prepare the result map (detection index -> object ID)
        const resultMap = new Map<number, number>();
        // Add matches from initialized objects
        for (const detIdStr in matchedPairsInitialized) {
            resultMap.set(parseInt(detIdStr, 10), matchedPairsInitialized[detIdStr]);
        }
        // Add matches from initializing objects (these will have ID -1)
        for (const detIdStr in matchedPairsInitializing) {
             // Even if matched to an initializing object, the result map should
             // reflect that the object isn't "officially" tracked yet.
             // However, the C++ code assigns the initializing object's ID (-1).
            resultMap.set(parseInt(detIdStr, 10), matchedPairsInitializing[detIdStr]);
        }


        // Create the final result array based on the original detection order
        const finalResult: number[] = detections.map((_, i) => resultMap.get(i) ?? -1);

        return finalResult;
    }

    /**
     * Matches detections to a list of objects (mutates detections list).
     * @param objects List of TrackedObjects (either initializing or initialized).
     * @param detections List of Detections (will be modified).
     * @returns A map where keys are original detection IDs and values are assigned object IDs.
     */
    private updateObjectInPlace(
        objects: TrackedObject[],
        detections: Detection[] // Modified in place
    ): { [key: number]: number } { // Returns map: { detection_original_id: object_id }

        const numDets = detections.length;
        const numObjs = objects.length;
        const matchedPairs: { [key: number]: number } = {};

        if (numDets === 0 || numObjs === 0) {
            return matchedPairs; // No detections or no objects to match
        }

        // Calculate all pairwise distances (Euclidean)
        const distances: { dist: number, cIdx: number, oIdx: number }[] = [];
        for (let c = 0; c < numDets; c++) {
            for (let o = 0; o < numObjs; o++) {
                const detPoint = detections[c].point;
                const objEst = objects[o].estimate();
                const dx = detPoint[0] - objEst[0];
                const dy = detPoint[1] - objEst[1];
                const dist = Math.sqrt(dx * dx + dy * dy);
                distances.push({ dist, cIdx: c, oIdx: o });
            }
        }

        // Sort by distance ascending
        distances.sort((a, b) => a.dist - b.dist);

        const matchedDetIndices = new Set<number>();
        const matchedObjIndices = new Set<number>();

        // Greedy matching
        for (const match of distances) {
            if (match.dist > this.distThreshold) {
                break; // Distances are too large now
            }

            // Check if detection or object is already matched
            if (!matchedDetIndices.has(match.cIdx) && !matchedObjIndices.has(match.oIdx)) {
                // Match found!
                matchedDetIndices.add(match.cIdx);
                matchedObjIndices.add(match.oIdx);

                // Update the matched object
                objects[match.oIdx].hit(detections[match.cIdx].point, this.period);

                // Store the mapping (Original Detection ID -> Object ID)
                matchedPairs[detections[match.cIdx].id] = objects[match.oIdx].id;
            }
        }

        // Remove matched detections from the input list (iterate backwards for safe splice)
        for (let c = numDets - 1; c >= 0; c--) {
            if (matchedDetIndices.has(c)) {
                detections.splice(c, 1);
            }
        }

        return matchedPairs;
    }
}


// --- Example Usage (Conceptual) ---
/*
const tracker = new Tracker({ distanceThreshold: 50 });

// Example frame detections (replace with actual data)
const frame1Detections: Point[] = [[100, 100], [300, 300]];
const frame2Detections: Point[] = [[105, 105], [350, 350]]; // obj1 moved, obj2 disappeared, new obj3 appears

const results1 = tracker.update(frame1Detections);
console.log("Frame 1 Results:", results1); // Likely [-1, -1] or [0, 1] after init delay

const results2 = tracker.update(frame2Detections);
console.log("Frame 2 Results:", results2); // Should show mapping for obj1, maybe -1 for the new one

// Access tracked objects (e.g., after initialization)
tracker.trackedObjects.forEach(obj => {
    if (obj.id !== -1) {
        console.log(`Object ID: ${obj.id}, Estimate: ${obj.estimate()}`);
    }
});
*/
