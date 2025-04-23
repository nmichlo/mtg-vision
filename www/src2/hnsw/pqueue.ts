import FastPriorityQueue from 'fastpriorityqueue';

// PriorityQueue (Push & Pop) x 27.21 ops/sec ±1.65% (47 runs sampled)
// @datastructures-js (Enqueue & Dequeue) x 831 ops/sec ±0.31% (91 runs sampled)
// js-priority-queue (Queue & Dequeue) x 1,347 ops/sec ±0.44% (88 runs sampled)
// fastpriorityqueue (Add & Poll) x 1,783 ops/sec ±0.67% (85 runs sampled)

class PriorityQueueNaive<T> {
  private items: T[] = [];

  constructor(private compare: (a: T, b: T) => number) {}

  push(item: T) {
    let i = 0;
    while (i < this.items.length && this.compare(item, this.items[i]) > 0) {
      i++;
    }
    this.items.splice(i, 0, item);
  }

  pop(): T | undefined {
    return this.items.shift();
  }

  isEmpty(): boolean {
    return this.items.length === 0;
  }
}

export class PriorityQueue<T> {
  private queue: FastPriorityQueue<T>;

  constructor(compare: (a: T, b: T) => number) {
    const cmpBool = (a: T, b: T) => compare(a, b) < 0;
    this.queue = new FastPriorityQueue(cmpBool);
  }

  push(item: T) {
    this.queue.add(item);
  }

  pop(): T | undefined {
    return this.queue.poll();
  }

  isEmpty(): boolean {
    return this.queue.isEmpty();
  }
}
