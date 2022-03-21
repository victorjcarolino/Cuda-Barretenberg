import { Pedersen } from './pedersen';
import { WorkerPool } from '../../wasm/worker_pool';
import { BarretenbergWasm } from '../../wasm';
// import createDebug from 'debug';

// const debug = createDebug('bb:pooled_pedersen');

/**
 * Multi-threaded implementation of pedersen.
 */
export class PooledPedersen extends Pedersen {
  public pool: Pedersen[] = [];

  /**
   * @param wasm Synchronous functions will use use this wasm directly on the calling thread.
   * @param pool Asynchronous functions use this pool of workers to multi-thread processing.
   */
  constructor(wasm: BarretenbergWasm, pool: WorkerPool) {
    super(wasm);
    this.pool = pool.workers.map(w => new Pedersen(wasm, w));
  }

  public async init() {
    await Promise.all(this.pool.map(p => p.init()));
  }

  public async hashToTree(values: Buffer[]) {
    const isPowerOf2 = (v: number) => v && !(v & (v - 1));
    if (!isPowerOf2(values.length)) {
      throw new Error('PooledPedersen::hashValuesToTree can only handle powers of 2.');
    }

    const workers = this.pool.slice(0, Math.min(values.length / 2, this.pool.length));
    const numPerThread = values.length / workers.length;

    const results = await Promise.all(
      workers.map((pedersen, i) => pedersen.hashToTree(values.slice(i * numPerThread, (i + 1) * numPerThread))),
    );

    const sliced = results.map(hashes => {
      const treeHashes: Buffer[][] = [];
      for (let i = numPerThread, j = 0; i >= 1; j += i, i /= 2) {
        treeHashes.push(hashes.slice(j, j + i));
      }
      return treeHashes;
    });

    const flattened = sliced[0];
    for (let i = 1; i < sliced.length; ++i) {
      for (let j = 0; j < sliced[i].length; ++j) {
        flattened[j] = [...flattened[j], ...sliced[i][j]];
      }
    }

    while (flattened[flattened.length - 1].length > 1) {
      const lastRow = flattened[flattened.length - 1];
      const newRow: Buffer[] = [];
      for (let i = 0; i < lastRow.length; i += 2) {
        newRow[i / 2] = this.pool[0].compress(lastRow[i], lastRow[i + 1]);
      }
      flattened.push(newRow);
    }

    return flattened.flat();
  }
}
