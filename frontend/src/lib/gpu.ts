// Shared GPU-count folds. Partition and MlxpNode share the numeric fields
// (gpu_free, gpu_total, queued_gpus) that the monitor page and the submit
// availability cards repeatedly reduce over, so collapse that fold here.

/** Sum a numeric field across a list of GPU-bearing items. */
export function sumGpu<K extends string>(
  items: ReadonlyArray<Record<K, number>>,
  key: K,
): number {
  return items.reduce((sum, item) => sum + item[key], 0);
}
