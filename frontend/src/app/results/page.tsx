export default function ResultsPage() {
  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <h1 className="text-2xl font-semibold tracking-tight">Results</h1>
      <p className="mt-2 text-slate-600 dark:text-slate-400">
        Eval results viewer — coming in v1.5. Will read{" "}
        <code className="text-xs">results.json</code> from completed eval jobs and render the per-eval-set
        success-rate table.
      </p>
    </div>
  );
}
