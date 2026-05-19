"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api, type SubmitResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

type ResumePhase = "train" | "resume" | "eval" | "unknown" | "other";

export function ResumeJobButton({
  cluster,
  jobId,
  phase,
  variant,
  jobName,
  className,
}: {
  cluster: string;
  jobId: string;
  phase?: ResumePhase | null;
  variant?: string | null;
  jobName?: string | null;
  className?: string;
}) {
  const router = useRouter();
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const resume = useMutation({
    mutationFn: () =>
      api<SubmitResponse>(`/api/jobs/${cluster}/${jobId}/resume`, {
        method: "POST",
      }),
    onSuccess: (data) => {
      toast.success(`Submitted resume job ${data.job_id} on ${cluster}`);
      setOpen(false);
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["job", cluster, jobId] });
      qc.invalidateQueries({ queryKey: ["job-details"] });
      router.push(`/jobs/${cluster}/${data.job_id}`);
    },
    onError: (err: Error) => toast.error(`Resume failed: ${err.message}`),
  });

  const normalizedPhase = phase === "resume" ? "train" : phase;
  const phaseLabel = normalizedPhase === "train"
    ? "training"
    : normalizedPhase === "eval"
      ? "evaluation"
      : "job";

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        className={className}
        onClick={() => setOpen(true)}
        disabled={resume.isPending}
      >
        {resume.isPending ? "Resuming..." : "Resume"}
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Resume timed-out job?</DialogTitle>
            <DialogDescription>
              This submits a new {phaseLabel} job on{" "}
              <span className="font-mono">{cluster}</span> from timed-out job{" "}
              <span className="font-mono">{jobId}</span>
              {variant ? (
                <>
                  {" "}for <span className="font-mono">{variant}</span>
                </>
              ) : null}
              .
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
            {jobName && (
              <p>
                Original job name:{" "}
                <span className="font-mono text-xs">{jobName}</span>
              </p>
            )}
            {normalizedPhase === "train" ? (
              <p>
                Training resumes from the latest checkpoint found for this
                variant. New checkpoints and logs will be written by the new
                Slurm job.
              </p>
            ) : normalizedPhase === "eval" ? (
              <p>
                Evaluation resume seeds existing eval results into the staged
                experiment directory, skips runs that already have a
                <span className="font-mono"> results.json</span>, and rewrites
                aggregate result files as remaining runs complete.
              </p>
            ) : (
              <p>
                The backend will recover the original phase and variant before
                submitting the replacement Slurm job.
              </p>
            )}
            <p>
              This can update staged result artifacts for the same variant.
              Continue only if this is the timeout you intend to resume.
            </p>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setOpen(false)}
              disabled={resume.isPending}
            >
              Cancel
            </Button>
            <Button
              onClick={() => resume.mutate()}
              disabled={resume.isPending}
            >
              {resume.isPending ? "Submitting..." : "Submit resume"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
