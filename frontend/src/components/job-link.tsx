import Link from "next/link";
import { cn } from "@/lib/utils";
import { jobDetailHref } from "@/lib/job-links";

// Blue external-style link to a job-detail page. Centralizes the
// jobDetailHref + standard link styling (target=_blank, blue text, hover
// underline) that was hand-rolled per call site (N158).
export function JobLink({
  cluster,
  jobId,
  className,
  title,
  children,
}: {
  cluster: string;
  jobId: string;
  className?: string;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <Link
      href={jobDetailHref(cluster, jobId)!}
      target="_blank"
      rel="noreferrer"
      className={cn("text-blue-600 hover:underline", className)}
      title={title}
    >
      {children}
    </Link>
  );
}
