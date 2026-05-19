import Link from "next/link";
import { Activity, Cpu, FileCog, Send, Settings, Trophy } from "lucide-react";

export default function Home() {
  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <h1 className="text-2xl font-semibold tracking-tight">train-eval-web</h1>
      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        <Tile href="/submit" icon={Send} title="Submit a job" desc="Pick cluster + variant, send to sbatch" />
        <Tile href="/jobs" icon={Activity} title="Jobs" desc="Active and recent jobs across both clusters" />
        <Tile href="/monitor" icon={Cpu} title="GPU monitor" desc="Cluster GPU availability and node status" />
        <Tile href="/experiments" icon={FileCog} title="Experiments" desc="Create / edit variant configs" />
        <Tile href="/results" icon={Trophy} title="Results" desc="Eval success-rate tables" />
        <Tile href="/settings" icon={Settings} title="Settings" desc="Configure integrations and local app state" />
      </div>
    </div>
  );
}

function Tile({
  href,
  icon: Icon,
  title,
  desc,
}: {
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  desc: string;
}) {
  return (
    <Link
      href={href}
      className="rounded-lg border border-slate-200 bg-white p-5 transition-colors hover:border-slate-300 hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-950 dark:hover:border-slate-700 dark:hover:bg-slate-900"
    >
      <Icon className="h-5 w-5 text-slate-600 dark:text-slate-400" />
      <div className="mt-3 font-medium">{title}</div>
      <div className="mt-1 text-sm text-slate-500 dark:text-slate-400">{desc}</div>
    </Link>
  );
}
