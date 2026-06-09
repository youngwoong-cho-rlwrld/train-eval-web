import { Activity, Cpu, FileCog, Send, Settings, Trophy } from "lucide-react";
import type { ComponentType } from "react";

export type NavRoute = {
  href: string;
  icon: ComponentType<{ className?: string }>;
  /** Sidebar label. */
  label: string;
  /** Home landing-tile title. */
  title: string;
  /** Home landing-tile description. */
  desc: string;
};

/** Single source of truth for the app's primary routes, shared by the sidebar
 *  (Nav) and the home landing tiles. */
export const navRoutes: NavRoute[] = [
  { href: "/submit", icon: Send, label: "Submit", title: "Submit a job", desc: "Pick cluster + experiment, send to sbatch" },
  { href: "/jobs", icon: Activity, label: "Jobs", title: "Jobs", desc: "Active and recent jobs across both clusters" },
  { href: "/monitor", icon: Cpu, label: "GPU monitor", title: "GPU monitor", desc: "Cluster GPU availability and node status" },
  { href: "/experiments", icon: FileCog, label: "Experiments", title: "Experiments", desc: "Create / edit experiment configs" },
  { href: "/results", icon: Trophy, label: "Results", title: "Results", desc: "Eval success-rate tables" },
  { href: "/settings", icon: Settings, label: "Settings", title: "Settings", desc: "Configure integrations and local app state" },
];
