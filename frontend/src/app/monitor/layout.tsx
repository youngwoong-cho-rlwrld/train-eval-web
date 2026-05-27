import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "GPU monitor",
};

export default function MonitorLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return children;
}
