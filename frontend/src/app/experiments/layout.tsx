import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Experiments",
};

export default function ExperimentsLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return children;
}
