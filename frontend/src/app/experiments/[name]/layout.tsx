import type { Metadata } from "next";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ name: string }>;
}): Promise<Metadata> {
  const { name } = await params;
  return {
    title: `Experiment - ${decodeURIComponent(name)}`,
  };
}

export default function ExperimentDetailLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return children;
}
