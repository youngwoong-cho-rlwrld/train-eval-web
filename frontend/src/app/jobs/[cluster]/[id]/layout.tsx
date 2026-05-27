import type { Metadata } from "next";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ cluster: string; id: string }>;
}): Promise<Metadata> {
  const { id } = await params;
  return {
    title: `Job ${id}`,
  };
}

export default function JobDetailLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return children;
}
