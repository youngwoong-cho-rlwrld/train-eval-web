import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Submit",
};

export default function SubmitLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return children;
}
