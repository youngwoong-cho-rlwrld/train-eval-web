"use client";

import { useEffect } from "react";
import { usePathname } from "next/navigation";

export const APP_TITLE = "train-eval-web";

export function formatPageTitle(title: string | null | undefined): string {
  return title ? `${title} · ${APP_TITLE}` : APP_TITLE;
}

export function RouteTitle() {
  const pathname = usePathname();

  useEffect(() => {
    const title = titleForPath(pathname);
    if (title !== null) document.title = formatPageTitle(title);
  }, [pathname]);

  return null;
}

function titleForPath(pathname: string): string | null {
  if (pathname === "/") return "Home";
  if (pathname === "/submit") return "Submit";
  if (pathname === "/jobs") return "Jobs";
  if (pathname === "/monitor") return "GPU monitor";
  if (pathname === "/experiments") return "Experiments";
  if (pathname === "/results") return "Results";
  if (pathname === "/settings") return "Settings";
  if (pathname.startsWith("/jobs/")) return null;
  if (pathname.startsWith("/experiments/")) {
    const name = decodeURIComponent(pathname.split("/").filter(Boolean).at(-1) ?? "");
    return name ? `Experiment - ${name}` : "Experiment";
  }
  return null;
}
