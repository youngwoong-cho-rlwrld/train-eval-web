"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { navRoutes } from "@/lib/nav-routes";

export function Nav() {
  const pathname = usePathname();
  return (
    <aside className="flex h-screen w-56 shrink-0 flex-col border-r border-slate-200 bg-slate-50/40 px-3 py-6 dark:border-slate-800 dark:bg-slate-950/40">
      <Link href="/" className="mb-6 px-3 text-sm font-semibold tracking-tight text-slate-900 dark:text-slate-50">
        train-eval-web
      </Link>
      <nav className="flex flex-col gap-1">
        {navRoutes.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || pathname.startsWith(href + "/");
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
                active
                  ? "bg-slate-200/60 text-slate-900 dark:bg-slate-800 dark:text-slate-50"
                  : "text-slate-600 hover:bg-slate-200/40 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800/40 dark:hover:text-slate-50",
              )}
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
