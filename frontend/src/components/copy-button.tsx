"use client";

import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { ImmediateTooltip } from "@/components/immediate-tooltip";
import { copyText } from "@/lib/clipboard";

export function CopyButton({ value, title = "Copy" }: { value: string; title?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <ImmediateTooltip content={title}>
      <button
        aria-label={title}
        onClick={async (e) => {
          e.preventDefault();
          e.stopPropagation();
          try {
            await copyText(value);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          } catch {
            // clipboard unavailable; leave the icon unchanged
          }
        }}
        className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-slate-800"
      >
        {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
    </ImmediateTooltip>
  );
}
