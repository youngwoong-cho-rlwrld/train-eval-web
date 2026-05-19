"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { cn } from "@/lib/utils";

const VIEWPORT_GAP = 12;
const TOOLTIP_OFFSET = 8;

export function ImmediateTooltip({
  content,
  children,
  className,
  contentClassName,
  side = "top",
}: {
  content?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  contentClassName?: string;
  side?: "top" | "bottom";
}) {
  const triggerRef = useRef<HTMLSpanElement>(null);
  const tooltipRef = useRef<HTMLSpanElement>(null);
  const [visible, setVisible] = useState(false);

  const updatePosition = useCallback(() => {
    const trigger = triggerRef.current;
    const tooltip = tooltipRef.current;
    if (!trigger || !tooltip) return;

    const triggerRect = trigger.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const tooltipWidth = tooltipRect.width || 0;
    const tooltipHeight = tooltipRect.height || 0;
    const centeredLeft = triggerRect.left + triggerRect.width / 2 - tooltipWidth / 2;
    const left = Math.min(
      Math.max(VIEWPORT_GAP, centeredLeft),
      Math.max(VIEWPORT_GAP, viewportWidth - tooltipWidth - VIEWPORT_GAP),
    );
    const top =
      side === "bottom"
        ? Math.min(
            triggerRect.bottom + TOOLTIP_OFFSET,
            viewportHeight - tooltipHeight - VIEWPORT_GAP,
          )
        : Math.max(
            VIEWPORT_GAP,
            triggerRect.top - tooltipHeight - TOOLTIP_OFFSET,
          );

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }, [side]);

  useLayoutEffect(() => {
    if (visible) updatePosition();
  }, [visible, content, updatePosition]);

  useEffect(() => {
    if (!visible) return;
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [visible, updatePosition]);

  if (!content) return <>{children}</>;
  return (
    <span
      ref={triggerRef}
      className={cn("inline-flex min-w-0", className)}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      onFocus={() => setVisible(true)}
      onBlur={() => setVisible(false)}
    >
      {children}
      {visible && typeof document !== "undefined"
        ? createPortal(
            <span
              ref={tooltipRef}
              role="tooltip"
              className={cn(
                "pointer-events-none fixed z-[1000] w-max max-w-[min(32rem,calc(100vw-2rem))] whitespace-normal rounded-md border border-slate-200 bg-white px-2 py-1 text-xs font-normal leading-snug text-slate-700 shadow-lg [overflow-wrap:anywhere] dark:border-slate-800 dark:bg-slate-950 dark:text-slate-200",
                contentClassName,
              )}
              style={{ left: VIEWPORT_GAP, top: VIEWPORT_GAP }}
            >
              {content}
            </span>,
            document.body,
          )
        : null}
    </span>
  );
}
