"use client";

import { useEffect, useRef, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function RefreshButton({
  isFetching,
  onRefresh,
  intervalMs,
}: {
  isFetching: boolean;
  onRefresh: () => void;
  intervalMs?: number;
}) {
  const totalSec = intervalMs ? Math.round(intervalMs / 1000) : 0;
  const [left, setLeft] = useState(totalSec);
  const wasFetching = useRef(isFetching);

  // Reset the countdown when a fetch ends (the auto-refresh just fired).
  useEffect(() => {
    if (wasFetching.current && !isFetching) setLeft(totalSec);
    wasFetching.current = isFetching;
  }, [isFetching, totalSec]);

  // Tick down once a second while idle.
  useEffect(() => {
    if (!totalSec || isFetching) return;
    const id = setInterval(() => setLeft((s) => Math.max(0, s - 1)), 1000);
    return () => clearInterval(id);
  }, [isFetching, totalSec]);

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={onRefresh}
      disabled={isFetching}
      className="gap-2"
    >
      <RefreshCw
        className={cn("h-3.5 w-3.5", isFetching && "animate-spin")}
      />
      {isFetching
        ? "Refreshing..."
        : totalSec
          ? `Refresh now (${left}s)`
          : "Refresh now"}
    </Button>
  );
}
