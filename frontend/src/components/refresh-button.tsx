"use client";

import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function RefreshButton({
  isFetching,
  onRefresh,
  label = "Refresh now",
}: {
  isFetching: boolean;
  onRefresh: () => void;
  label?: string;
}) {
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
      {isFetching ? "Refreshing…" : label}
    </Button>
  );
}
