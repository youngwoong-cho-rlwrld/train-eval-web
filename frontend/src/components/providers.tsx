"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Toaster } from "sonner";
import { resumeActiveMoves } from "@/lib/move-watcher";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { staleTime: 5_000, refetchOnWindowFocus: false },
        },
      }),
  );
  // Reattach toast progress to any move-checkpoint transfers that were in
  // flight before the page refreshed. The transfer itself keeps running
  // server-side; this just brings the notification back.
  useEffect(() => {
    resumeActiveMoves();
  }, []);
  return (
    <QueryClientProvider client={client}>
      {children}
      <Toaster position="top-right" richColors />
    </QueryClientProvider>
  );
}
