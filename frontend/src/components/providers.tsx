"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Toaster } from "sonner";
import { resumeActiveCopies } from "@/lib/copy-watcher";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { staleTime: 5_000, refetchOnWindowFocus: false },
        },
      }),
  );
  // Reattach toast progress to any copy-checkpoint transfers that were in
  // flight before the page refreshed. The transfer itself keeps running
  // server-side; this just brings the notification back.
  useEffect(() => {
    resumeActiveCopies();
  }, []);
  return (
    <QueryClientProvider client={client}>
      {children}
      <Toaster position="bottom-right" richColors visibleToasts={9} />
    </QueryClientProvider>
  );
}
