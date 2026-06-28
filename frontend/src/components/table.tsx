import { cn } from "@/lib/utils";

export function Th({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <th className={cn("py-2 pr-4 font-medium whitespace-nowrap", className)}>
      {children}
    </th>
  );
}
