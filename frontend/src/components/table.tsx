export function Th({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <th
      className={`py-2 pr-4 font-medium whitespace-nowrap${className ? ` ${className}` : ""}`}
    >
      {children}
    </th>
  );
}
