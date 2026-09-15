"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navigation = [
  { href: "/", label: "Research" },
  { href: "/documents", label: "Documents" },
  { href: "/compare", label: "Compare" },
  { href: "/evaluation", label: "Evaluation" },
];

export function AppShell({
  children,
  eyebrow,
}: {
  children: React.ReactNode;
  eyebrow?: string;
}) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-6 px-6 py-4">
          <Link href="/" className="flex items-center gap-3">
            <span className="grid h-9 w-9 place-items-center rounded-lg bg-slate-900 text-sm font-bold text-white">
              P
            </span>
            <span>
              <span className="block font-semibold tracking-tight">Pramaan</span>
              <span className="block text-xs text-slate-500">
                Evidence-first legal research
              </span>
            </span>
          </Link>
          <nav aria-label="Primary navigation" className="flex flex-wrap items-center justify-end gap-1">
            {navigation.map((item) => {
              const active =
                item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`rounded-md px-3 py-2 text-sm transition-colors ${
                    active
                      ? "bg-slate-100 font-medium text-slate-900"
                      : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>
      {eyebrow && (
        <div className="border-b border-slate-200 bg-white">
          <div className="mx-auto max-w-7xl px-6 py-2 text-xs font-medium uppercase tracking-wide text-teal-700">
            {eyebrow}
          </div>
        </div>
      )}
      {children}
    </div>
  );
}
