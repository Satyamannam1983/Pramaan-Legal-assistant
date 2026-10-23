"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

type IconName = "dashboard" | "assistant" | "documents" | "compare" | "analytics" | "menu" | "close" | "scales";

const navigation: Array<{
  group: string;
  items: Array<{ href: string; label: string; icon: IconName }>;
}> = [
  {
    group: "Workspace",
    items: [
      { href: "/dashboard", label: "Dashboard", icon: "dashboard" },
      { href: "/", label: "Legal Assistant", icon: "assistant" },
      { href: "/documents", label: "Documents", icon: "documents" },
      { href: "/compare", label: "Document Comparison", icon: "compare" },
    ],
  },
  {
    group: "Analytics",
    items: [{ href: "/evaluation", label: "Evaluation", icon: "analytics" }],
  },
];

function Icon({ name, className = "" }: { name: IconName; className?: string }) {
  const common = {
    className,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.7,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true as const,
  };

  if (name === "dashboard") {
    return <svg {...common}><rect x="3.5" y="3.5" width="7" height="7" rx="1.3" /><rect x="13.5" y="3.5" width="7" height="7" rx="1.3" /><rect x="3.5" y="13.5" width="7" height="7" rx="1.3" /><rect x="13.5" y="13.5" width="7" height="7" rx="1.3" /></svg>;
  }
  if (name === "assistant") {
    return <svg {...common}><path d="M5 4.5h14a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-7l-5 3v-3H5a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2Z" /><path d="M8 9h8M8 13h5" /></svg>;
  }
  if (name === "documents") {
    return <svg {...common}><path d="M6 3.5h8l4 4v13H6a2 2 0 0 1-2-2v-13a2 2 0 0 1 2-2Z" /><path d="M14 3.5v5h5M8 13h8M8 17h8" /></svg>;
  }
  if (name === "compare") {
    return <svg {...common}><rect x="3.5" y="4" width="7" height="16" rx="1.5" /><rect x="13.5" y="4" width="7" height="16" rx="1.5" /><path d="M6 8h2M6 12h2M16 8h2M16 12h2" /></svg>;
  }
  if (name === "analytics") {
    return <svg {...common}><path d="M4 20V11h4v9M10 20V5h4v15M16 20v-7h4v7" /></svg>;
  }
  if (name === "scales") {
    return <svg {...common}><path d="M12 3v17M7 21h10M5 6h14M12 3l-2-1M12 3l2-1M5 6l-3 7h6L5 6ZM19 6l-3 7h6l-3-7Z" /><path d="M2 13a3 3 0 0 0 6 0M16 13a3 3 0 0 0 6 0" /></svg>;
  }
  if (name === "close") {
    return <svg {...common}><path d="m6 6 12 12M18 6 6 18" /></svg>;
  }
  return <svg {...common}><path d="M4 7h16M4 12h16M4 17h16" /></svg>;
}

export function AppShell({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const [drawerOpen, setDrawerOpen] = useState(false);

  useEffect(() => {
    setDrawerOpen(false);
  }, [pathname]);

  useEffect(() => {
    if (!drawerOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setDrawerOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [drawerOpen]);

  return (
    <div className="app-shell">
      {drawerOpen && (
        <button
          type="button"
          className="sidebar-backdrop"
          aria-label="Close navigation menu"
          onClick={() => setDrawerOpen(false)}
        />
      )}
      <aside className={`app-sidebar${drawerOpen ? " app-sidebar-open" : ""}`} aria-label="Application navigation">
        <Link href="/dashboard" className="brand-lockup">
          <Icon name="scales" className="brand-mark" />
          <span className="brand-copy">
            <span className="brand-name">Pramaan</span>
            <span className="brand-subtitle">Legal Intelligence</span>
          </span>
        </Link>
        <nav className="sidebar-nav" aria-label="Primary navigation">
          {navigation.map((section) => (
            <div className="nav-section" key={section.group}>
              <p className="nav-section-title">{section.group}</p>
              <div className="nav-items">
                {section.items.map((item) => {
                  const active = item.href === "/"
                    ? pathname === "/"
                    : pathname === item.href || pathname.startsWith(`${item.href}/`);
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      aria-current={active ? "page" : undefined}
                      className={`nav-item${active ? " nav-item-active" : ""}`}
                    >
                      <Icon name={item.icon} className="nav-icon" />
                      <span>{item.label}</span>
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>
        <div className="sidebar-footnote">Evidence-led legal research</div>
      </aside>
      <div className="app-main">
        <header className="app-topbar">
          <button
            type="button"
            className="mobile-menu-button"
            aria-label={drawerOpen ? "Close navigation menu" : "Open navigation menu"}
            aria-expanded={drawerOpen}
            onClick={() => setDrawerOpen((open) => !open)}
          >
            <Icon name={drawerOpen ? "close" : "menu"} className="topbar-icon" />
          </button>
          <span className="topbar-context">Pramaan workspace</span>
        </header>
        <div className="app-page">{children}</div>
      </div>
    </div>
  );
}
