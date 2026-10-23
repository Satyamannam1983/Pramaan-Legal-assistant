import "./globals.css";

export const metadata = {
  title: "Pramaan — Legal Intelligence",
  description: "Evidence-led legal research and document analysis workspace.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
