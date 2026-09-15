import "./globals.css";

export const metadata = {
  title: "Pramaan - Legal research workspace",
  description: "Evidence-first legal research dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
