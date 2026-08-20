import type { Metadata } from "next";
import { Geist } from "next/font/google";
import "./globals.css";

const geist = Geist({ variable: "--font-geist", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Mortgage XAI Studio",
  description:
    "A mortgage-specific hybrid explainability workbench for evidence, local rules, recourse, and audit.",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
  openGraph: {
    title: "Mortgage XAI Studio",
    description: "Evidence you can act on. Decisions you can audit.",
    type: "website",
    images: [{ url: "/mortgage-xai-og.png", width: 1734, height: 907, alt: "Mortgage XAI Studio explanation pathway" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Mortgage XAI Studio",
    description: "Evidence you can act on. Decisions you can audit.",
    images: ["/mortgage-xai-og.png"],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={geist.variable}>{children}</body>
    </html>
  );
}
