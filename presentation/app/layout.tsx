import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  metadataBase: new URL("https://amazon-review-intelligence.ma-slri2128.chatgpt.site"),
  title: "Amazon Review Intelligence | Rating Prediction",
  description: "An end-to-end Transformer study for predicting exact one-to-five-star Amazon review ratings.",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
  openGraph: {
    title: "Amazon Review Intelligence",
    description: "From review text to a precise 1–5 star prediction.",
    type: "website",
    images: [{ url: "/og.png", width: 1536, height: 1024, alt: "Amazon Review Intelligence model comparison" }],
  },
  twitter: { card: "summary_large_image", title: "Amazon Review Intelligence", description: "From review text to a precise 1–5 star prediction.", images: ["/og.png"] },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body></html>;
}
