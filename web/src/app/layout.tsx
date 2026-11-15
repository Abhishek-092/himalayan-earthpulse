import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import Header from "@/components/Header";
import "./globals.css";

const plex = IBM_Plex_Sans({ variable: "--font-plex", subsets: ["latin"], weight: ["400", "500", "600", "700"] });
const mono = IBM_Plex_Mono({ variable: "--font-mono", subsets: ["latin"], weight: ["400", "500"] });

export const metadata: Metadata = {
  title: "Himalayan EarthPulse",
  description:
    "Watch the Himalaya move with NASA-ISRO NISAR radar: ground deformation, glacier flow and forest change, measured and explained.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${plex.variable} ${mono.variable}`}>
      <head>
        {/* map tiles: open connections early so the 3D map paints sooner */}
        <link rel="preconnect" href="https://server.arcgisonline.com" crossOrigin="anonymous" />
        <link rel="preconnect" href="https://s3.amazonaws.com" crossOrigin="anonymous" />
        <link rel="preconnect" href="https://tiles.openfreemap.org" crossOrigin="anonymous" />
      </head>
      <body className="min-h-screen">
        <Header />
        {children}
      </body>
    </html>
  );
}
