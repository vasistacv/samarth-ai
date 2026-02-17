import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Samarth AI — Advanced Agricultural Intelligence Platform",
  description:
    "Enterprise-grade AI-powered agricultural insights with voice assistance, real-time analytics, crop intelligence, and data visualization. Powered by Llama 3.3.",
  keywords:
    "agriculture, AI, machine learning, rainfall analytics, crop intelligence, voice assistant, data analytics, Samarth AI",
  authors: [{ name: "Vashista C V" }],
  openGraph: {
    title: "Samarth AI — Agricultural Intelligence",
    description: "AI-powered agricultural insights with voice assistance and real-time analytics",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <body className={`${inter.variable} ${jetbrainsMono.variable} ${inter.className}`}>
        {children}
      </body>
    </html>
  );
}
