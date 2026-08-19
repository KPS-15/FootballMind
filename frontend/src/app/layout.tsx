import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FootballMind AI - Multimodal Football Intelligence Framework",
  description: "A Multimodal Deep Learning Framework for Predictive, Tactical and Explainable Football Intelligence",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full antialiased dark">
      <body className="min-h-full flex flex-col bg-slate-950 text-slate-100">{children}</body>
    </html>
  );
}
