import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Retinal Image Classification",
  description:
    "Upload a retinal fundus photo to classify it with a trained 7-class model.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
