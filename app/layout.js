import "./globals.css";

export const metadata = {
  title: "Poe Desk",
  description: "Local-first Poe client for GPT-5.4-Pro with usage reconciliation and audit recovery.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
