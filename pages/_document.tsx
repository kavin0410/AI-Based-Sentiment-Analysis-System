import { Html, Head, Main, NextScript } from "next/document";

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <meta charSet="utf-8" />
        <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🧠</text></svg>" />
        <meta name="theme-color" content="#6366F1" />
      </Head>
      <body className="antialiased selection:bg-indigo-500 selection:text-white">
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
