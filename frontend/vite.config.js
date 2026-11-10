import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Production CSP injected into built index.html only (keeps Vite HMR working in dev).
const makeCsp = (apiOrigin) =>
  "default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; " +
  "font-src https://fonts.gstatic.com; img-src 'self' data:; " +
  `connect-src 'self'${apiOrigin ? ` ${apiOrigin}` : ""};`;

/** @returns {import('vite').Plugin} */
const cspPlugin = (apiOrigin) => ({
  name: "inject-csp",
  apply: "build",
  transformIndexHtml(html) {
    return html.replace(
      "</head>",
      `  <meta http-equiv="Content-Security-Policy" content="${makeCsp(apiOrigin)}" />\n  </head>`
    );
  },
});

export default defineConfig(({ mode, command }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  // Development always uses the local proxy, including with stale hosted settings.
  const apiUrl = command === "serve" ? "" : env.VITE_API_URL?.trim() || "";
  let apiOrigin = "";
  if (apiUrl.startsWith("//")) {
    throw new Error("VITE_API_URL must use an explicit HTTP(S) scheme or a relative path.");
  }
  if (/[?#\\\s]/.test(apiUrl)) {
    throw new Error("VITE_API_URL must not contain a query, fragment, backslash, or whitespace.");
  }
  if (apiUrl && !apiUrl.startsWith("/")) {
    const parsed = new URL(apiUrl);
    if (!["https:", "http:"].includes(parsed.protocol) || parsed.username || parsed.password) {
      throw new Error("VITE_API_URL must be an HTTP(S) URL without credentials.");
    }
    if (!/^https?:\/\//i.test(apiUrl)) {
      throw new Error("VITE_API_URL must use an explicit HTTP(S) scheme or a relative path.");
    }
    apiOrigin = parsed.origin;
  }
  return {
    plugins: [react(), cspPlugin(apiOrigin)],
    server: {
      port: 8080,
      strictPort: true,
      proxy: { "/api": process.env.BLOOM_BACKEND_URL || "http://127.0.0.1:8000" },
    },
  };
});
