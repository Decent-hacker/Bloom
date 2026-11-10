import assert from "node:assert/strict";
import { test } from "node:test";
import config from "./vite.config.js";

test("production CSP follows the configured API origin", () => {
  const previous = process.env.VITE_API_URL;
  try {
    process.env.VITE_API_URL = "https://custom-api.example.com/";
    const resolved = config({ mode: "production" });
    const html = resolved.plugins[1].transformIndexHtml("<head></head>");
    assert.ok(html.includes("connect-src 'self' https://custom-api.example.com;"));
    assert.ok(!html.includes("bloom-api.onrender.com"));

    process.env.VITE_API_URL = "/";
    assert.ok(config({ mode: "production" }).plugins[1]
      .transformIndexHtml("<head></head>").includes("connect-src 'self';"));

    process.env.VITE_API_URL = "javascript:alert(1)";
    assert.throws(() => config({ mode: "production" }), /HTTP\(S\)/);
    process.env.VITE_API_URL = "//api.example.com";
    assert.throws(() => config({ mode: "production" }), /explicit HTTP\(S\)/);
  } finally {
    if (previous === undefined) delete process.env.VITE_API_URL;
    else process.env.VITE_API_URL = previous;
  }
});
