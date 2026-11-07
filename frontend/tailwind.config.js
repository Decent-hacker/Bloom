/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f0f9ff",
          100: "#e0f2fe",
          500: "#0ea5e9",
          600: "#0284c7",
          700: "#0369a1",
        },
        concern: {
          acne: "#3b82f6",
          redness: "#ef4444",
          blemish: "#eab308",
        },
        severity: {
          mild: { bg: "#dcfce7", text: "#166534" },
          moderate: { bg: "#fef9c3", text: "#854d0e" },
          severe: { bg: "#fee2e2", text: "#991b1b" },
        },
        surface: {
          base: "#ffffff",
          subtle: "#f8fafc",
          border: "#e2e8f0",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
