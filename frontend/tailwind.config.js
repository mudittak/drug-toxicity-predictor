/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ["Space Mono", "monospace"],
        sans: ["DM Sans", "sans-serif"],
      },
      colors: {
        bg: "#050a0e", surface: "#0d1117", card: "#111827",
        border: "#1f2937", accent: "#00d4aa", danger: "#ff4d6d",
        warn: "#ffb703", safe: "#06d6a0", muted: "#6b7280",
      },
    },
  },
  plugins: [],
}
