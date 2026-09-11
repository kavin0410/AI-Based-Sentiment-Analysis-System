/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#F4F6FB",
        card: "#FFFFFF",
        primary: {
          50: "#EEF2FF",
          100: "#E0E7FF",
          500: "#6366F1",
          600: "#4F46E5",
          700: "#4338CA",
        },
        slate: {
          850: "#151e2e",
          900: "#0F172A",
        },
        emerald: {
          500: "#10B981",
          600: "#059669",
        },
        rose: {
          500: "#EF4444",
          600: "#DC2626",
        }
      },
      fontFamily: {
        sans: ["var(--font-plus-jakarta)", "Plus Jakarta Sans", "Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      boxShadow: {
        card: "0 6px 20px rgba(15, 23, 42, 0.05), 0 2px 6px rgba(15, 23, 42, 0.03)",
        glass: "0 14px 40px rgba(15, 23, 42, 0.07), 0 4px 12px rgba(15, 23, 42, 0.03)",
        glow: "0 8px 30px rgba(99, 102, 241, 0.25)",
      },
      borderRadius: {
        xl: "16px",
        "2xl": "20px",
        "3xl": "28px",
      }
    },
  },
  plugins: [],
};
