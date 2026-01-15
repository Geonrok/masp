/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        trade: {
          up: '#10B981',
          down: '#EF4444',
          neutral: '#6B7280',
        },
      },
    },
  },
  plugins: [],
};
