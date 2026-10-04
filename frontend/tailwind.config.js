/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: { 950: '#0b1020', 900: '#111833', 800: '#1a2347', 700: '#26315c' },
        accent: { 400: '#7dd3fc', 500: '#38bdf8', 600: '#0ea5e9' },
      },
      fontFamily: { sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'] },
    },
  },
  plugins: [],
}
