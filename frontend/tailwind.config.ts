import type { Config } from "tailwindcss";
const config: Config = { content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"], theme: { extend: { colors: { ink: "#122033", teal: "#0c7771", gold: "#c99b4a" } } }, plugins: [] };
export default config;
