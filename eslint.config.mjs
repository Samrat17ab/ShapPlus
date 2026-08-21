import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Local build/tooling output -- gitignored, never source.
    "dist/**",
    ".wrangler/**",
    ".vinext/**",
    "tmp/**",
    // Python virtualenv -- vendored JS bundles inside installed packages
    // (lime, matplotlib, shap, sklearn) are not part of this app.
    ".venv-shap-plus/**",
  ]),
]);

export default eslintConfig;
