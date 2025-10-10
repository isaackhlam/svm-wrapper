import js from "@eslint/js";
import globals from "globals";
import tseslint from "typescript-eslint";
import pluginVue from "eslint-plugin-vue";
import json from "@eslint/json";
import markdown from "@eslint/markdown";
import css from "@eslint/css";
import { defineConfig } from "eslint/config";

//export default defineConfig([
  //{ files: ["**/*.{js,mjs,cjs,ts,mts,cts,vue}"], plugins: { js }, extends: ["js/recommended"], languageOptions: { globals: {...globals.browser, ...globals.node} } },
  //tseslint.configs.recommended,
  //{ files: ["**/*.vue"], ...pluginVue.configs['flat/recommended'], languageOptions: { parserOptions: { parser: tseslint.parser } } },
  //{ files: ["**/*.json"], plugins: { json }, language: "json/json", extends: ["json/recommended"] },
  //{ files: ["**/*.md"], plugins: { markdown }, language: "markdown/gfm", extends: ["markdown/recommended"] },
  //{ files: ["**/*.css"], plugins: { css }, language: "css/css", extends: ["css/recommended"] },
//]);


export default defineConfig([
  {
    ignores: ['**/.vite', '**/megalinter-reports'],
  },

  {
    files: ["**/*.{js,mjs,cjs,ts,mts,cts,vue}"],
    plugins: { js },
    extends: ["js/recommended"],
    languageOptions: {
      globals: { ...globals.browser, ...globals.node },
    },
  },

  tseslint.configs.recommended,

  ...pluginVue.configs["flat/recommended"].map((config) => ({
    ...config,
    files: ["**/*.vue"],
    languageOptions: {
      ...(config.languageOptions || {}),
      parserOptions: {
        ...(config.languageOptions?.parserOptions || {}),
        parser: tseslint.parser,
      },
    },
  })),


  {
    files: ["**/*.json"],
    plugins: { json },
    language: "json/json",
    extends: ["json/recommended"],
  },

  {
    files: ["**/*.md"],
    plugins: { markdown },
    language: "markdown/gfm",
    extends: ["markdown/recommended"],
  },

  {
    files: ["**/*.css"],
    plugins: { css },
    language: "css/css",
    extends: ["css/recommended"],
  },
]);

