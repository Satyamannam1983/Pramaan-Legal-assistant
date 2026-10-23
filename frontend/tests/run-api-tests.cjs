const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const frontendDir = path.resolve(__dirname, "..");
const outputDir = path.join(frontendDir, ".test-build");
const compiler = path.join(frontendDir, "node_modules", "typescript", "bin", "tsc");

try {
  const compile = spawnSync(process.execPath, [
    compiler,
    "--target", "es2022",
    "--module", "commonjs",
    "--moduleResolution", "node",
    "--esModuleInterop",
    "--skipLibCheck",
    "--types", "node",
    "--outDir", outputDir,
    "lib/api.ts",
    "lib/follow-up.ts",
    "tests/api.test.ts",
    "tests/follow-ups.test.ts",
  ], { cwd: frontendDir, stdio: "inherit" });
  if (compile.status !== 0) process.exitCode = compile.status ?? 1;
  else {
    const testRun = spawnSync(process.execPath, [
      "--test",
      path.join(outputDir, "tests", "api.test.js"),
      path.join(outputDir, "tests", "follow-ups.test.js"),
    ], { cwd: frontendDir, stdio: "inherit" });
    if (testRun.status !== 0) process.exitCode = testRun.status ?? 1;
  }
} finally {
  fs.rmSync(outputDir, { recursive: true, force: true });
}
