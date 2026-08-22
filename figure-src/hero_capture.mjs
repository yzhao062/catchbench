import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, "..");
const assetsDir = join(repoRoot, "assets");
const boardPath = join(repoRoot, "tests", "golden", "board.txt");
const htmlPath = join(scriptDir, "hero.html");
const viewport = { width: 1280, height: 760, scale: 2 };
const candidates = ["timeline", "masks", "boards"];

const chromeCandidates = [
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
].filter(Boolean);

const chromePath = chromeCandidates.find((candidate) => existsSync(candidate));
if (!chromePath) throw new Error("No Chrome-family browser found. Set CHROME_PATH.");
if (!existsSync(htmlPath)) throw new Error(`Missing HTML source: ${htmlPath}`);
if (!existsSync(boardPath)) throw new Error(`Missing source-of-truth board: ${boardPath}`);

const boardText = readFileSync(boardPath, "utf8");
const sectionHeaders = [...boardText.matchAll(/^\[(PRE|LIVE|POST)\]\s+([a-z_]+)\s+::\s+([^\r\n]+)$/gm)];
for (const phase of ["PRE", "LIVE", "POST"]) {
  if (!sectionHeaders.some((match) => match[1] === phase)) {
    throw new Error(`The golden board does not contain a ${phase} section.`);
  }
}

mkdirSync(assetsDir, { recursive: true });

const mimeTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".txt", "text/plain; charset=utf-8"],
]);

const server = createServer((request, response) => {
  try {
    const requestPath = decodeURIComponent(new URL(request.url, "http://127.0.0.1").pathname);
    const relativePath = requestPath.replace(/^\/+/, "");
    const filePath = normalize(join(repoRoot, relativePath));
    const rootPrefix = repoRoot.endsWith(sep) ? repoRoot : `${repoRoot}${sep}`;
    if (filePath !== repoRoot && !filePath.startsWith(rootPrefix)) {
      response.writeHead(403).end("Forbidden");
      return;
    }
    if (!existsSync(filePath) || !statSync(filePath).isFile()) {
      response.writeHead(404).end("Not found");
      return;
    }
    response.writeHead(200, { "Content-Type": mimeTypes.get(extname(filePath)) || "application/octet-stream" });
    response.end(readFileSync(filePath));
  } catch (error) {
    response.writeHead(500).end(error.message);
  }
});

await new Promise((resolveListen, rejectListen) => {
  server.once("error", rejectListen);
  server.listen(0, "127.0.0.1", resolveListen);
});

const address = server.address();
const profileDir = mkdtempSync(join(tmpdir(), "catchbench-hero-chrome-"));

function runChrome(extraArgs) {
  return new Promise((resolveRun, rejectRun) => {
    const child = spawn(chromePath, [
      "--headless=new",
      "--disable-gpu",
      "--hide-scrollbars",
      "--no-first-run",
      "--no-default-browser-check",
      `--user-data-dir=${profileDir}`,
      `--force-device-scale-factor=${viewport.scale}`,
      `--window-size=${viewport.width},${viewport.height}`,
      "--virtual-time-budget=1500",
      ...extraArgs,
    ], { windowsHide: true });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.once("error", rejectRun);
    child.once("close", (status) => {
      if (status !== 0) {
        rejectRun(new Error(`Chrome exited ${status}: ${stderr.trim()}`));
        return;
      }
      resolveRun({ stdout, stderr });
    });
  });
}

try {
  console.log(`board=${boardPath}`);
  console.log(`board_sections=${sectionHeaders.length}`);
  console.log(`browser=${chromePath}`);
  console.log(`viewport=${viewport.width}x${viewport.height}@${viewport.scale}x`);

  for (const candidate of candidates) {
    const url = `http://127.0.0.1:${address.port}/figure-src/hero.html?candidate=${candidate}`;
    const domResult = await runChrome(["--dump-dom", url]);
    if (!domResult.stdout.includes('data-ready="true"')) {
      throw new Error(`${candidate}: page did not reach ready state`);
    }
    if (!domResult.stdout.includes('data-visible-digits="none"')) {
      throw new Error(`${candidate}: rendered text contains a digit`);
    }
    if (!domResult.stdout.includes('data-visible-number-words="none"')) {
      throw new Error(`${candidate}: rendered text contains a spelled-out number`);
    }
    if (!domResult.stdout.includes('data-board-source="tests/golden/board.txt"')) {
      throw new Error(`${candidate}: board source marker missing`);
    }

    const outputPath = join(assetsDir, `hero-${candidate}.png`);
    await runChrome([`--screenshot=${outputPath}`, url]);
    const png = readFileSync(outputPath);
    if (png.toString("hex", 0, 8) !== "89504e470d0a1a0a") {
      throw new Error(`${candidate}: output is not a PNG`);
    }
    const width = png.readUInt32BE(16);
    const height = png.readUInt32BE(20);
    const expectedWidth = viewport.width * viewport.scale;
    const expectedHeight = viewport.height * viewport.scale;
    if (width !== expectedWidth || height !== expectedHeight) {
      throw new Error(`${candidate}: expected ${expectedWidth}x${expectedHeight}, got ${width}x${height}`);
    }
    console.log(`${candidate}=assets/hero-${candidate}.png ${width}x${height} ${png.length}bytes ready numbers:none`);
  }
} finally {
  await new Promise((resolveClose) => server.close(resolveClose));
  const resolvedProfile = resolve(profileDir);
  const resolvedTemp = resolve(tmpdir());
  if (resolvedProfile.startsWith(`${resolvedTemp}${sep}`) && resolvedProfile.includes("catchbench-hero-chrome-")) {
    rmSync(resolvedProfile, { recursive: true, force: true });
  }
}
