// Tight, demo-only recording (no intro hold — the spoken pitch covers that).
// Target: ~28-30s. load -> straight to #demo -> bulletin read -> reveal ->
// flagged detail -> scroll data/models/results -> chart animates.
//
// Usage: node record_demo_short.js <url> <out_dir>
const { chromium } = require("/home/pauli/wspace/agenthub/node_modules/playwright");

async function main() {
  const url = process.argv[2] || "http://localhost:8731/";
  const outDir = process.argv[3] || "./recording";

  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
    recordVideo: { dir: outDir, size: { width: 1600, height: 1000 } },
  });
  const page = await context.newPage();

  await page.goto(url, { waitUntil: "networkidle" });
  await page.waitForTimeout(600);

  await page.locator("#demo").scrollIntoViewIfNeeded();
  await page.waitForTimeout(3200); // read bulletin A and B

  await page.locator("#revealBtn").scrollIntoViewIfNeeded();
  await page.waitForTimeout(400);
  await page.locator("#revealBtn").click();
  await page.waitForTimeout(900);

  await page.mouse.move(0, 0);
  await page.waitForTimeout(2800); // read the flagged "Region 3794" detail

  const sections = ["#data", "#models", "#results"];
  for (const sel of sections) {
    await page.locator(sel).scrollIntoViewIfNeeded();
    await page.waitForTimeout(2400);
  }

  await page.evaluate(() => window.scrollBy({ top: 420, behavior: "smooth" }));
  await page.waitForTimeout(3600); // hold on the chart as bars animate in

  await page.waitForTimeout(800);
  await context.close();
  await browser.close();
  console.log("done, video written to", outDir);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
