// Records a clean, automated walkthrough of the SpaceWeather-CoT landing page:
// hold on the intro (project, problem, benefit) -> scroll into the demo ->
// pause on the bulletin comparison -> click "Reveal sources" -> pause ->
// scroll through dataset / models / results sections, pausing on the chart as it
// animates. Output is a silent video; add voiceover and edit in OBS afterward.
//
// Usage: node record_demo.js <url> <out_dir>
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

  // Hold on the intro long enough to narrate: what this is, the problem, the benefit.
  await page.waitForTimeout(9000);

  // Scroll into the demo section.
  await page.locator("#demo").scrollIntoViewIfNeeded();
  await page.waitForTimeout(3500); // let the viewer read bulletin A and B

  // Scroll the comparison into comfortable framing, then reveal.
  await page.locator("#revealBtn").scrollIntoViewIfNeeded();
  await page.waitForTimeout(600);
  await page.locator("#revealBtn").click();
  await page.waitForTimeout(1200);

  // Give time to read the flagged "Region 3794" detail.
  await page.mouse.move(0, 0);
  await page.waitForTimeout(2800);

  // Smooth scroll through the rest of the page.
  const sections = ["#data", "#models", "#results"];
  for (const sel of sections) {
    await page.locator(sel).scrollIntoViewIfNeeded();
    await page.waitForTimeout(2600);
  }

  // Hold on the chart while its bars animate in, then the honesty box.
  await page.evaluate(() => window.scrollBy({ top: 420, behavior: "smooth" }));
  await page.waitForTimeout(3200);

  await page.waitForTimeout(1500);
  await context.close();
  await browser.close();
  console.log("done, video written to", outDir);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
