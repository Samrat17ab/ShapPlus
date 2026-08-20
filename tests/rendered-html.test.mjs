import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("renders the mortgage hybrid XAI workbench", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Mortgage XAI Studio<\/title>/i);
  assert.match(html, /Evidence you can/);
  assert.match(html, /Signed contribution vector/);
  assert.match(html, /CONSTRAINED LOCAL RULE/);
  assert.match(html, /COUNTERFACTUAL RECOURSE/);
  assert.match(html, /COMPLIANCE LOG/);
});

test("documents the production boundary", async () => {
  const response = await render();
  const html = await response.text();
  assert.match(html, /Decision support, not autonomous underwriting/);
  assert.match(html, /Production use requires a trained and validated mortgage model/);
  assert.doesNotMatch(html, /Moodly|codex-preview|Building your site/);
});
