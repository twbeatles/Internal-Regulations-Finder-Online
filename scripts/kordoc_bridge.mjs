import { parse } from "kordoc";

function collectText(value, bucket) {
  if (value === null || value === undefined) {
    return;
  }
  if (typeof value === "string") {
    const text = value.trim();
    if (text) {
      bucket.push(text);
    }
    return;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      collectText(item, bucket);
    }
    return;
  }
  if (typeof value === "object") {
    for (const [key, child] of Object.entries(value)) {
      if (key === "bbox" || key === "style" || key === "pageNumber" || key === "type") {
        continue;
      }
      collectText(child, bucket);
    }
  }
}

function buildPageMap(blocks) {
  if (!Array.isArray(blocks)) {
    return [];
  }
  const pages = new Map();
  for (const block of blocks) {
    const pageNumber = Number(block?.pageNumber);
    if (!Number.isFinite(pageNumber) || pageNumber <= 0) {
      continue;
    }
    const parts = [];
    collectText(block, parts);
    if (parts.length === 0) {
      continue;
    }
    const existing = pages.get(pageNumber) ?? [];
    existing.push(parts.join("\n"));
    pages.set(pageNumber, existing);
  }
  return [...pages.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([pageNumber, parts]) => [pageNumber, parts.join("\n")]);
}

function normalizeWarnings(result) {
  if (!Array.isArray(result?.warnings)) {
    return [];
  }
  return result.warnings.map((item) => String(item)).filter(Boolean);
}

const filePath = process.argv[2];
if (!filePath) {
  console.error(JSON.stringify({ error: "missing file path" }));
  process.exit(1);
}

try {
  const result = await parse(filePath);
  if (!result?.success) {
    console.error(JSON.stringify(result ?? { error: "kordoc parse failed" }));
    process.exit(2);
  }
  const payload = {
    title: result?.metadata?.title ?? null,
    markdown: result?.markdown ?? "",
    metadata: result?.metadata ?? {},
    warnings: normalizeWarnings(result),
    pageMap: buildPageMap(result?.blocks),
    blocks: Array.isArray(result?.blocks) ? result.blocks : [],
  };
  process.stdout.write(JSON.stringify(payload));
} catch (error) {
  console.error(
    JSON.stringify({
      error: error instanceof Error ? error.message : String(error),
    }),
  );
  process.exit(3);
}