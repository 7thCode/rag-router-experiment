// RAGルーター実証実験
// 使い方: node router.mjs [チャットモデル名] [埋め込みモデル名]
//   引数を省略すると、Ollamaに入っているモデルから自動で選ぶ。
// 前提: Ollamaが起動していること（http://localhost:11434）
// 出力: 標準出力に結果、同じフォルダに results.md を保存する。

import { readFileSync, writeFileSync } from "node:fs";
import os from "node:os";

const OLLAMA = "http://localhost:11434";
const history = JSON.parse(readFileSync(new URL("./history.json", import.meta.url)));
const tests = JSON.parse(readFileSync(new URL("./tests.json", import.meta.url)));

async function api(path, body) {
  const res = await fetch(OLLAMA + path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

function pickModel(models, prefs, fallbackTest) {
  for (const p of prefs) {
    const hit = models.find((m) => m.toLowerCase().includes(p));
    if (hit) return hit;
  }
  return models.find(fallbackTest);
}

async function main() {
  let tags;
  try {
    tags = await api("/api/tags");
  } catch {
    console.error("Ollamaに接続できません。Ollamaアプリを起動してから再実行してください。");
    process.exit(1);
  }
  const models = tags.models.map((m) => m.name);
  console.log("インストール済みモデル:", models.join(", ") || "(なし)");

  const chatModel =
    process.argv[2] ||
    pickModel(models, ["qwen3", "llama3.1", "llama3", "gemma3", "qwen2.5", "mistral"], () => true);
  const embedModel =
    process.argv[3] ||
    pickModel(models, ["nomic-embed", "mxbai-embed", "bge", "embed"], (m) => m.includes("embed")) ||
    chatModel;

  if (!chatModel) {
    console.error("チャットモデルがありません。例: ollama pull qwen3:8b");
    process.exit(1);
  }
  console.log(`ルーター用モデル: ${chatModel} / 埋め込みモデル: ${embedModel}`);
  console.log(`履歴 ${history.length} 件 / テスト ${tests.length} 件\n`);

  // --- 埋め込み ---
  async function embed(texts) {
    const r = await api("/api/embed", { model: embedModel, input: texts });
    return r.embeddings;
  }
  const histVecs = await embed(history.map((h) => h.task));
  function cosine(a, b) {
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return d / (Math.sqrt(na) * Math.sqrt(nb) || 1);
  }

  // --- 判定 ---
  async function judge(task, refs) {
    const refText = refs
      ? refs.map((h) => `- 仕事: ${h.task} / 適切な振り分け: ${h.label === "heavy" ? "上位モデル" : "軽いモデル"}（実際に使ったのは${h.model === "heavy" ? "上位" : "軽い"}モデル、結果: ${h.outcome === "completed" ? "完了" : `手戻り${h.rework}回`}）`).join("\n")
      : "（参考履歴なし）";
    const prompt = `あなたは仕事の振り分け係です。新しい仕事を「軽いモデル」で処理するか「上位モデル」を使うべきか判断してください。
判断の基準: 単純・定型・影響範囲が小さい仕事は軽いモデル。複雑・曖昧・セキュリティや正確さが重要な仕事は上位モデル。

過去の類似した仕事とその結果:
${refText}

新しい仕事: ${task}

次のJSONだけを出力してください: {"route": "light" または "heavy", "reason": "短い理由"}`;
    const t0 = performance.now();
    const r = await api("/api/chat", {
      model: chatModel,
      messages: [{ role: "user", content: prompt }],
      stream: false,
      format: "json",
      options: { temperature: 0 },
    });
    const ms = performance.now() - t0;
    let route = "?";
    try { route = JSON.parse(r.message.content).route; } catch {}
    return { route, ms };
  }

  const rows = [];
  let ragHit = 0, baseHit = 0, ragMs = 0, baseMs = 0;
  for (const t of tests) {
    const [vec] = await embed([t.task]);
    const refs = history
      .map((h, i) => ({ h, s: cosine(vec, histVecs[i]) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, 3)
      .map((x) => x.h);
    const rag = await judge(t.task, refs);
    const base = await judge(t.task, null);
    if (rag.route === t.expected) ragHit++;
    if (base.route === t.expected) baseHit++;
    ragMs += rag.ms; baseMs += base.ms;
    rows.push({ task: t.task, pattern: t.pattern, expected: t.expected, rag: rag.route, base: base.route });
    console.log(`${rag.route === t.expected ? "○" : "×"} RAG:${rag.route} / 素:${base.route} / 正解:${t.expected} — ${t.task}`);
  }

  const n = tests.length;
  const lines = [
    "# RAGルーター実証実験 結果",
    "",
    `- 実行環境: ${os.arch()} / メモリ ${(os.totalmem() / 2 ** 30).toFixed(0)}GB / Node ${process.version}`,
    `- ルーター用モデル: ${chatModel}（Ollama、ローカル実行）`,
    `- 埋め込みモデル: ${embedModel}`,
    `- 履歴データ: ${history.length}件（5パターン） / テスト: ${n}件`,
    "",
    "## 結果",
    "",
    `| 条件 | 一致率 | 平均判定時間 |`,
    `|---|---|---|`,
    `| 履歴参照あり（RAG） | ${ragHit}/${n}（${((ragHit / n) * 100).toFixed(0)}%） | ${(ragMs / n / 1000).toFixed(2)}秒 |`,
    `| 履歴参照なし | ${baseHit}/${n}（${((baseHit / n) * 100).toFixed(0)}%） | ${(baseMs / n / 1000).toFixed(2)}秒 |`,
    "",
    "## 明細",
    "",
    "| 仕事 | パターン | 正解 | RAG | 履歴なし |",
    "|---|---|---|---|---|",
    ...rows.map((r) => `| ${r.task} | ${r.pattern} | ${r.expected} | ${r.rag} | ${r.base} |`),
    "",
  ];
  writeFileSync(new URL("./results.md", import.meta.url), lines.join("\n"));
  console.log(`\nRAG: ${ragHit}/${n}（${((ragHit / n) * 100).toFixed(0)}%） / 履歴なし: ${baseHit}/${n}（${((baseHit / n) * 100).toFixed(0)}%）`);
  console.log("results.md に保存しました。");
}

main().catch((e) => { console.error(e); process.exit(1); });
