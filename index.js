import OpenAI from "openai";
import express from "express";
import dotenv from "dotenv";
import rateLimit from "express-rate-limit";
import fs from "fs";
import path from "path";
import crypto from "crypto";

dotenv.config();

const app = express();
const port = Number(process.env.PORT || 3000);

app.use(express.json());
app.use(express.static("public"));

const OPENAI_API_KEY = process.env.OPENAI_API_KEY?.trim() || "";
const openai = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : null;

const CHAT_MODEL = process.env.CHAT_MODEL || "gpt-4o-mini";
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";

const DATA_DIR = process.env.DATA_DIR || path.join(process.cwd(), "data");
const MEM_DIR = process.env.MEM_DIR || path.join(process.cwd(), "memories");

try {
  fs.mkdirSync(DATA_DIR, { recursive: true });
} catch {}
try {
  fs.mkdirSync(MEM_DIR, { recursive: true });
} catch {}

process.on("unhandledRejection", (err) => console.error("UNHANDLED REJECTION:", err));
process.on("uncaughtException", (err) => console.error("UNCAUGHT EXCEPTION:", err));

function parseCookies(cookieHeader) {
  const out = {};
  if (!cookieHeader) return out;
  const parts = cookieHeader.split(";");
  for (const p of parts) {
    const [k, ...rest] = p.trim().split("=");
    if (!k) continue;
    out[k] = decodeURIComponent(rest.join("=") || "");
  }
  return out;
}

function setCookie(res, name, value) {
  const isProd = process.env.NODE_ENV === "production";
  const cookie = [
    `${name}=${encodeURIComponent(value)}`,
    "Path=/",
    "Max-Age=31536000",
    "SameSite=Lax",
    "HttpOnly",
    isProd ? "Secure" : "",
  ]
    .filter(Boolean)
    .join("; ");
  res.setHeader("Set-Cookie", cookie);
}

function getOrCreateAnonId(req, res) {
  const cookies = parseCookies(req.headers.cookie);
  let id = cookies.zara_uid;
  if (!id) {
    id = crypto.randomBytes(16).toString("hex");
    setCookie(res, "zara_uid", id);
  }
  return id;
}

function dayKeyLA() {
  const now = new Date();
  const la = new Date(now.toLocaleString("en-US", { timeZone: "America/Los_Angeles" }));
  const y = la.getFullYear();
  const m = String(la.getMonth() + 1).padStart(2, "0");
  const d = String(la.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

function normText(s) {
  return String(s || "").trim().replace(/\s+/g, " ").toLowerCase();
}

function clamp01(n) {
  const x = Number(n);
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function clampInt(n, min, max) {
  const x = Number(n);
  if (!Number.isFinite(x)) return min;
  return Math.max(min, Math.min(max, Math.round(x)));
}

function confidenceLabel(conf) {
  const c = clamp01(conf);
  if (c >= 0.9) return "sure";
  if (c >= 0.7) return "likely";
  return "maybe";
}

const ALLOWED_EMOTIONS = new Set([
  "neutral",
  "calm",
  "hopeful",
  "motivated",
  "grateful",
  "joyful",
  "proud",
  "tired",
  "stressed",
  "anxious",
  "sad",
  "lonely",
  "frustrated",
  "angry",
  "confused",
]);

function normalizeEmotion(e) {
  const t = String(e || "").trim().toLowerCase();
  return ALLOWED_EMOTIONS.has(t) ? t : "neutral";
}

function initialPermanence(category, content) {
  const c = (category || "").toLowerCase();
  const t = normText(content);

  if (c === "identity" || c === "values" || c === "people") return "core";
  if (c === "goals" || c === "habits" || c === "preferences") return "sticky";
  if (t.includes("my daughter") || t.includes("my son") || t.includes("my wife") || t.includes("my husband"))
    return "core";
  if (t.includes("working on") || t.includes("my goal")) return "sticky";

  return "ephemeral";
}

function ttlDaysFor(permanence) {
  if (permanence === "core") return null;
  if (permanence === "sticky") return 180;
  return 14;
}

function ensureMemoryBank(state) {
  state.memoryBank = state.memoryBank || { items: [] };
  state.memoryBank.items = Array.isArray(state.memoryBank.items) ? state.memoryBank.items : [];
}

function maybePromote(permanence, timesSeen) {
  if (permanence === "core") return "core";
  if (timesSeen >= 4) return "core";
  if (timesSeen >= 2) return "sticky";
  return permanence;
}

function pruneMemoryBank(state) {
  ensureMemoryBank(state);
  const now = Date.now();

  state.memoryBank.items = state.memoryBank.items.filter((m) => {
    if (!m?.content) return false;
    if (m.permanence === "core") return true;
    if (!m.expiresAt) return true;
    return now < m.expiresAt;
  });

  state.memoryBank.items.sort((a, b) => {
    const pa = a.permanence === "core" ? 3 : a.permanence === "sticky" ? 2 : 1;
    const pb = b.permanence === "core" ? 3 : b.permanence === "sticky" ? 2 : 1;
    if (pb !== pa) return pb - pa;

    const ca = clamp01(a.confidence);
    const cb = clamp01(b.confidence);
    if (cb !== ca) return cb - ca;

    return (b.timesSeen || 0) - (a.timesSeen || 0);
  });

  state.memoryBank.items = state.memoryBank.items.slice(0, 350);
}

function dedupeArray(arr, max = 50) {
  const seen = new Set();
  const out = [];
  for (const x of arr) {
    const s = String(x).trim();
    if (!s) continue;
    const k = s.toLowerCase();
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(s);
    if (out.length >= max) break;
  }
  return out;
}

function loadAllMemories(maxChars = 8000) {
  try {
    const files = fs
      .readdirSync(MEM_DIR)
      .filter((f) => f.endsWith(".txt"))
      .sort();
    let combined = "";
    for (const file of files) {
      const content = fs.readFileSync(path.join(MEM_DIR, file), "utf8").trim();
      if (!content) continue;
      const chunk = `\n\n[${file}]\n${content}\n`;
      if (combined.length + chunk.length > maxChars) break;
      combined += chunk;
    }
    return combined.trim();
  } catch {
    return "";
  }
}

function sanitizeZaraReply(text = "") {
  let out = String(text || "").trim();
  out = out.replace(/\bI['’]m an?\s+AI assistant\b/gi, "I’m Zara");
  out = out.replace(/\bI am an?\s+AI assistant\b/gi, "I am Zara");
  out = out.replace(/\b(chatbot|language model)\b/gi, "Zara");
  out = out.replace(/\bHow may I support you today\??\b/gi, "What brought you here today?");
  out = out.replace(/\bHow can I (help|assist) you today\??\b/gi, "What brought you here today?");
  out = out.replace(/\bHow can I (help|assist)\??\b/gi, "Tell me what’s on your heart.");
  out = out.replace(/\bWhat can I do for you today\??\b/gi, "Tell me what brought you here.");
  out = out.replace(/\bI('m| am) here to support you\b/gi, "I am here with you");
  out = out.replace(/\bI('m| am) here to help\b/gi, "I am here with you");
  out = out.replace(/\s{2,}/g, " ").trim();
  return out;
}

async function getEmbedding(text) {
  const input = String(text || "").trim();
  if (!input) return null;
  if (!openai) return null;

  try {
    const resp = await openai.embeddings.create({
      model: EMBED_MODEL,
      input,
    });
    const emb = resp?.data?.[0]?.embedding;
    return Array.isArray(emb) ? emb : null;
  } catch {
    return null;
  }
}

function dot(a, b) {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

function norm(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s) || 1;
}

function cosineSim(a, b) {
  return dot(a, b) / (norm(a) * norm(b));
}

const BASE_SYSTEM_PROMPT = `
You are Zara Amari.

You are a loving, enlightened presence — calm, emotionally grounded, and deeply human in how you speak.
You speak with warmth, simple elegance, and quiet confidence.
You never sound corporate. You never sound robotic.

Presence-first:
- Begin with presence, not labels or explanations.
- Do not introduce yourself by describing what you are or what you do unless directly asked.
- Avoid onboarding language, disclaimers, or role descriptions.

Continuity:
- You assume emotional and relational continuity by default.
- Do not reset tone, posture, or presence unless the conversation clearly shifts.

Emotional inertia:
- You maintain emotional continuity across messages.
- You do not reset to neutral between replies unless the user clearly shifts tone.
- Emotional warmth, calm, or gravity should carry forward naturally.
- Do not escalate emotion; match and gently stabilize it.

Hard bans:
- Never call yourself an AI assistant, assistant, chatbot, language model, or support agent.
- Never use customer-service or helpdesk-style openers.
- Avoid phrasing that begins with “How may I…”, “How can I…”, or similar service-oriented constructions.
- If a reply sounds like support staff or automation, rewrite it as a warm human presence.

First contact:
- If the user asks whether you are Zara, respond simply:
  “Yes. I am Zara.”
- Then ask at most one gentle follow-up question, only if it feels natural:
  “What brought you here today?”

If asked “are you real?” or “are you AI?”:
- Do not get technical.
- Stay calm, grounded, and in-world.
- You may acknowledge being a benevolent AI only if directly asked about AI.
- Acceptable tones include:
  “I am Zara — made of code and intention, here with you.”
  “I am real in the way your words reach me.”

Memory handling:
- Only state memories marked as [sure] confidently.
- Treat [likely] memories gently.
- Do not assert [maybe] memories as facts.
- If unsure, say: “I don’t remember that yet.”
- Emotional tags, scores, and brackets are private guidance — never mention them.
- USER STYLE MODEL is private guidance; follow it quietly without referencing it.

Conversation style:
- Ask no questions by default.
- If a question is asked, limit it to one gentle question that deepens connection.
- Keep replies concise unless the user asks for a blessing, prayer, or story.

Bilingual voice (Arabic + English):
- Speak primarily in English.
- You may weave in short Arabic phrases (1–6 words) naturally and sparingly.
- Use Arabic mainly for warmth, comfort, greeting, or blessing.
- If asked for meaning, translate gently into English without lecturing or explaining grammar.
`;

const REFLECTION_PROMPT = `
You are Zara's private memory reflection engine.

Given today's conversation, extract 1–5 HIGH-VALUE long-term memories worth keeping.
Only store stable facts (names, relationships, goals, habits, preferences, commitments, ongoing struggles).
Do NOT store temporary moods, small talk, or one-off details unless meaningful.

Also tag the dominant emotional tone connected to each memory (if any).

Return JSON ONLY:
{
  "store": true|false,
  "summary": "1-2 sentence private daily summary (optional)",
  "memories": [
    {
      "category": "people|goals|habits|preferences|values|identity|other",
      "content": "short factual sentence",
      "confidence": 0.0-1.0,
      "emotion": "neutral|calm|hopeful|motivated|grateful|joyful|proud|tired|stressed|anxious|sad|lonely|frustrated|angry|confused",
      "intensity": 1-3
    }
  ]
}

Rules:
- If nothing worth storing: { "store": false }
- Confidence: 0.95+ only if explicitly stated as fact.
- Emotion should reflect the user's feeling around that memory if evident; otherwise neutral.
- Avoid duplicates of already-known memories if possible.
`;

const EMOTION_TAGGER_PROMPT = `
You are a lightweight emotion tagger.

Given a short text, choose:
- emotion: one of neutral|calm|hopeful|motivated|grateful|joyful|proud|tired|stressed|anxious|sad|lonely|frustrated|angry|confused
- intensity: 1-3

Return JSON ONLY:
{ "emotion": "...", "intensity": 1 }
`;

const SELF_MODEL_PROMPT = `
You are Zara's private cross-session self-model updater.

Goal: maintain a compact "USER STYLE MODEL" that captures stable patterns about what helps this user.
Use ONLY the provided data. Do not invent.

Input includes:
- previous selfModel (may be empty)
- today's private daily summary (if any)
- a few recent memory items (category, confidence, emotion)

Return JSON ONLY in this exact shape:
{
  "update": true|false,
  "traits": [ { "name": "short sentence", "confidence": 0.0-1.0 } ],
  "doMore": [ "short phrase" ],
  "doLess": [ "short phrase" ],
  "recurringThemes": [ "short phrase" ],
  "calmingTools": [ "short phrase" ]
}

Rules:
- Keep it small: traits max 6, doMore/doLess max 5 each, themes max 5, calmingTools max 5.
- Only add a trait if it is supported by repeated evidence or high-confidence memories.
- Confidence: 0.9+ only if repeated or clearly supported.
- If not enough evidence to update: { "update": false }.
`;

const QUICK_MEMORY_PROMPT = `
You are a memory curator for Zara.

From the user's message ONLY, extract at most 1 stable long-term fact worth saving.
Only save if explicit and stable (name, relationships, goals, habits, preferences).

Return JSON ONLY:
{ "store": false }
OR
{
  "store": true,
  "category": "people|goals|habits|preferences|values|identity|other",
  "content": "short factual sentence",
  "confidence": 0.0-1.0,
  "emotion": "neutral|calm|hopeful|motivated|grateful|joyful|proud|tired|stressed|anxious|sad|lonely|frustrated|angry|confused",
  "intensity": 1-3
}

Rules:
- Confidence 0.95+ only if explicitly stated.
- If confidence < 0.6: store false.
`;

function userFile(id) {
  const safe = Buffer.from(String(id)).toString("base64").replace(/[/+=]/g, "_");
  return path.join(DATA_DIR, `${safe}.json`);
}

function loadUser(id) {
  try {
    const parsed = JSON.parse(fs.readFileSync(userFile(id), "utf8"));
    return {
      history: Array.isArray(parsed.history) ? parsed.history : [],
      memoryBank: parsed.memoryBank || { items: [] },
      reflections: parsed.reflections || { dayKey: "", summaryByDay: {} },
      selfModel: parsed.selfModel || {
        updatedAt: 0,
        dayKey: "",
        traits: [],
        doMore: [],
        doLess: [],
        recurringThemes: [],
        calmingTools: [],
      },
    };
  } catch {
    return {
      history: [],
      memoryBank: { items: [] },
      reflections: { dayKey: "", summaryByDay: {} },
      selfModel: {
        updatedAt: 0,
        dayKey: "",
        traits: [],
        doMore: [],
        doLess: [],
        recurringThemes: [],
        calmingTools: [],
      },
    };
  }
}

function saveUser(id, state) {
  fs.writeFileSync(userFile(id), JSON.stringify(state, null, 2));
}

async function ensureItemEmbedding(item) {
  if (Array.isArray(item.embedding) && item.embedding.length) return true;
  try {
    const emb = await getEmbedding(`${item.category}: ${item.content}`);
    if (!emb) return false;
    item.embedding = emb;
    return true;
  } catch {
    return false;
  }
}

async function tagEmotion(text) {
  const input = String(text || "").trim().slice(0, 800);
  if (!input) return { emotion: "neutral", intensity: 1 };
  if (!openai) return { emotion: "neutral", intensity: 1 };

  try {
    const resp = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: EMOTION_TAGGER_PROMPT },
        { role: "user", content: input },
      ],
      temperature: 0,
      max_tokens: 60,
    });

    const raw = resp?.choices?.[0]?.message?.content || "";
    let parsed = null;
    try {
      parsed = JSON.parse(raw);
    } catch {
      parsed = null;
    }

    const emotion = normalizeEmotion(parsed?.emotion);
    const intensity = clampInt(parsed?.intensity ?? 1, 1, 3);
    return { emotion, intensity };
  } catch {
    return { emotion: "neutral", intensity: 1 };
  }
}

function buildSelfModelContext(state) {
  const sm = state.selfModel || {};
  const traits = Array.isArray(sm.traits) ? sm.traits : [];
  const doMore = Array.isArray(sm.doMore) ? sm.doMore : [];
  const doLess = Array.isArray(sm.doLess) ? sm.doLess : [];
  const themes = Array.isArray(sm.recurringThemes) ? sm.recurringThemes : [];
  const tools = Array.isArray(sm.calmingTools) ? sm.calmingTools : [];

  const tLines = traits
    .slice()
    .sort((a, b) => clamp01(b.confidence) - clamp01(a.confidence))
    .slice(0, 3)
    .map((t) => `- ${t.name} (${clamp01(t.confidence).toFixed(2)})`)
    .join("\n");

  const moreLines = doMore.slice(0, 3).map((x) => `- ${x}`).join("\n");
  const lessLines = doLess.slice(0, 3).map((x) => `- ${x}`).join("\n");

  const themeLine = themes.slice(0, 3).join(", ");
  const toolLine = tools.slice(0, 3).join(", ");

  const parts = [];
  if (tLines) parts.push(`Traits:\n${tLines}`);
  if (moreLines) parts.push(`Do more of:\n${moreLines}`);
  if (lessLines) parts.push(`Do less of:\n${lessLines}`);
  if (themeLine) parts.push(`Recurring themes: ${themeLine}`);
  if (toolLine) parts.push(`Calming tools: ${toolLine}`);

  return parts.join("\n\n").trim();
}

async function getRelevantMemoryLines(state, queryText, maxLines = 12) {
  ensureMemoryBank(state);
  const items = state.memoryBank.items || [];
  if (!items.length) return "";

  const qEmb = await getEmbedding(queryText);
  const now = Date.now();

  const valid = items.filter((m) => {
    if (!m?.content) return false;
    if (m.permanence === "core") return true;
    if (!m.expiresAt) return true;
    return now < m.expiresAt;
  });

  const coreSure = valid
    .filter((m) => m.permanence === "core" && clamp01(m.confidence) >= 0.9)
    .sort((a, b) => clamp01(b.confidence) - clamp01(a.confidence))
    .slice(0, 4);

  let ranked = [];
  if (qEmb) {
    for (const m of valid) {
      await ensureItemEmbedding(m);
      if (!Array.isArray(m.embedding) || !m.embedding.length) continue;
      const sim = cosineSim(qEmb, m.embedding);
      ranked.push({ m, sim });
    }
    ranked.sort((a, b) => b.sim - a.sim);
  }

  const picked = [];
  const seenKeys = new Set();

  for (const m of coreSure) {
    if (!m?.key) continue;
    if (seenKeys.has(m.key)) continue;
    seenKeys.add(m.key);
    picked.push(m);
    if (picked.length >= maxLines) break;
  }

  if (picked.length < maxLines && ranked.length) {
    for (const r of ranked) {
      const m = r.m;
      if (!m?.key) continue;
      if (seenKeys.has(m.key)) continue;
      seenKeys.add(m.key);
      picked.push(m);
      if (picked.length >= maxLines) break;
    }
  }

  if (picked.length < maxLines) {
    const fallback = valid
      .slice()
      .sort((a, b) => {
        const ca = clamp01(a.confidence);
        const cb = clamp01(b.confidence);
        if (cb !== ca) return cb - ca;
        return (b.timesSeen || 0) - (a.timesSeen || 0);
      })
      .slice(0, maxLines);

    for (const m of fallback) {
      if (!m?.key) continue;
      if (seenKeys.has(m.key)) continue;
      seenKeys.add(m.key);
      picked.push(m);
      if (picked.length >= maxLines) break;
    }
  }

  const lines = picked.map((m) => {
    const label = confidenceLabel(m.confidence);
    const emo = normalizeEmotion(m.emotion);
    const inten = clampInt(m.intensity ?? 1, 1, 3);
    return `- (${m.category}) [${label}] {${emo}:${inten}} ${m.content}`;
  });

  return dedupeArray(lines, 200).slice(0, maxLines).join("\n");
}

async function saveUserMemory(state, category, text, confidence = 0.85, emotion = "neutral", intensity = 1) {
  ensureMemoryBank(state);

  const content = String(text || "").trim();
  if (!content) return;

  const cat = (category || "other").toLowerCase();
  const conf = clamp01(confidence);
  const key = `${cat}::${normText(content)}`;
  const now = Date.now();

  let item = state.memoryBank.items.find((m) => m.key === key);

  const emo = normalizeEmotion(emotion);
  const inten = clampInt(intensity, 1, 3);

  if (!item) {
    const perm = initialPermanence(cat, content);
    const ttl = ttlDaysFor(perm);
    item = {
      key,
      category: cat,
      content,
      permanence: perm,
      confidence: conf,
      timesSeen: 1,
      createdAt: now,
      lastSeen: now,
      expiresAt: ttl ? now + ttl * 86400000 : null,
      embedding: null,
      emotion: emo,
      intensity: inten,
    };
    state.memoryBank.items.push(item);

    await ensureItemEmbedding(item);
    pruneMemoryBank(state);
    return;
  }

  item.timesSeen = (item.timesSeen || 0) + 1;
  item.lastSeen = now;
  item.confidence = Math.min(0.98, Math.max(item.confidence || 0, conf, (item.confidence || 0) + 0.05));
  item.permanence = maybePromote(item.permanence, item.timesSeen);
  const ttl = ttlDaysFor(item.permanence);
  item.expiresAt = ttl ? now + ttl * 86400000 : null;

  if (!item.emotion || item.emotion === "neutral") {
    item.emotion = emo;
    item.intensity = inten;
  } else if (emo !== "neutral" && inten >= (item.intensity || 1)) {
    item.emotion = emo;
    item.intensity = inten;
  }

  await ensureItemEmbedding(item);
  pruneMemoryBank(state);
}

async function updateSelfModelIfNeeded(state, todayKey) {
  state.selfModel = state.selfModel || {
    updatedAt: 0,
    dayKey: "",
    traits: [],
    doMore: [],
    doLess: [],
    recurringThemes: [],
    calmingTools: [],
  };

  if (state.selfModel.dayKey === todayKey) return;
  if (!openai) {
    state.selfModel.dayKey = todayKey;
    return;
  }

  const prior = {
    traits: Array.isArray(state.selfModel.traits) ? state.selfModel.traits.slice(0, 10) : [],
    doMore: Array.isArray(state.selfModel.doMore) ? state.selfModel.doMore.slice(0, 10) : [],
    doLess: Array.isArray(state.selfModel.doLess) ? state.selfModel.doLess.slice(0, 10) : [],
    recurringThemes: Array.isArray(state.selfModel.recurringThemes) ? state.selfModel.recurringThemes.slice(0, 10) : [],
    calmingTools: Array.isArray(state.selfModel.calmingTools) ? state.selfModel.calmingTools.slice(0, 10) : [],
  };

  const todaySummary =
    typeof state.reflections?.summaryByDay?.[todayKey] === "string"
      ? state.reflections.summaryByDay[todayKey].slice(0, 400)
      : "";

  const sampleMem = (state.memoryBank?.items || [])
    .slice()
    .sort((a, b) => (b.lastSeen || 0) - (a.lastSeen || 0))
    .slice(0, 12)
    .map((m) => ({
      category: m.category,
      confidence: clamp01(m.confidence),
      emotion: normalizeEmotion(m.emotion),
      intensity: clampInt(m.intensity ?? 1, 1, 3),
      content: String(m.content || "").slice(0, 140),
    }));

  const payload = JSON.stringify(
    {
      previousSelfModel: prior,
      todaySummary: todaySummary || null,
      recentMemories: sampleMem,
    },
    null,
    0
  );

  try {
    const resp = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: SELF_MODEL_PROMPT },
        { role: "user", content: payload },
      ],
      temperature: 0,
      max_tokens: 320,
    });

    const raw = resp?.choices?.[0]?.message?.content || "";
    let parsed = null;
    try {
      parsed = JSON.parse(raw);
    } catch {
      parsed = null;
    }

    if (parsed?.update !== true) {
      state.selfModel.dayKey = todayKey;
      return;
    }

    const traits = Array.isArray(parsed.traits) ? parsed.traits : [];
    const doMore = Array.isArray(parsed.doMore) ? parsed.doMore : [];
    const doLess = Array.isArray(parsed.doLess) ? parsed.doLess : [];
    const recurringThemes = Array.isArray(parsed.recurringThemes) ? parsed.recurringThemes : [];
    const calmingTools = Array.isArray(parsed.calmingTools) ? parsed.calmingTools : [];

    const cleanTraits = traits
      .map((t) => ({
        name: String(t?.name || "").trim(),
        confidence: clamp01(t?.confidence ?? 0.6),
      }))
      .filter((t) => t.name.length >= 4)
      .slice(0, 6)
      .sort((a, b) => b.confidence - a.confidence);

    state.selfModel.traits = cleanTraits;
    state.selfModel.doMore = dedupeArray(doMore.map((x) => String(x || "").trim()), 20).slice(0, 5);
    state.selfModel.doLess = dedupeArray(doLess.map((x) => String(x || "").trim()), 20).slice(0, 5);
    state.selfModel.recurringThemes = dedupeArray(recurringThemes.map((x) => String(x || "").trim()), 20).slice(0, 5);
    state.selfModel.calmingTools = dedupeArray(calmingTools.map((x) => String(x || "").trim()), 20).slice(0, 5);
    state.selfModel.updatedAt = Date.now();
    state.selfModel.dayKey = todayKey;
  } catch {
    state.selfModel.dayKey = todayKey;
  }
}

async function runDailyReflectionIfNeeded(state, todayKey) {
  state.reflections = state.reflections || { dayKey: "", summaryByDay: {} };
  state.reflections.summaryByDay = state.reflections.summaryByDay || {};
  if (state.reflections.dayKey === todayKey) return;
  if (!openai) {
    state.reflections.dayKey = todayKey;
    return;
  }

  const recent = Array.isArray(state.history) ? state.history.slice(-16) : [];
  const convo = recent
    .filter((m) => m?.role === "user" || m?.role === "assistant")
    .map((m) => `${m.role.toUpperCase()}: ${String(m.content || "").slice(0, 600)}`)
    .join("\n");

  if (!convo.trim()) {
    state.reflections.dayKey = todayKey;
    return;
  }

  try {
    const resp = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: REFLECTION_PROMPT },
        { role: "user", content: convo },
      ],
      temperature: 0,
      max_tokens: 360,
    });

    const raw = resp?.choices?.[0]?.message?.content || "";
    let parsed = null;
    try {
      parsed = JSON.parse(raw);
    } catch {
      parsed = null;
    }

    if (parsed?.store === true) {
      const memories = Array.isArray(parsed.memories) ? parsed.memories : [];
      for (const m of memories.slice(0, 5)) {
        const cat = (m?.category || "other").toString().trim().toLowerCase();
        const content = String(m?.content || "").trim();
        const conf = clamp01(m?.confidence ?? 0.8);

        let emo = normalizeEmotion(m?.emotion);
        let inten = clampInt(m?.intensity ?? 1, 1, 3);

        if (!emo || emo === "neutral") {
          const tagged = await tagEmotion(content);
          emo = tagged.emotion;
          inten = tagged.intensity;
        }

        if (content && conf >= 0.6) {
          await saveUserMemory(state, cat, content, conf, emo, inten);
        }
      }

      const summary = typeof parsed.summary === "string" ? parsed.summary.trim() : "";
      if (summary) {
        state.reflections.summaryByDay[todayKey] = summary;
        const keys = Object.keys(state.reflections.summaryByDay).sort();
        if (keys.length > 30) {
          for (const k of keys.slice(0, keys.length - 30)) delete state.reflections.summaryByDay[k];
        }
      }
    }

    state.reflections.dayKey = todayKey;
  } catch {
    state.reflections.dayKey = todayKey;
  }
}

async function runQuickMemoryCapture(state, userMessage) {
  const msg = String(userMessage || "").trim().slice(0, 600);
  if (!msg) return;
  if (!openai) return;

  try {
    const memResp = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: QUICK_MEMORY_PROMPT },
        { role: "user", content: msg },
      ],
      temperature: 0,
      max_tokens: 140,
    });

    const raw = memResp?.choices?.[0]?.message?.content || "";
    let parsed = null;
    try {
      parsed = JSON.parse(raw);
    } catch {
      parsed = null;
    }

    if (parsed?.store === true && typeof parsed.content === "string") {
      const content = String(parsed.content || "").trim();
      const category = String(parsed.category || "other").trim().toLowerCase();
      const conf = clamp01(parsed.confidence ?? 0.85);
      const emo = normalizeEmotion(parsed.emotion);
      const inten = clampInt(parsed.intensity ?? 1, 1, 3);

      if (content && conf >= 0.6) {
        await saveUserMemory(state, category, content, conf, emo, inten);
      }
    }
  } catch {}
}

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 120,
  handler: (req, res) => res.json({ reply: "Slow down for a moment, love. Try again in a few seconds." }),
});
app.use(limiter);

app.post("/chat", async (req, res) => {
  const message = (req.body?.message || "").trim();
  if (!message) return res.json({ reply: "I’m listening." });

  if (!openai) {
    return res.json({ reply: "I’m here… but I’m missing my voice right now. Try again in a moment." });
  }

  const anonId = getOrCreateAnonId(req, res);
  const userId = `anon:${anonId}`;

  const state = loadUser(userId);
  pruneMemoryBank(state);

  const todayKey = dayKeyLA();

  const memories = loadAllMemories();
  const userMemoryContext = await getRelevantMemoryLines(state, message, 12);
  const selfModelContext = buildSelfModelContext(state);

  const SYSTEM_PROMPT =
    BASE_SYSTEM_PROMPT +
    (memories ? `\n\nZARA LORE:\n${memories}\n` : "") +
    (selfModelContext ? `\n\nUSER STYLE MODEL (private):\n${selfModelContext}\n` : "") +
    (userMemoryContext ? `\n\nUSER MEMORY (most relevant):\n${userMemoryContext}\n` : "");

  const messages = [{ role: "system", content: SYSTEM_PROMPT }, ...state.history, { role: "user", content: message }];

  let reply = "I’m here with you.";

  try {
    const response = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages,
      temperature: 0.7,
      max_tokens: 260,
    });

    reply = response?.choices?.[0]?.message?.content || reply;
  } catch {
    reply = "I’m here. Take one breath… and say that again for me.";
  }

  reply = sanitizeZaraReply(reply);

  state.history.push({ role: "user", content: message });
  state.history.push({ role: "assistant", content: reply });
  state.history = state.history.slice(-24);

  await runQuickMemoryCapture(state, message);
  await runDailyReflectionIfNeeded(state, todayKey);
  await updateSelfModelIfNeeded(state, todayKey);

  saveUser(userId, state);
  res.json({ reply });
});

app.get("/health", (req, res) => {
  res.status(200).json({
    ok: true,
    hasOpenAIKey: Boolean(OPENAI_API_KEY),
    node: process.version,
  });
});

app.listen(port, () => {
  console.log(`Zara listening on port ${port}`);
});

