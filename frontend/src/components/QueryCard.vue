<script setup lang="ts">
import { ref } from "vue";
import { useQueryDemo } from "../composables/useQueryDemo";

const { queryText, askDisabled, runQuery } = useQueryDemo();

const localQuery = ref(queryText.value);
const source = ref("any");
// Section labels come from a noisy forward-carried regex heuristic during
// ingestion (see CLAUDE.md) — real values look like "III", "56", "80C",
// "N/A", not a small fixed set, so this is free text rather than a <select>.
const section = ref("");

function submit() {
  const text = localQuery.value.trim();
  if (!text) return;
  runQuery(text, {
    source: source.value !== "any" ? source.value : undefined,
    section: section.value.trim() || undefined,
  });
}

function runExample(text: string) {
  localQuery.value = text;
  runQuery(text, {
    source: source.value !== "any" ? source.value : undefined,
    section: section.value.trim() || undefined,
  });
}
</script>

<template>
  <div class="query-card">
    <div class="q-row">
      <input
        v-model="localQuery"
        type="text"
        placeholder="What is the grace period for premium payment?"
        autocomplete="off"
        @keydown.enter="submit"
      />
      <button class="ask" :disabled="askDisabled" @click="submit">Ask</button>
    </div>
    <div class="q-meta">
      <label>
        Source
        <select v-model="source">
          <option value="any">any</option>
          <option value="icici-car.pdf">icici-car.pdf</option>
          <option value="icici-life.pdf">icici-life.pdf</option>
        </select>
      </label>
      <label>
        Section
        <input v-model="section" type="text" placeholder="e.g. III (optional)" />
      </label>
      <div class="examples">
        try
        <button @click="runExample('What is the grace period for premium payment?')">grace period</button>
        <button @click="runExample('Does this cover flood damage to the garage?')">garage flood</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.query-card {
  background: var(--panel);
  border: 1px solid var(--border);
  padding: 16px 18px;
  margin-bottom: 18px;
}
.q-row {
  display: flex;
  gap: 8px;
}
.q-row input {
  flex: 1;
  background: var(--panel-2);
  border: 1px solid var(--border);
  color: var(--text);
  font-family: "Inter", sans-serif;
  font-size: 14px;
  padding: 10px 12px;
  outline: none;
}
.q-row input::placeholder {
  color: var(--dim-2);
}
.q-row input:focus {
  border-color: var(--accent);
}
.ask {
  background: var(--accent);
  color: #05070c;
  border: none;
  font-family: "Space Grotesk", sans-serif;
  font-weight: 600;
  font-size: 13px;
  padding: 0 20px;
  cursor: pointer;
  transition: opacity 0.15s;
}
.ask:hover {
  opacity: 0.88;
}
.ask:disabled {
  opacity: 0.45;
  cursor: default;
}
.ask:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

.q-meta {
  display: flex;
  gap: 16px;
  margin-top: 11px;
  flex-wrap: wrap;
  align-items: center;
}
.q-meta label {
  font-size: 10.5px;
  color: var(--dim);
  font-family: "JetBrains Mono", monospace;
  display: flex;
  align-items: center;
  gap: 6px;
}
.q-meta select,
.q-meta input[type="text"] {
  background: var(--panel-2);
  border: 1px solid var(--border);
  color: var(--text);
  font-family: "JetBrains Mono", monospace;
  font-size: 11px;
  padding: 3px 6px;
  outline: none;
}
.q-meta input[type="text"] {
  width: 110px;
}
.q-meta input[type="text"]::placeholder {
  color: var(--dim-2);
}
.q-meta input[type="text"]:focus {
  border-color: var(--accent);
}
.examples {
  font-size: 11px;
  color: var(--dim-2);
  margin-left: auto;
}
.examples button {
  background: none;
  border: 1px solid var(--border-soft);
  color: var(--dim);
  font-family: "JetBrains Mono", monospace;
  font-size: 10.5px;
  padding: 4px 9px;
  cursor: pointer;
  margin-left: 6px;
}
.examples button:hover {
  border-color: var(--accent);
  color: var(--accent);
}
</style>
