<script setup lang="ts">
import { useQueryDemo } from "../composables/useQueryDemo";

const { status, activeScenario, answerMeta, streamingText, errorMessage, focusChunk } = useQueryDemo();
</script>

<template>
  <div class="panel">
    <div class="panel-head">
      <span>Answer</span><span class="meta">{{ answerMeta }}</span>
    </div>
    <div class="panel-body">
      <div class="answer-text">
        <div v-if="errorMessage" class="abstain">{{ errorMessage }}</div>
        <span v-else-if="status === 'idle'" class="status">
          Run a query — answers are generated only from retrieved policy text, with
          inline citations you can click to trace back to the source chunk.
        </span>
        <span v-else-if="status === 'searching'" class="status">
          Embedding query, running hybrid retrieval + RRF fusion…
        </span>
        <template v-else-if="status === 'done' && activeScenario">
          <template v-for="(part, i) in activeScenario.answerParts" :key="i">
            <template v-if="part.text !== undefined">{{ part.text }}</template>
            <span
              v-else
              class="cite"
              :class="{ bad: !activeScenario.chunks[part.ref!].verified }"
              @click="focusChunk(part.ref!)"
            >
              p.{{ activeScenario.chunks[part.ref!].page }} §{{ activeScenario.chunks[part.ref!].section }}
            </span>
          </template>
          <div v-if="activeScenario.abstain" class="abstain">
            ABSTAINED — no retrieved chunk cleared the relevance bar for this query.
          </div>
        </template>
        <template v-else-if="status === 'generating'">
          <span v-if="!streamingText" class="status">Retrieved context — generating answer…</span>
          <template v-else>{{ streamingText }}</template>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
}
.panel-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 11px 16px;
  border-bottom: 1px solid var(--border);
  font-family: "Space Grotesk", sans-serif;
  font-size: 12px;
  font-weight: 600;
}
.panel-head .meta {
  font-family: "JetBrains Mono", monospace;
  font-size: 10.5px;
  font-weight: 400;
  color: var(--dim);
}
.panel-body {
  padding: 16px;
}

.answer-text {
  font-size: 14.5px;
  line-height: 1.65;
  color: #d7dae0;
}
.answer-text .status {
  font-style: italic;
  color: var(--dim);
  font-size: 13px;
}
.cite {
  font-family: "JetBrains Mono", monospace;
  font-size: 11px;
  color: var(--accent);
  background: var(--accent-dim);
  padding: 1px 6px;
  margin: 0 2px;
  cursor: pointer;
  white-space: nowrap;
  border: 1px solid transparent;
}
.cite.bad {
  color: var(--danger);
  background: rgba(240, 102, 92, 0.1);
}
.cite:hover {
  border-color: currentColor;
}
.abstain {
  margin-top: 12px;
  padding: 9px 12px;
  border-left: 2px solid var(--danger);
  background: rgba(240, 102, 92, 0.07);
  font-family: "JetBrains Mono", monospace;
  font-size: 11.5px;
  color: var(--danger);
}
</style>
