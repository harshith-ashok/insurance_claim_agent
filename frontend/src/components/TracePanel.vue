<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { useQueryDemo } from "../composables/useQueryDemo";

const { status, activeScenario, activeChunk, chunkMeta } = useQueryDemo();

const animated = ref(false);
const chunkEls = ref<(HTMLElement | null)[]>([]);

const traceSegments = computed(() => {
  if (!activeScenario.value) return [];
  const t = activeScenario.value.trace;
  const total = t.embed + t.hybrid + t.rerank + t.gen;
  return [
    { cls: "embed", pct: (t.embed / total) * 100, label: `embed ${t.embed}ms` },
    { cls: "hybrid", pct: (t.hybrid / total) * 100, label: `hybrid+rrf ${t.hybrid}ms` },
    { cls: "rerank", pct: (t.rerank / total) * 100, label: `rerank ${t.rerank}ms` },
    { cls: "gen", pct: (t.gen / total) * 100, label: `generate ${t.gen}ms` },
  ];
});

watch(activeScenario, async (scenario) => {
  if (scenario) {
    animated.value = false;
    await nextTick();
    requestAnimationFrame(() => {
      animated.value = true;
    });
  } else {
    animated.value = false;
  }
});

watch(activeChunk, async (i) => {
  if (i === null) return;
  await nextTick();
  chunkEls.value[i]?.scrollIntoView({ block: "nearest" });
});
</script>

<template>
  <div class="panel">
    <div class="panel-head">
      <span>Retrieval trace</span><span class="meta">{{ chunkMeta }}</span>
    </div>
    <div class="panel-body">
      <div v-if="activeScenario" class="trace">
        <div class="trace-bar">
          <div
            v-for="seg in traceSegments"
            :key="seg.cls"
            class="trace-seg"
            :class="seg.cls"
            :style="{ width: animated ? seg.pct + '%' : '0%' }"
            :title="seg.label"
          ></div>
        </div>
        <div class="trace-legend">
          <span><i style="background: #3e5ca3"></i>embed</span>
          <span><i style="background: #5b8cff"></i>hybrid+rrf</span>
          <span><i style="background: #8c6fe0"></i>rerank</span>
          <span><i style="background: #37c978"></i>generate (first token)</span>
        </div>
      </div>

      <div>
        <span
          v-if="!activeScenario"
          class="status-row"
          style="color: var(--dim); font-style: italic"
        >
          {{ status === "idle" ? "No query run yet." : "Retrieving…" }}
        </span>
        <div
          v-for="(chunk, i) in activeScenario?.chunks ?? []"
          :key="i"
          :ref="(el) => (chunkEls[i] = el as HTMLElement)"
          class="chunk"
          :class="{ hit: activeChunk === i }"
        >
          <div class="chunk-top">
            <span class="chunk-src"><b>{{ chunk.src }}</b> · p.{{ chunk.page }} · §{{ chunk.section }}</span>
            <span class="chunk-score">{{ chunk.score.toFixed(2) }}</span>
          </div>
          <div class="bar"><i :style="{ width: animated ? Math.round(chunk.score * 100) + '%' : '0%' }"></i></div>
          <div class="chunk-text">{{ chunk.text }}</div>
          <div class="status-row" :class="chunk.verified ? 'ok' : 'bad'">
            <span class="d"></span>{{ chunk.verified ? "verified" : "unverified" }}
          </div>
        </div>
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

.trace {
  margin-bottom: 14px;
}
.trace-bar {
  display: flex;
  height: 22px;
  width: 100%;
  overflow: hidden;
  background: var(--panel-2);
  border: 1px solid var(--border-soft);
}
.trace-seg {
  height: 100%;
  transition: width 0.5s ease;
  position: relative;
}
.trace-seg.embed {
  background: #3e5ca3;
}
.trace-seg.hybrid {
  background: #5b8cff;
}
.trace-seg.rerank {
  background: #8c6fe0;
}
.trace-seg.gen {
  background: #37c978;
}
.trace-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 8px;
  font-size: 10px;
  font-family: "JetBrains Mono", monospace;
  color: var(--dim);
}
.trace-legend span {
  display: flex;
  align-items: center;
  gap: 5px;
}
.trace-legend i {
  width: 8px;
  height: 8px;
  display: inline-block;
}

.chunk {
  border-bottom: 1px solid var(--border-soft);
  padding: 10px 0;
}
.chunk:last-child {
  border-bottom: none;
}
.chunk-top {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
}
.chunk-src {
  font-family: "JetBrains Mono", monospace;
  font-size: 11px;
  color: var(--dim);
}
.chunk-src b {
  color: var(--text);
  font-weight: 500;
}
.chunk-score {
  font-family: "JetBrains Mono", monospace;
  font-size: 11px;
  color: var(--dim);
}
.bar {
  height: 3px;
  background: var(--border-soft);
  margin: 6px 0;
}
.bar i {
  display: block;
  height: 100%;
  background: var(--accent);
  width: 0;
  transition: width 0.5s ease 0.15s;
}
.chunk-text {
  font-size: 12px;
  color: #aeb3bc;
  line-height: 1.5;
}
.status-row {
  display: flex;
  align-items: center;
  gap: 5px;
  margin-top: 5px;
  font-family: "JetBrains Mono", monospace;
  font-size: 10px;
}
.status-row .d {
  width: 6px;
  height: 6px;
  border-radius: 50%;
}
.status-row.ok .d {
  background: var(--ok);
}
.status-row.ok {
  color: var(--ok);
}
.status-row.bad .d {
  background: var(--danger);
}
.status-row.bad {
  color: var(--danger);
}
.chunk.hit {
  background: rgba(91, 140, 255, 0.06);
  margin: 0 -16px;
  padding: 10px 16px;
}
</style>
