<script setup lang="ts"></script>

<template>
  <div class="topbar">
    <div class="crumb">policy-ledger / <b>query</b></div>
    <div class="pills">
      <div class="pill"><span class="dot"></span>qdrant</div>
      <div class="pill"><span class="dot"></span>ollama</div>
    </div>
  </div>
</template>

<style scoped>
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 13px 22px;
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
  gap: 8px;
}
.crumb {
  font-size: 12px;
  color: var(--dim);
  font-family: "JetBrains Mono", monospace;
}
.crumb b {
  color: var(--text);
  font-weight: 500;
}
.pills {
  display: flex;
  gap: 8px;
}
.pill {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 10.5px;
  font-family: "JetBrains Mono", monospace;
  padding: 4px 9px;
  border: 1px solid var(--border);
  color: var(--dim);
  background: var(--panel-2);
}
.pill .dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--ok);
}
</style>
