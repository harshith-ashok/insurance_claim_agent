<script setup lang="ts"></script>

<template>
  <div class="diag">
    <div class="diag-item">
      <span class="k">Confidence gate</span>
      <div class="v warn">disabled</div>
    </div>
    <div class="diag-item">
      <span class="k">Faithfulness (RAGAS)</span>
      <div class="v">0.825</div>
    </div>
    <div class="diag-item">
      <span class="k">Recall@4</span>
      <div class="v">88.8%</div>
    </div>
    <div class="diag-item">
      <span class="k">Citation verify</span>
      <div class="v">86.8%</div>
    </div>
    <div class="diag-item">
      <span class="k">Golden set</span>
      <div class="v" style="font-size: 12px">53q · 1 hallucination</div>
    </div>
  </div>
</template>

<style scoped>
.diag {
  margin-top: 16px;
  display: flex;
  flex-wrap: wrap;
  border: 1px solid var(--border);
  background: var(--panel);
}
.diag-item {
  padding: 11px 16px;
  border-right: 1px solid var(--border-soft);
  flex: 1;
  min-width: 130px;
}
.diag-item:last-child {
  border-right: none;
}
.diag-item .k {
  font-family: "JetBrains Mono", monospace;
  font-size: 9.5px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--dim-2);
}
.diag-item .v {
  font-family: "Space Grotesk", sans-serif;
  font-size: 15px;
  font-weight: 600;
  margin-top: 3px;
}
.diag-item .v.warn {
  color: var(--warn);
}
</style>
