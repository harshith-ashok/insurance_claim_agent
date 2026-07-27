<script setup lang="ts"></script>

<template>
  <aside class="sidebar">
    <div class="brand">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <rect x="2" y="2" width="20" height="20" stroke="#5B8CFF" stroke-width="1.6" />
        <path d="M7 8h10M7 12h10M7 16h6" stroke="#5B8CFF" stroke-width="1.6" stroke-linecap="round" />
      </svg>
      <div>
        <div class="n">Policy Ledger</div>
        <span class="s">insurance-rag</span>
      </div>
    </div>

    <div class="navgroup">
      <div class="navlabel">Pipeline</div>
      <div class="navitem active">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          <circle cx="11" cy="11" r="7" stroke="currentColor" stroke-width="1.8" />
          <path d="M20 20l-4-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
        </svg>
        Query
      </div>
      <div class="navitem disabled" title="Not wired in this demo">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          <path d="M4 7h16M4 12h16M4 17h10" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
        </svg>
        Ingestion
      </div>
      <div class="navitem disabled" title="Not wired in this demo">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          <path
            d="M4 19V5M4 19h16M8 15l3-4 3 3 4-6"
            stroke="currentColor"
            stroke-width="1.8"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
        Evaluation
      </div>
    </div>

    <div class="sidebar-foot">
      collection<br /><b>insurance_policies</b><br /><br />
      sources<br /><b>icici-car.pdf<br />icici-life.pdf</b>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  background: var(--panel);
  border-right: 1px solid var(--border);
  padding: 20px 14px;
  display: flex;
  flex-direction: column;
}
@media (max-width: 780px) {
  .sidebar {
    display: none;
  }
}

.brand {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 0 6px 22px;
}
.brand .n {
  font-family: "Space Grotesk", sans-serif;
  font-weight: 600;
  font-size: 14.5px;
  letter-spacing: -0.01em;
}
.brand .s {
  font-size: 9.5px;
  color: var(--dim-2);
  font-family: "JetBrains Mono", monospace;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  display: block;
  margin-top: 1px;
}

.navgroup {
  margin-bottom: 18px;
}
.navlabel {
  font-size: 9.5px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--dim-2);
  font-family: "JetBrains Mono", monospace;
  padding: 0 10px;
  margin-bottom: 6px;
}
.navitem {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 7px 10px;
  font-size: 12.5px;
  color: var(--dim);
  cursor: default;
  border-left: 2px solid transparent;
}
.navitem svg {
  opacity: 0.7;
  flex-shrink: 0;
}
.navitem.active {
  color: var(--text);
  background: var(--accent-dim);
  border-left-color: var(--accent);
}
.navitem.active svg {
  opacity: 1;
  stroke: var(--accent);
}
.navitem.disabled {
  opacity: 0.4;
}

.sidebar-foot {
  margin-top: auto;
  padding: 10px;
  font-size: 10.5px;
  color: var(--dim-2);
  font-family: "JetBrains Mono", monospace;
  line-height: 1.6;
}
.sidebar-foot b {
  color: var(--dim);
  font-weight: 500;
}
</style>
