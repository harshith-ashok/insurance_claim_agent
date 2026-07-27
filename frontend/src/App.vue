<script setup lang="ts">
import AppSidebar from "./components/AppSidebar.vue";
import AppTopbar from "./components/AppTopbar.vue";
import QueryCard from "./components/QueryCard.vue";
import AnswerPanel from "./components/AnswerPanel.vue";
import TracePanel from "./components/TracePanel.vue";
import DiagStrip from "./components/DiagStrip.vue";
</script>

<template>
  <div class="shell">
    <AppSidebar />
    <div class="main">
      <AppTopbar />
      <div class="content">
        <QueryCard />
        <div class="results">
          <AnswerPanel />
          <TracePanel />
        </div>
        <DiagStrip />
      </div>
    </div>
  </div>
</template>

<style scoped>
.shell {
  display: grid;
  grid-template-columns: 210px 1fr;
  min-height: 100vh;
}
@media (max-width: 780px) {
  .shell {
    grid-template-columns: 1fr;
  }
}

.main {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.content {
  padding: 20px 22px 40px;
  flex: 1;
}

.results {
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 16px;
  align-items: start;
}
@media (max-width: 960px) {
  .results {
    grid-template-columns: 1fr;
  }
}
</style>
