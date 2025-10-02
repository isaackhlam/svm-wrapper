<script setup lang="ts">
  import { useRoute } from 'vue-router';
  const route = useRoute();
  const jobId = route.params.jobId;
</script>

<template>
  <p>This is result page, Your job {{ jobId }} is processing</p>
</template>
