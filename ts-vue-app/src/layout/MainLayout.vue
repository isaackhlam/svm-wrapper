<script setup lang="ts">
  import NavBar from '../components/NavBar.vue';
</script>

<template>
  <div>
    <NavBar class=navbar />
    <slot />
  </div>
</template>

<style scoped>
.navbar {
  position: fixed;
  top: 1rem;
  left: 1rem;
  width: 95%;
  z-index: 1000;
}
</style>

