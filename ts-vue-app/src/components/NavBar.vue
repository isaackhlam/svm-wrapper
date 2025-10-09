<script setup lang="ts">
  import { ref } from "vue";
  import { useRouter } from 'vue-router';

  const router = useRouter();

  const items = ref([
    {
      label: 'Home',
      command: () => {
        router.push('/')
      }
    },
    {
      label: 'About',
      command: () => {
        router.push('/about')
      }
    },
    {
      label: 'External',
      items: [
        {
          label: 'Vue.js',
          url: 'https://vuejs.org/'
        },
        {
          label: 'Vite.js',
          url: 'https://vitejs.dev/'
        }
      ]
    },
  ]);

</script>

<template>
  <Menubar
    :model="items"
    breakpoint="800px"
  >
    <template #item="{ item, props, hasSubmenu }">
      <router-link
        v-if="item.route"
        v-slot="{ href, navigate }"
        :to="item.route"
        custom
      >
        <a
          v-ripple
          :href="href"
          v-bind="props.action"
          @click="navigate"
        >
          <span :class="item.icon" />
          <span>{{ item.label }}</span>
        </a>
      </router-link>
      <a
        v-else
        v-ripple
        :href="item.url"
        :target="item.target"
        v-bind="props.action"
      >
        <span :class="item.icon" />
        <span>{{ item.label }}</span>
        <span
          v-if="hasSubmenu"
          class="pi pi-fw pi-angle-down"
        />
      </a>
    </template>
  </Menubar>
</template>
