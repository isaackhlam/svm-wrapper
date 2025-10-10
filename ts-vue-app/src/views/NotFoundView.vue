<template>
  <div class="flex items-center justify-center min-h-screen bg-surface-100">
    <Card class="w-full max-w-md shadow-2xl rounded-2xl text-center p-6">
      <template #title>
        <h1 class="text-4xl font-bold text-red-500">
          404
        </h1>
      </template>

      <template #subtitle>
        <p class="text-lg text-gray-600">
          Page Not Found
        </p>
      </template>

      <template #content>
        <p class="mb-6 text-gray-500">
          The page you are looking for does not exist.  
          Please check the link or navigate using the buttons below.
        </p>
      </template>

      <template #footer>
        <div class="flex justify-center gap-4">
          <Button
            label="Previous Page"
            icon="pi pi-arrow-left"
            severity="secondary"
            @click="goBack"
          />
          <RouterLink to="/">
            <Button
              label="Home Page"
              icon="pi pi-home"
            />
          </RouterLink>
        </div>
      </template>
    </Card>
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'
import Card from 'primevue/card'
import Button from 'primevue/button'
import { RouterLink } from 'vue-router'

const router = useRouter()

const goBack = () => {
  router.back()
}
</script>

