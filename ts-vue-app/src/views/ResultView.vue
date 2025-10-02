<script setup lang="ts">
  import { ref, onMounted } from 'vue';
  import { useRoute } from 'vue-router';
  import { GraphQLClient, gql } from 'graphql-request';

  const endpoint = import.meta.env.VITE_GRAPHQL_ENDPOINT || "http://localhost:4000/graphql/";
  const client = new GraphQLClient(endpoint);
  const route = useRoute();
  const jobId = route.params.jobId as string;

  const jobStatus = ref<string | null>(null);

  // TODO: Separete import for qgl query
  const query = gql`
    query GetJobStatus($input: JobStatusInput!) {
      getJobStatus(input: $input)
    }
  `
  onMounted(async () => {
    try {
      const result = await client.request(query, { input: { id: jobId} });
      jobStatus.value = result.getJobStatus;
    } catch (error) {
      console.error("Failed to fetch job status", error);
      jobStatus.value = "Error";
    }
  });
</script>

<template>
  <p v-if="jobStatus">This is result page, Your job {{ jobId }} is {{ jobStatus }}</p>
  <p v-else>Loading job status...</p>
</template>
