<script setup lang="ts">
  import { ref, onMounted } from 'vue';
  import { useRoute } from 'vue-router';
  import { GraphQLClient, gql } from 'graphql-request';
  import axios from 'axios';

  const endpoint = import.meta.env.VITE_GRAPHQL_ENDPOINT || "http://localhost:4000/graphql/";
const fileEndpoint = import.meta.env.VITE_FILE_HANDLER_ENDPOINT || "http://localhost:8000/";
  const client = new GraphQLClient(endpoint);
  const route = useRoute();
  const jobId = route.params.jobId as string;

  const jobStatus = ref<string | null>(null);
  const isDownloading = ref(false);

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

  // TODO: Refactor this download logic
  const downloadResult = async () => {
    isDownloading.value = true;

    try {
      const response = await axios.get(`${fileEndpoint}download/${jobId}`,
      {
        reponseType: 'text',
      }
    );
    const csvText = response.data;
    const filename = "output.csv";

    const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' });
    const blobUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(blobUrl);
  } catch (error) {
    console.error('Download failed:', error);
    alert('Failed to download the result file.');
  } finally {
    isDownloading.value = false;
  }
};
</script>

<template>
  <div>
    <p v-if="jobStatus">This is result page, Your job {{ jobId }} is {{ jobStatus }}</p>
    <p v-else>Loading job status...</p>

    <Button
      v-if="jobStatus === 'FINISHED'"
      :label="isDownloading ? 'Downloading...' : 'Download Result'"
      icon="pi pi-download"
      @click="downloadResult"
      class="p-button-success mt-3"
      :disabled="isDownloading"
    />
  </div>
</template>
