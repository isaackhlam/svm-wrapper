<script setup lang="ts">
  import SVMClassification from './SVMClassification.vue';
  import SVMRegression from './SVMRegression.vue';
</script>

<template>
  <Tabs value="SVM">
    <TabList>
      <Tab value="SVM">SVM</Tab>
      <Tab value="DNN">DNN</Tab>
    </TabList>
    <TabPanels>
      <TabPanel value="SVM">
        <SVMRegression />
      </TabPanel>
      <TabPanel value="DNN">
        <!-- Temp check, CLS vs REG will be in second menu. And Card should be used in this component -->
        <SVMClassification />
      </TabPanel>
    </TabPanels>
  </Tabs>
</template>
