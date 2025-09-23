<script setup lang="ts">
  import SVMClassification from './SVMClassification.vue';
  import SVMRegression from './SVMRegression.vue';
  import { ref } from 'vue';
  const modelMode = ref('regression');
</script>

<template>
  <Card>
    <template #title> Submit Job </template>
    <template #content>
      <Tabs value="SVM">
        <TabList>
          <Tab value="SVM">SVM</Tab>
          <Tab value="DNN">DNN</Tab>
        </TabList>
        <TabPanels>
          <TabPanel value="SVM">
            <Fieldset legend="Select Model Mode" >
              <RadioButtonGroup class="flex flex-wrap gap-4">
                <div class="flex items-center gap-2">
                  <RadioButton v-model="modelMode" inputId="mode1" value="regression" />
                  <label for="mode1">Regression</label>
                </div>
                <div class="flex items-center gap-2">
                  <RadioButton v-model="modelMode" inputId="mode2" value="classification" />
                  <label for="mode2">Classification</label>
                </div>
              </RadioButtonGroup>
            </Fieldset>
            <div v-show="modelMode === 'regression'">
              <SVMRegression />
            </div>
            <div v-show="modelMode === 'classification'">
              <SVMRegression />
            </div>
          </TabPanel>
          <TabPanel value="DNN">
            <!-- Temp check, CLS vs REG will be in second menu. And Card should be used in this component -->
            <SVMClassification />
          </TabPanel>
        </TabPanels>
      </Tabs>
    </template>
  </Card>
</template>
