<script setup lang="ts">
  import SVMClassification from './SVMClassification.vue';
  import SVMRegression from './SVMRegression.vue';
  import DNNClassification from './DNNClassification.vue';
  import DNNRegression from './DNNRegression.vue';
  import { ref } from 'vue';
  const SVMMode = ref('regression');
  const DNNMode = ref('regression');
</script>

<template>
  <Card>
    <template #title>
      Submit Job
    </template>
    <template #content>
      <Tabs value="SVM">
        <TabList>
          <Tab value="SVM">
            SVM
          </Tab>
          <Tab value="DNN">
            DNN
          </Tab>
        </TabList>
        <TabPanels>
          <TabPanel value="SVM">
            <Fieldset
              legend="Select Model Mode"
              class="flex"
            >
              <div class="flex flex-wrap gap-4">
                <div class="flex items-center gap-2">
                  <RadioButton
                    v-model="SVMMode"
                    input-id="svm-mode1"
                    value="regression"
                  />
                  <label for="svm-mode1">Regression</label>
                </div>
                <div class="flex items-center gap-2">
                  <RadioButton
                    v-model="SVMMode"
                    input-id="svm-mode2"
                    value="classification"
                  />
                  <label for="svm-mode2">Classification</label>
                </div>
              </div>
            </Fieldset>
            <div v-show="SVMMode === 'regression'">
              <SVMRegression />
            </div>
            <div v-show="SVMMode === 'classification'">
              <SVMClassification />
            </div>
          </TabPanel>
          <TabPanel value="DNN">
            <!-- Temp check, CLS vs REG will be in second menu. And Card should be used in this component -->
            <Fieldset
              legend="Select Model Mode"
              class="flex"
            >
              <div class="flex flex-wrap gap-4">
                <div class="flex items-center gap-2">
                  <RadioButton
                    v-model="DNNMode"
                    input-id="dnn-mode1"
                    value="regression"
                  />
                  <label for="dnn-mode1">Regression</label>
                </div>
                <div class="flex items-center gap-2">
                  <RadioButton
                    v-model="DNNMode"
                    input-id="dnn-mode2"
                    value="classification"
                  />
                  <label for="dnn-mode2">Classification</label>
                </div>
              </div>
            </Fieldset>
            <div v-show="DNNMode === 'regression'">
              <DNNRegression />
            </div>
            <div v-show="DNNMode === 'classification'">
              <DNNClassification />
            </div>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </template>
  </Card>
</template>
