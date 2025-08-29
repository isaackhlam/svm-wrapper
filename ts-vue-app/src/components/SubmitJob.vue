<script setup lang="ts">
import { reactive } from 'vue';
import { useToast } from 'primevue/usetoast';

const toast = useToast();

const initialValues = reactive({
    username: '',
    labelName: 'label',
    svmType: 'C',
    advanceOption: false,
    penalty: 'l2',
    dual: 'auto',
    multiClass: 'ovr',
    fitIntercept: true,
    interceptScaling: 1.0,
    kernel: 'rbf',
    degree: 3,
    gamma: '',
    coef0: 0.0,
    shirnking: true,
    probability: false,
    tol: 1e-3,
    cacheSize: 200,
    classWeight: null,
    verbose: false,
    maxIter: -1,
    decisionFunctionShape: 'ovr',
    breakTies: false,
    randomState: null,
});


const resolver = ({ values }) => {
    const errors = {};

    if (!values.username) {
        errors.username = [{ message: 'Username is required.' }];
    }

    return {
        values, // (Optional) Used to pass current form values to submit event.
        errors
    };
};

const onFormSubmit = ({ valid }) => {
    if (valid) {
        toast.add({
            severity: 'success',
            summary: 'Form is submitted.',
            life: 3000
        });
    }
};
</script>

<template>
  <Card>
  <template #title> Submit Job </template>
  <template #content>
    <div class="card flex justify-center">
        <Toast />

        <Form v-slot="$form" :initialValues :resolver @submit="onFormSubmit" :validateOnValueUpdate="true" class="flex flex-col gap-4 w-full sm:w-56">
            <div class="flex flex-col gap-1">
                <label for="username" class="font-bold block mb-2"> Username </label>
                <InputText name="username" type="text" placeholder="Username" fluid />
                <Message v-if="$form.username?.invalid" severity="error" size="small" variant="simple">{{ $form.username.error?.message }}</Message>
            </div>
            <div class="flex flex-col gap-1">
                <label for="labelName" class="font-bold block mb-2"> Label Name </label>
                <InputText name="labelName" type="text" placeholder="label" fluid />
                <Message v-if="$form.labelName?.invalid" severity="error" size="small" variant="simple">{{ $form.labelName.error?.message }}</Message>
            </div>
            <!-- Radio Button Group -->
            <Fieldset legend="Support Vector Type">
              <RadioButtonGroup name="svmType" class="flex flex-wrap gap-4">
                <div class="flex items-center gap-2">
                  <RadioButton inputId="c" value="C" />
                  <label for="c">C</label>
                </div>
                <div class="flex items-center gap-2">
                  <RadioButton inputId="nu" value="Nu" />
                  <label for="nu">Nu</label>
                </div>
                <div class="flex items-center gap-2">
                  <RadioButton inputId="linear" value="Linear" />
                  <label for="nu">Linear</label>
                </div>
              </RadioButtonGroup>
            </Fieldset>
            <div v-if="$form?.svmType?.value === 'C'">
              <label for="cValue" class="font-bold block mb-2"> C </label>
              <InputNumber name="cValue" fluid />
            </div>
            <div v-if="$form?.svmType?.value === 'Nu'">
              <label for="nuValue" class="font-bold block mb-2"> Nu </label>
              <InputNumber name="nuValue" fluid />
            </div>

            <!-- Advance Option -->
            <div class="flex flex-col items-center gap-2">
              <label for="advanceOption" class="font-bold block mb-2"> Advance Option </label>
              <ToggleSwitch name="advanceOption" />
            </div>
            <div v-show="$form.advanceOption?.value">
              <Fieldset legend="Penalty">
                <RadioButtonGroup name="penalty" class="flex flex-wrap gap-4">
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="l1" value="l1" />
                    <label for="l1">l1</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="l2" value="l2" />
                    <label for="l2">l2</label>
                  </div>
                </RadioButtonGroup>
              </Fieldset>
              <Fieldset legend="Dual">
                <RadioButtonGroup name="dual" class="flex flex-wrap gap-4">
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="auto" value="auto" />
                    <label for="auto">auto</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="true" value="true" />
                    <label for="true">true</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="false" value="false" />
                    <label for="false">false</label>
                  </div>
                </RadioButtonGroup>
              </Fieldset>
              <Fieldset legend="MultiClass">
                <RadioButtonGroup name="multiClass" class="flex flex-wrap gap-4">
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="ovr" value="ovr" />
                    <label for="ovr">ovr</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="crammer_singer" value="crammer_singer" />
                    <label for="crammer_singer">crammer_singer</label>
                  </div>
                </RadioButtonGroup>
              </Fieldset>
              <div class="flex flex-col items-center gap-2">
                <label for="fitIntercept" class="font-bold block mb-2"> Fit Intercept </label>
                <ToggleSwitch name="fitIntercept" />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="interceptScaling" class="font-bold block mb-2"> Intercept Scaling </label>
                <InputNumber name="interceptScaling" fluid />
              </div>
              <Fieldset legend="Kernel">
                <RadioButtonGroup name="kernel" class="flex flex-wrap gap-4">
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="linear" value="linear" />
                    <label for="linear">linear</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="poly" value="poly" />
                    <label for="poly">polynomial</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="rbf" value="rbf" />
                    <label for="rbf">rbf</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="sigmoid" value="sigmoid" />
                    <label for="sigmoid">sigmoid</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="precomputed" value="precomputed" />
                    <label for="precomputed">precomputed</label>
                  </div>
                </RadioButtonGroup>
              </Fieldset>
              <div v-show="$form?.kernel?.value === 'poly'">
                <label for="degree" class="font-bold block mb-2"> Degree </label>
                <InputNumber name="degree" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="gamma" class="font-bold block mb-2"> Gamma </label>
                <InputNumber name="gamma" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="coef0" class="font-bold block mb-2"> coef0 </label>
                <InputNumber name="coef0" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="shinrking" class="font-bold block mb-2"> Shirnking </label>
                <ToggleSwitch name="shirnking" />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="probability" class="font-bold block mb-2"> Probability </label>
                <ToggleSwitch name="probability" />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="tol" class="font-bold block mb-2"> tol </label>
                <InputNumber name="tol"fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="cacheSize" class="font-bold block mb-2"> Cache Size </label>
                <InputNumber name="cacheSize" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="classWeight" class="font-bold block mb-2"> Class Weight </label>
                <InputText name="classWeight" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="verbose" class="font-bold block mb-2"> Verbose </label>
                <ToggleSwitch name="verbose" />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="maxIter" class="font-bold block mb-2"> Max Iteration </label>
                <InputText name="maxIter" fluid />
              </div>
              <Fieldset legend="Decision Function Shape">
                <RadioButtonGroup name="decisionFunctionShape" class="flex flex-wrap gap-4">
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="ovr" value="ovr" />
                    <label for="ovr">ovr</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="ovo" value="ovo" />
                    <label for="ovo">ovo</label>
                  </div>
                </RadioButtonGroup>
              </Fieldset>
              <div class="flex flex-col items-center gap-2">
                <label for="breakTies" class="font-bold block mb-2"> Break Ties </label>
                <ToggleSwitch name="breakTies" />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="randomState" class="font-bold block mb-2"> Random State </label>
                <InputText name="randomState" fluid />
              </div>
            </div>

            <Button type="submit" severity="secondary" label="Submit" />
        </Form>
    </div>
  </template>
  </Card>
</template>

