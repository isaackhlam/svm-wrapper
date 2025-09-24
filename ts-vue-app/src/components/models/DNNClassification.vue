<script setup lang="ts">
import { reactive } from 'vue';
import { useToast } from 'primevue/usetoast';

const toast = useToast();

const initialValues = reactive({
    username: '',
    labelName: 'label',
    advanceOption: false,
    hiddenLayerSizes: '(100,)',
    activation: 'relu',
    solver: 'adam',
    alpha: 0.0001,
    batchSize: 'auto',
    learningRate: 'constant',
    learningRateInit: 0.001,
    powerT: 0.5,
    maxIter: 200,
    randomState: '',
    tol: 1e-4,
    verbose: false,
    warmStart: false,
    momentum: 0.9,
    nesterovsMomentum: true,
    earlyStopping: false,
    validationFraction: 0.1,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    nIterNoChange: 10,
    maxFun: 15000,
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

            <!-- Advance Option -->
            <div class="flex flex-col items-center gap-2">
              <label for="advanceOption" class="font-bold block mb-2"> Advance Option </label>
              <ToggleSwitch name="advanceOption" />
            </div>
            <div v-show="$form.advanceOption?.value">
              <!-- Consider a better input method and parse for this -->
              <div class="flex flex-col items-center gap-2">
                <label for="hiddenLayerSizes" class="font-bold block mb-2"> Hidden Layer Sizes </label>
                <InputText name="hiddenLayerSizes" fluid />
              </div>
              <Fieldset legend="Activation">
                <RadioButtonGroup name="activation" class="flex flex-wrap gap-4">
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="relu" value="relu" />
                    <label for="relu">relu</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="tanh" value="tanh" />
                    <label for="tanh">tanh</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="logistic" value="logistic" />
                    <label for="logistic">logistic</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="identity" value="identity" />
                    <label for="identity">identity</label>
                  </div>
                </RadioButtonGroup>
              </Fieldset>
              <Fieldset legend="Solver">
                <RadioButtonGroup name="solver" class="flex flex-wrap gap-4">
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="lbfgs" value="lbfgs" />
                    <label for="lbfgs">lbfgs</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="sgd" value="sgd" />
                    <label for="sgd">sgd</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="adam" value="adam" />
                    <label for="adam">adam</label>
                  </div>
                </RadioButtonGroup>
              </Fieldset>
              <div class="flex flex-col items-center gap-2">
                <label for="alpha" class="font-bold block mb-2"> Alpha </label>
                <InputNumber name="alpha" fluid />
              </div>
              <Fieldset legend="Learning Rate">
                <RadioButtonGroup name="learningRate" class="flex flex-wrap gap-4">
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="constant" value="constant" />
                    <label for="constant">constant</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="invscaling" value="invscaling" />
                    <label for="invscaling">invscaling</label>
                  </div>
                  <div class="flex items-center gap-2">
                    <RadioButton inputId="adaptive" value="adaptive" />
                    <label for="adaptive">adaptive</label>
                  </div>
                </RadioButtonGroup>
              </Fieldset>
              <div class="flex flex-col items-center gap-2">
                <label for="learningRateInit" class="font-bold block mb-2"> Learning Rate Init </label>
                <InputNumber name="learningRateInit" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="powerT" class="font-bold block mb-2"> Power t </label>
                <InputNumber name="powerT" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="maxIter" class="font-bold block mb-2"> Max iter </label>
                <InputNumber name="maxIter" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="randomState" class="font-bold block mb-2"> Random state </label>
                <InputText name="randomState" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="tol" class="font-bold block mb-2"> Tolerance </label>
                <InputNumber name="tol" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="verbose" class="font-bold block mb-2"> Verbose </label>
                <ToggleSwitch name="verbose" />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="warmStart" class="font-bold block mb-2"> Warm start </label>
                <ToggleSwitch name="warmStart" />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="momentum" class="font-bold block mb-2"> Momentum </label>
                <InputNumber name="momentum" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="nesterovsMomentum" class="font-bold block mb-2"> Nesterovs Momentum </label>
                <ToggleSwitch name="nesterovsMomentum" />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="earlyStopping" class="font-bold block mb-2"> Early Stopping </label>
                <ToggleSwitch name="earlyStopping" />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="validationFraction" class="font-bold block mb-2"> Validation Fraction </label>
                <InputNumber name="validationFraction" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="beta1" class="font-bold block mb-2"> beta 1 </label>
                <InputNumber name="beta1" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="beta2" class="font-bold block mb-2"> beta 2 </label>
                <InputNumber name="beta2" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="epsilon" class="font-bold block mb-2"> epsilon </label>
                <InputNumber name="epsilon" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="nIterNoChange" class="font-bold block mb-2"> n Iter No Change </label>
                <InputNumber name="nIterNoChange" fluid />
              </div>
              <div class="flex flex-col items-center gap-2">
                <label for="maxFun" class="font-bold block mb-2"> Max Fun </label>
                <InputNumber name="maxFun" fluid />
              </div>

            </div>
            <Button type="submit" severity="secondary" label="Submit" />
        </Form>
    </div>
</template>

