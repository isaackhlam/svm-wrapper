<script setup lang="ts">
import { reactive } from 'vue';
import { useRouter } from 'vue-router';
import { useToast } from 'primevue/usetoast';
import { submitJobMutation } from './query.ts';
import { GraphQLClient } from 'graphql-request';
import { v4 as uuidv4 } from 'uuid';
import axios from "axios";

const endpoint = import.meta.env.VITE_GRAPHQL_ENDPOINT || "http://localhost:4000/graphql/";
const fileEndpoint = import.meta.env.VITE_FILE_HANDLER_ENDPOINT || "http://localhost:8000/";
const client = new GraphQLClient(endpoint);
const router = useRouter();

const toast = useToast();
const jobId = uuidv4();

const initialValues = reactive({
    jobId,
    labelName: 'label',
    explainModel: false,
    svmType: 'C',
    cValue: 1,
    nuValue: 0.5,
    advanceOption: false,
    dual: 'auto',
    loss: 'epsilon_insensitive',
    fitIntercept: true,
    interceptScaling: 1.0,
    kernel: 'rbf',
    degree: 3,
    gamma: 'scale',
    coef0: 0.0,
    shrinking: true,
    tol: 1e-3,
    tolLinear: 1e-4,
    cacheSize: 200,
    maxIter: -1,
    maxIterLinear: 1000,
    randomState: null,
    epsilon: '0.1',
});

const resolver = ({ values }) => {
    const errors = {};

    if (typeof(values.explainModel) !== "boolean") {
      errors.explainModel = [{ message: "Value of explain model must be true or false." }];
    }

    if (values.svmType !== "C" && values.svmType !== "Nu" && values.svmType !== "Linear") {
      errors.svmType = [{ message: "svmType must be one of 'C', 'Nu', or 'Linear'." }];
    }

    if (values.cValue <= 0.0) {
      errors.cValue = [{ message: "Value of C must be strictly positive." }];
    }

    if (values.svmType === "Nu" && (values.nuValue <= 0.0 || values.nuValue > 1.0)) {
      errors.nuValue = [{ message: "Value of nu must be in (0, 1]." }]
    }

    if (values.dual === "true") {
      values.dual = true;
    } else if (values.dual === "false") {
      values.dual = false;
    } else if (values.dual !== "auto") {
      errors.dual = [{ message: "Value of dual must be 'auto', 'true', or 'false'." }];
    }

    if (values.loss !== "epsilon_insensitive" && values.loss !== "squared_epsilon_insensitive") {
      errors.loss = [{ message: "Value of loss must be 'epsilon_insensitive' or 'squared_epsilon_insensitive'." }];
    }

    if (typeof(values.fitIntercept) !== "boolean") {
      errors.fitIntercept = [{ message: "Value of Fit Intercept must be true or false." }];
    }

    if (typeof(values.interceptScaling) !== "number") {
      errors.interceptScaling = [{ message: "Value of Intercept Scaling must be number" }];
    }

    if (!["rbf", "linear", "poly", "sigmoid", "precomputed"].includes(values.kernel)) {
      errors.kernel = [{ message: "Value of kernel must be one of 'rbf', 'linear', 'poly', 'sigmoid', or 'precomputed'."}];
    }

    if (values.degree < 0) {
      errors.degree = [{ message: "Value of degree must be non-negative." }];
    }

    if (["auto", "scale"].includes(values.gamma)) {
      values.gamma = values.gamma;
    } else {
      const num = Number(values.gamma);
      if (Number.isNaN(num)) {
        errors.gamma = [{ message: "Value of gamma must be 'scale', 'auto' or non-negative number" }];
      } else if (num < 0) {
        errors.gamma = [{ message: "Value of gamma must be non-negative." }];
      } else {
      values.gamma = num;
      }
    }

    if (Number.isNaN(values.coef0)) {
      errors.coef0 = [{ message: "Value of coef0 must be a number." }];
    }

    if (typeof(values.shrinking) !== "boolean") {
      errors.shrinking = [{ message: "Value of shirnking must be true or false." }];
    }

    if (values.svmType === "Linear" && !Number.isNaN(values.tolLinear)) {
      values.tol = values.tolLinear;
    } else if (values.svmType !== "Linear" && Number.isNaN(values.tol)) {
      errors.tol = [{ message: "Value of tol must be a number." }];
    } else if (values.svmType === "Linear" && Number.isNaN(values.tolLinear)) {
      errors.tolLinear = [{ message: "Value of tol must be a number." }];
    }

    if (!Number.isInteger(values.cacheSize)) {
      errors.cacheSize = [{ message: "Value of cache size must be an integer." }];
    }

    if (values.svmType === "Linear" && Number.isInteger(values.maxIterLinear)) {
      values.maxIter = values.maxIterLinear;
    } else if (values.svmType !== "Linear" && !Number.isInteger(values.maxIter)) {
      errors.maxIter = [{ message: "Value of max iter must be an integer." }];
    } else if (values.svmType === "Linear" && !Number.isInteger(values.maxIterLinear)) {
      errors.maxIterLinear = [{ message: "Value of max iter must be an integer." }];
    }

    if (Number.isNaN(values.epsilon)) {
      errors.epsilon = [{ message: "Value of epsilon must be a number." }];
    }

    if (values.randomState !== null && !Number.isInteger(values.randomState)) {
      errors.randomState = [{ message: "Value of randomState must be empty or integer." }];
    }

    return {
        values,
        errors
    };
};

const onFormSubmit = async (form) => {
    const valid = form.valid;
    if (valid) {
        toast.add({
            severity: 'success',
            summary: 'Form is submitted.',
            life: 3000
        });
    client.request(submitJobMutation, { input: {modelType: "SVM", taskType: "REGRESSION", id: jobId, hyperparameters: form.values}});
    router.push(`/result/${jobId}`);
    }
    // console.log(`isValid: ${valid}\n Form: ${JSON.stringify(form.values)}`);
    // console.log(`Error: ${JSON.stringify(form.errors)}`);
};

// TODO: I think it could be merge as one function...
const onTrainFileUpload = async (event) => {
  console.log(event);
  const formData = new FormData();
  formData.append("file", event.files[0]);
  formData.append("key", `${jobId}/train.csv`);

  try {
    const res = await axios.post(`${fileEndpoint}upload`, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    console.log("Upload success", res.data);
  } catch (err) {
    console.error("Upload failed:", err);
  }
}
const onTestFileUpload = async (event) => {
  console.log(event);
  const formData = new FormData();
  formData.append("file", event.files[0]);
  formData.append("key", `${jobId}/test.csv`);

  try {
    const res = await axios.post(`${fileEndpoint}upload`, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    console.log("Upload success", res.data);
  } catch (err) {
    console.error("Upload failed:", err);
  }
}
</script>

<template>
  <div class="card flex justify-center">
    <Toast />

    <Form
      v-slot="$form"
      :initial-values
      :resolver
      :validate-on-value-update="true"
      class="flex flex-col gap-4 w-full sm:w-56"
      @submit="onFormSubmit"
    >
      <div class="flex flex-col gap-1">
        <label
          for="jobId"
          class="font-bold block mb-2"
        > Job ID </label>
        <InputText
          name="jobId"
          type="text"
          fluid
          disabled
        />
      </div>
      <div class="flex flex-col gap-1">
        <label
          for="labelName"
          class="font-bold block mb-2"
        > Label Name </label>
        <InputText
          name="labelName"
          type="text"
          placeholder="label"
          fluid
        />
        <Message
          v-if="$form.labelName?.invalid"
          severity="error"
          size="small"
          variant="simple"
        >
          {{ $form.labelName.error?.message }}
        </Message>
      </div>
      <div>
        <label
          for="trainFile"
          class="font-bold block mb-2"
        > Train File </label>
        <FileUpload
          name="trainFile"
          :custom-upload="true"
          accept=".csv"
          :max-file-size="1_000_000"
          @uploader="onTrainFileUpload"
        />
      </div>
      <div>
        <label
          for="testFile"
          class="font-bold block mb-2"
        > Test File </label>
        <FileUpload
          name="testFile"
          :custom-upload="true"
          accept=".csv"
          :max-file-size="1_000_000"
          @uploader="onTestFileUpload"
        />
      </div>
      <div class="flex flex-col items-center gap-2">
        <label
          for="explainModel"
          class="font-bold block mb-2"
        > Explain Model (SHAP)</label>
        <ToggleSwitch name="explainModel" />
      </div>

      <!-- Radio Button Group -->
      <Fieldset legend="Support Vector Type">
        <RadioButtonGroup
          name="svmType"
          class="flex flex-wrap gap-4"
        >
          <div class="flex items-center gap-2">
            <RadioButton
              input-id="c"
              value="C"
            />
            <label for="c">C</label>
          </div>
          <div class="flex items-center gap-2">
            <RadioButton
              input-id="nu"
              value="Nu"
            />
            <label for="nu">Nu</label>
          </div>
          <div class="flex items-center gap-2">
            <RadioButton
              input-id="linear"
              value="Linear"
            />
            <label for="nu">Linear</label>
          </div>
        </RadioButtonGroup>
      </Fieldset>
      <div>
        <label
          for="cValue"
          class="font-bold block mb-2"
        > C </label>
        <InputNumber
          name="cValue"
          fluid
        />
      </div>
      <div v-show="$form?.svmType?.value === 'Nu'">
        <label
          for="nuValue"
          class="font-bold block mb-2"
        > Nu </label>
        <InputNumber
          name="nuValue"
          fluid
        />
      </div>

      <!-- Advance Option -->
      <div class="flex flex-col items-center gap-2">
        <label
          for="advanceOption"
          class="font-bold block mb-2"
        > Advance Option </label>
        <ToggleSwitch name="advanceOption" />
      </div>
      <div v-show="$form.advanceOption?.value">
        <div v-show="$form.svmType === 'C'">
          <div class="flex flex-col items-center gap-2">
            <label
              for="epsilon"
              class="font-bold block mb-2"
            > Epsilon </label>
            <InputText
              name="epsilon"
              fluid
            />
          </div>
        </div>
        <div v-show="$form.svmType?.value !== 'Linear'">
          <Fieldset legend="Kernel">
            <RadioButtonGroup
              name="kernel"
              class="flex flex-wrap gap-4"
            >
              <div class="flex items-center gap-2">
                <RadioButton
                  input-id="linear"
                  value="linear"
                />
                <label for="linear">linear</label>
              </div>
              <div class="flex items-center gap-2">
                <RadioButton
                  input-id="poly"
                  value="poly"
                />
                <label for="poly">polynomial</label>
              </div>
              <div class="flex items-center gap-2">
                <RadioButton
                  input-id="rbf"
                  value="rbf"
                />
                <label for="rbf">rbf</label>
              </div>
              <div class="flex items-center gap-2">
                <RadioButton
                  input-id="sigmoid"
                  value="sigmoid"
                />
                <label for="sigmoid">sigmoid</label>
              </div>
              <div class="flex items-center gap-2">
                <RadioButton
                  input-id="precomputed"
                  value="precomputed"
                />
                <label for="precomputed">precomputed</label>
              </div>
            </RadioButtonGroup>
          </Fieldset>
          <div v-show="$form?.kernel?.value === 'poly'">
            <label
              for="degree"
              class="font-bold block mb-2"
            > Degree </label>
            <InputNumber
              name="degree"
              fluid
            />
          </div>
          <div class="flex flex-col items-center gap-2">
            <label
              for="gamma"
              class="font-bold block mb-2"
            > Gamma </label>
            <InputText
              name="gamma"
              fluid
            />
          </div>
          <div class="flex flex-col items-center gap-2">
            <label
              for="coef0"
              class="font-bold block mb-2"
            > coef0 </label>
            <InputNumber
              name="coef0"
              fluid
            />
          </div>
          <div class="flex flex-col items-center gap-2">
            <label
              for="shrinking"
              class="font-bold block mb-2"
            > Shirnking </label>
            <ToggleSwitch name="shrinking" />
          </div>
          <div class="flex flex-col items-center gap-2">
            <label
              for="cacheSize"
              class="font-bold block mb-2"
            > Cache Size </label>
            <InputNumber
              name="cacheSize"
              fluid
            />
          </div>
          <div class="flex flex-col items-center gap-2">
            <label
              for="maxIter"
              class="font-bold block mb-2"
            > Max Iteration </label>
            <InputText
              name="maxIter"
              fluid
            />
          </div>
          <div class="flex flex-col items-center gap-2">
            <label
              for="tol"
              class="font-bold block mb-2"
            > tol </label>
            <InputNumber
              name="tol"
              fluid
            />
          </div>
        </div>

        <div v-show="$form.svmType?.value === 'Linear'">
          <Fieldset legend="Loss">
            <RadioButtonGroup
              name="loss"
              class="flex flex-wrap gap-4"
            >
              <div class="flex items-center gap-2">
                <RadioButton
                  input-id="epsilonInsensitive"
                  value="epsilonInsensitive"
                />
                <label for="epsilonInsensitive">Epsilon Insensitive</label>
              </div>
              <div class="flex items-center gap-2">
                <RadioButton
                  input-id="squaredEpsilonInsensitive"
                  value="squaredEpsilonInsensitive"
                />
                <label for="squaredEpsilonInsensitive">Squared Epsilon Insensitive</label>
              </div>
            </RadioButtonGroup>
          </Fieldset>
          <Fieldset legend="Dual">
            <RadioButtonGroup
              name="dual"
              class="flex flex-wrap gap-4"
            >
              <div class="flex items-center gap-2">
                <RadioButton
                  input-id="auto"
                  value="auto"
                />
                <label for="auto">auto</label>
              </div>
              <div class="flex items-center gap-2">
                <RadioButton
                  input-id="true"
                  value="true"
                />
                <label for="true">true</label>
              </div>
              <div class="flex items-center gap-2">
                <RadioButton
                  input-id="false"
                  value="false"
                />
                <label for="false">false</label>
              </div>
            </RadioButtonGroup>
          </Fieldset>
          <div class="flex flex-col items-center gap-2">
            <label
              for="fitIntercept"
              class="font-bold block mb-2"
            > Fit Intercept </label>
            <ToggleSwitch name="fitIntercept" />
          </div>
          <div class="flex flex-col items-center gap-2">
            <label
              for="interceptScaling"
              class="font-bold block mb-2"
            > Intercept Scaling </label>
            <InputNumber
              name="interceptScaling"
              fluid
            />
          </div>
          <div class="flex flex-col items-center gap-2">
            <label
              for="tolLinear"
              class="font-bold block mb-2"
            > tol </label>
            <InputNumber
              name="tolLinear"
              fluid
            />
          </div>
          <div class="flex flex-col items-center gap-2">
            <label
              for="maxIterLinear"
              class="font-bold block mb-2"
            > Max Iteration </label>
            <InputText
              name="maxIterLinear"
              fluid
            />
          </div>
          <div class="flex flex-col items-center gap-2">
            <label
              for="randomState"
              class="font-bold block mb-2"
            > random state </label>
            <InputNumber
              name="randomState"
              fluid
            />
          </div>
        </div>
      </div>
      <Button
        type="submit"
        severity="secondary"
        label="Submit"
      />
    </Form>
  </div>
</template>

