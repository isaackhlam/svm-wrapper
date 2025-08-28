<script setup lang="ts">
import { reactive } from 'vue';
import { useToast } from 'primevue/usetoast';

const toast = useToast();

const initialValues = reactive({
    username: '',
    labelName: 'label',
    svmType: 'C',
    advanceOption: false,
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
                <div class="p-field-radiobutton">
                  <RadioButton inputId="nu" value="Nu" />
                  <label for="nu">Nu</label>
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
              <label for="nuValue" class="font-bold block mb-2"> BB </label>
              <InputNumber name="nuValue" fluid />
            </div>

            <Button type="submit" severity="secondary" label="Submit" />
        </Form>
    </div>
  </template>
  </Card>
</template>

