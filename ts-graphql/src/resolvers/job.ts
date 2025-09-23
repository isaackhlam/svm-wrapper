import axios from 'axios';

const MODEL_TRAINER_ENDPOINT = "http://127.0.0.1:8008";

const pingDB = async (_p : any, _a: any, { dbClient }: any) => (
  dbClient.customCommand(["PING"])
)

const trainModel = async (_p: any, a: any, { dbClient }: any) => {
  const { modelType, taskType, id, hyperparameters } = a.input;
  const train_data_key = `/${id}/classification_train.csv`;
  const test_data_key = `/${id}/classification_test.csv`;
  console.log(a);
  console.log(typeof(taskType));
  let model = "";
  let task = "";
  if (modelType == "DNN") {
      model = "dnn";
  } else if (modelType == "SVM") {
    model = "svm";
  }
  if (taskType == "CLASSIFICATION") {
    task = "cls";
  } else if (taskType == "REGRESSION") {
    task = "reg";
  }

  const endpoint = `${MODEL_TRAINER_ENDPOINT}/${model}/${task}/run/v2`;

  const resp = await axios.post(endpoint, hyperparameters, {params: {
    train_data_key,
    test_data_key
  }});

  console.log(endpoint);
  return {id, status: "PENDING"};
}

export const Query = {
  dbconnection: pingDB,
}

export const Mutation = {
  trainModel: trainModel,
}
