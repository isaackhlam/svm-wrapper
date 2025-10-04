import axios from 'axios';

const MODEL_TRAINER_ENDPOINT = "http://trainer:8000";

const pingDB = async (_p : any, _a: any, { sql }: any) => {
  const res = await sql`SELECT NOW() as now`;
  return res[0].now;
}


const trainModel = async (_p: any, a: any, { sql }: any) => {
  const { modelType, taskType, id, hyperparameters } = a.input;
  // I don't think this is a good idea...
  const { explainModel } = hyperparameters;
  const [job] = await sql`
    INSERT INTO jobs (job_id, status)
    VALUES (${id}, 'PENDING')
    RETURNING job_id, status
  `;

  const train_data_key = `/${id}/train.csv`;
  const test_data_key = `/${id}/test.csv`;
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
    test_data_key,
    do_explain_model: explainModel,
  }});

  return {id: job.job_id, status: job.status};
}

const getJobStatus = async (_p: any, { input }: any, { sql }: any) => {
  const { id } = input;
  const result = await sql`
    SELECT status
    FROM jobs
    WHERE job_id = ${id}
  `;
  if (result.length > 0) {
    const status = result[0].status;
    return status;
  } else {
    throw Error("Job not Found");
  }
}

export const Query = {
  dbconnection: pingDB,
  getJobStatus,
}

export const Mutation = {
  trainModel: trainModel,
}
