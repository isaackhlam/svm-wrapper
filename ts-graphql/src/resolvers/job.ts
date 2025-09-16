const pingDB = async (_p : any, _a: any, { dbClient }: any) => (
  dbClient.customCommand(["PING"])
)

const trainModel = (_p: any, a: any, { dbClient }: any) => {
  const { modelType, taskType, id, hyperparameters } = a.input;
  const train_data_path = `${id}_train.csv`;
  const test_data_path = `${id}_test.csv`;
  console.log(a);
  console.log(typeof(taskType));
  return {id, status: "PENDING"};
}

export const Query = {
  dbconnection: pingDB,
}

export const Mutation = {
  trainModel: trainModel,
}
