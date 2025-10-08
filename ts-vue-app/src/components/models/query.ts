import { gql } from 'graphql-request';

export const submitJobMutation = gql`
  mutation TrainModel($input: TrainModelInput!) {
    trainModel(input: $input) {
      id
      status
    }
  }
`;
