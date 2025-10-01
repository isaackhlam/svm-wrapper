import { createServer } from 'node:http'
import { createYoga, createSchema } from 'graphql-yoga'

import typeDefs from './schemas';
import resolvers from './resolvers';
import baseContext from './contexts/baseContext';

const main = async () => {
  const yoga = createYoga({
    schema: createSchema({
      typeDefs: await typeDefs,
      resolvers,
    }),
    context: baseContext,
    cors: {
      origin: '*',
    }
  });
  const server = createServer(yoga);
  server.listen(4000, '0.0.0.0', () => {
    console.info('Server is running on http://localhost:4000/graphql');
  });
};

main();
