import { createServer } from 'node:http'
import { createYoga, createSchema } from 'graphql-yoga'

import typeDefs from './schemas';
import resolvers from './resolvers';

const main = async () => {
  const yoga = createYoga({
    schema: createSchema({
      typeDefs: await typeDefs,
      resolvers,
    }),
  });
  const server = createServer(yoga);
  server.listen(4000, () => {
    console.info('Server is running on http://localhost:4000/graphql');
  });
};

main();
