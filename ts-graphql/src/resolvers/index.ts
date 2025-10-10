import merge from 'lodash/merge';

/* eslint-disable-next-line @typescript-eslint/no-require-imports */
const resolvers = merge({}, ...[require('./healthcheck'), require('./job')]);

export default resolvers;
