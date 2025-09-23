import merge from 'lodash/merge';

const resolvers = merge({}, ...[require('./healthcheck'), require('./job')]);

export default resolvers;
