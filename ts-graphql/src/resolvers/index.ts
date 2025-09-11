import merge from 'lodash/merge';

const resolvers = merge({}, ...[require('./healthcheck')]);

export default resolvers;
