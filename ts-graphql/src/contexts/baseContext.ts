import { GlideClient, GlideClusterClient, Logger } from "@valkey/valkey-glide";

const addresses = [
  {
    host: "0.0.0.0",
    port: 6379,
  },
];

const createDriver = async () => (
  GlideClient.createClient({
    addresses: addresses,
    requestTimeout: 500,
    clientName: "test_standalone_client",
  })
)

const baseContext = async () => {
  const dbClient = await createDriver();
  return { dbClient };
};

export default baseContext;
