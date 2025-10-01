import postgres from 'postgres';

const sql = postgres({
  host: process.env.DB_HOST ?? "db",   // docker service name
  port: Number(process.env.DB_PORT ?? 5432),
  database: process.env.DB_NAME ?? "mydb",
  username: process.env.DB_USER ?? "user",
  password: process.env.DB_PASSWORD ?? "password",
});


const baseContext = async () => {
  return { sql };
};

export default baseContext;
