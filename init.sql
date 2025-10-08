CREATE TYPE job_status AS ENUM ('PENDING', 'PROCESSING', 'FINISHED');

CREATE TABLE jobs (
  job_id UUID PRIMARY KEY,
  status job_status NOT NULL
);
