FROM node:24-alpine
WORKDIR /app
COPY . .
ENV CI=true
RUN npm install -g pnpm@10.15.0 && pnpm install
CMD ["pnpm", "run", "dev"]
