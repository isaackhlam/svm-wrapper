FROM node:24-alpine
WORKDIR /app
COPY . .
ENV CI=true
RUN npm install -g pnpm
RUN pnpm install
CMD ["pnpm", "run", "dev"]
