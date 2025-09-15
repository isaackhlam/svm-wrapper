const pingDB = async (_p : any, _a: any, { dbClient }: any) => (
  dbClient.customCommand(["PING"])
)

export const Query = {
  dbconnection: pingDB,
}
