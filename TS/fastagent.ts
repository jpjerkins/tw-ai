import * as setup from "./setup.ts";
import { SqlDatabase } from "@langchain/classic/sql_db";
import { DataSource } from "npm:typeorm";
import * as z from "npm:zod";
import { createAgent } from "npm:langchain";
import { tool } from "npm:langchain";


const datasource = new DataSource({
    type: "sqlite",
    database: "./Chinook.db", // Replace with the link to your database
});
const db = await SqlDatabase.fromDataSourceParams({
    appDataSource: datasource,
});

export const executeSQL = tool(({ query }, runtime) => {
    return runtime.context.db.run(query)
}, {
    name: "execute_sql",
    description: "Execute a SQLite command and return results.",
    schema: z.object({ query: z.string() })
})

export const systemPrompt = `You are a careful SQLite analyst.

Rules:
- Think step-by-step.
- When you need data, call the tool \`execute_sql\` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows of output unless the user explicitly asks otherwise.
- If the tool returns 'Error:', revise the SQL and try again.
- Prefer explicit column lists; avoid SELECT *.`

const contextSchema = z.object({
    db: z.instanceof(SqlDatabase),
});

const agent = createAgent({
    model: "anthropic:claude-sonnet-4-5-20250929",
    tools: [executeSQL],
    systemPrompt,
    contextSchema,
});

const stream = await agent.stream({
    messages: "Which table has the largest number of entries?"
}, {
    streamMode: "values",
    context: {
        db,
    }
})

for await (const step of stream) {
    const lastMessage = step?.messages?.at(-1)
    console.log(lastMessage)
}

const question = "Which genre on average has the longest tracks?"
const stream2 = await agent.stream({
    messages: question
}, {
    streamMode: "values",
    context: {
        db,
    }
})

for await (const step of stream2) {
    const lastMessage = step?.messages?.at(-1)
    console.log(lastMessage)
}