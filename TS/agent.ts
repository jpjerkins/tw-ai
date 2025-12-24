// langchain-dadjoke-agent.ts
// Requires: LangChain v1 for TypeScript and an OPENAI_API_KEY env var.
// Run under Deno (or Node with an environment providing fetch and top-level await).

import { tool, createAgent } from "npm:langchain";
import * as z from "npm:zod";
// import "@std/dotenv/load";

// Helper: read OPENAI_API_KEY from environment (works in Deno and Node)
const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY");

if (!OPENAI_API_KEY) {
  console.warn(
    "Warning: OPENAI_API_KEY not found in environment. Make sure to set it before running."
  );
}

// Tool that calls the icanhazdadjoke API and returns the joke text
export const dadJokeTool = tool(
  async ({}) => {
    // The icanhazdadjoke API returns JSON when you set Accept: application/json
    const res = await fetch("https://icanhazdadjoke.com/", {
      headers: {
        Accept: "application/json",
        "User-Agent": "langchain-dad-joke-tool/1.0",
      },
    });

    if (!res.ok) {
      throw new Error(
        `DadJoke API returned status ${res.status} ${res.statusText}`
      );
    }

    const data = await res.json().catch(() => null);
    if (!data || typeof data.joke !== "string") {
      throw new Error("DadJoke API returned unexpected response.");
    }
    return data.joke as string;
}, {
  name: "DadJokeAPI",
  description: "Fetches a dad joke from icanhazdadjoke.com. Call with any input; returns a short dad joke string.",
});

export const systemPrompt = `You are a careful tool user.

Rules:
- Think step-by-step.`

const contextSchema = z.object({});

const agent = createAgent({
    model: "openai:gpt-5-mini",
    tools: [dadJokeTool],
    systemPrompt,
    contextSchema,
});

const stream = await agent.stream({
    messages: "Tell me a dad joke."
}, {
    streamMode: "values",
    context: {}
})

for await (const step of stream) {
    const lastMessage = step?.messages?.at(-1)
    console.log(lastMessage)
}
