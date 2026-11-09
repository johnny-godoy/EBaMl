// An agent that explains an EBM through an MCP server. `explain_with_mcp` starts the server (the `ebaml-mcp` command),
// offers its tools to the model, lets the model call them, and returns the final report with the list of calls.

// What the agent returns: the answer, and the tools that it called, in order. A caller checks `tool_calls` to see
// that the answer rests on tool results.
class McpAgentReport {
    report: string,
    tool_calls: string[],
    usage: TokenUsage,
}

function mcp_agent_system_msg() -> string {
    `You are a data scientist who explains a fitted Explainable Boosting Machine (EBM) to a colleague. An EBM is a sum of one graph per feature, so each graph shows exactly how one feature moves the prediction. You reach the model only through tools, and every tool result is exact. Quote only numbers, ranges and effects that a tool returned, because the colleague acts on your report and an invented number would mislead. Say when you did not check something.`
}

function mcp_agent_user_msg(question: string) -> string {
    `
        <instructions>
        1. Call the tool that summarizes the model, then the tool that lists the features.
        2. Read the graph of every feature that matters for the question, the most important first.
        3. If the question concerns one prediction, call the tool that explains a prediction.
        4. Write the report in plain prose paragraphs. Name the strongest effects with their ranges, and name any surprising or non-monotonic region, because the reader checks it against business knowledge.
        </instructions>

        <question>
        ${question}
        </question>
    `
}

// One message of a chat: who wrote it, and the text.
class ChatTurn {
    speaker: "user" | "assistant",
    text: string,
}

function mcp_chat_system_msg() -> string {
    `${mcp_agent_system_msg()} You chat with the colleague, so answer the latest message and use the earlier messages as context for follow-up questions.`
}

// The earlier messages as one block of text, so a follow-up such as "and what if the income were higher?" has its
// context. The tool results of earlier turns are not kept, so the model calls a tool again when it needs a number.
function conversation_text(history: ChatTurn[]) -> string {
    history
        .map((turn) -> {
            `<${turn.speaker}>\n${turn.text}\n</${turn.speaker}>`
        })
        .join("\n")
}

function mcp_chat_user_msg(question: string, history: ChatTurn[]) -> string {
    `
        <conversation>
        ${conversation_text(history)}
        </conversation>

        <instructions>
        1. Call only the tools that the latest message needs. Call a tool again if you need a number from an earlier turn.
        2. To explain one prediction, call the tool that explains a prediction with the number of a row of the data, or with the value of every feature. If the colleague gives neither, ask for one of them.
        3. Answer in plain prose, in a few sentences, because the colleague reads it in a terminal. Quote only numbers that a tool returned, and round them to three digits, because a longer number hides the point.
        </instructions>

        <message>
        ${question}
        </message>
    `
}

// The agent loop. The model sees the system message and the question, and `ToolRequested` events fill `tool_calls`.
function McpExplainer(question: string) -> string {
    client: Router
    prompt: `
        ${role("system")}
        ${mcp_agent_system_msg()}
        ${role("user")}
        ${mcp_agent_user_msg(question)}
    `
}

// One answer in a chat. `history` holds the messages before `question`.
function McpChat(question: string, history: ChatTurn[]) -> string {
    client: Router
    prompt: `
        ${role("system")}
        ${mcp_chat_system_msg()}
        ${role("user")}
        ${mcp_chat_user_msg(question, history)}
    `
}

// The tool-calling loop for any spec, on tools that the caller supplies. The server gives the tools in
// `explain_with_mcp` and `chat_with_mcp`, and the offline tests pass tools with fixed results.
function run_with_tools(
    spec: ai.FunctionSpec<string>,
    tools: ai.tools.Tool[],
    client: ai.Client? = null,
    model: ModelSpec? = null,
    max_steps: int = 12,
) -> McpAgentReport {
    let tool_calls: string[] = [];
    let samples: TokenSample[] = [];
    let agent = ai.Agent.new(max_steps = max_steps, client = select_client(client, model, short_defaults()), on_event = (
        event,
    ) -> {
        if let call: ai.events.ToolRequested = event {
            tool_calls.push(call.name);
        };
        if let model_call: ai.events.ModelCall = event {
            let sample = sample_of(model_call);
            if (sample != null) {
                samples.push(sample);
            };
        };
    });
    //# The spec carries the tools of the caller. The generated spec has none, because the tools are known at run time.
    let result = agent.run(ai.FunctionSpec { ...spec, toolbox: ai.tools.Toolbox.new(tools) });
    let usage = TokenUsage { input_tokens: 0, output_tokens: 0, samples: samples };
    for (let entry in result.usage) {
        usage
            = add_usage(
                usage,
                TokenUsage {
                    input_tokens: entry.usage.input_tokens ?? 0,
                    output_tokens: entry.usage.output_tokens ?? 0,
                    samples: [],
                },
            );
    }
    McpAgentReport { report: result.value, tool_calls: tool_calls, usage: usage }
}

function explain_with_tools(
    question: string,
    tools: ai.tools.Tool[],
    client: ai.Client? = null,
    model: ModelSpec? = null,
    max_steps: int = 12,
) -> McpAgentReport {
    run_with_tools(McpExplainer@spec(question), tools, client = client, model = model, max_steps = max_steps)
}

// One chat answer on tools that the caller supplies.
function chat_with_tools(
    question: string,
    history: ChatTurn[],
    tools: ai.tools.Tool[],
    client: ai.Client? = null,
    model: ModelSpec? = null,
    max_steps: int = 12,
) -> McpAgentReport {
    run_with_tools(McpChat@spec(question, history), tools, client = client, model = model, max_steps = max_steps)
}

// Entry point called from Python or the command line. `command` and `args` start the MCP server, for example
// `ebaml-mcp` with `["--model", "model.skops", "--data", "train.csv"]`. The connection closes on every path.
function explain_with_mcp(
    question: string,
    command: string,
    args: string[],
    client: ai.Client? = null,
    model: ModelSpec? = null,
    max_steps: int = 12,
) -> McpAgentReport {
    //# Start the server and read its tool catalog.
    let connection = ai.mcp.Connection.connect("ebaml", command, args);
    defer { connection.close() }
    //# Let the model call the tools, then return its report.
    explain_with_tools(question, connection.tools(), client = client, model = model, max_steps = max_steps)
}

// A chat in the terminal. The server starts once and stays up for the whole chat, so a later question is fast, and a
// tool that changes the model keeps its change. `baml.io.input` reads end-of-input as an empty line, so an empty line
// ends the chat. A failed turn (a rate limit, a refused key) prints its error and the chat goes on, because the
// colleague can ask again.
function chat_with_mcp(
    command: string,
    args: string[],
    client: ai.Client? = null,
    model: ModelSpec? = null,
    max_steps: int = 12,
) -> int {
    //# Start the server once and read its tool catalog.
    let connection = ai.mcp.Connection.connect("ebaml", command, args);
    defer { connection.close() }
    let tools = connection.tools();
    let history: ChatTurn[] = [];
    baml.io.println(
        "Ask about the model, or about one prediction. Press Enter on an empty line to leave.",
    );
    //# Read a message, answer it with the tools, and keep both in the history.
    while (true) {
        let message = baml.io.input("you> ").trim();
        if (message.length() == 0) {
            break;
        };
        let reply = chat_with_tools(message, history, tools, client = client, model = model, max_steps = max_steps) catch_all (error) {
            _ => {
                McpAgentReport {
                    report: `This turn failed: ${error}`,
                    tool_calls: [],
                    usage: TokenUsage { input_tokens: 0, output_tokens: 0, samples: [] },
                }
            }
        };
        baml.io.println(`\n${reply.report}\n`);
        if (reply.tool_calls.length() > 0) {
            baml.io.eprintln(`(tools: ${reply.tool_calls.join(", ")})`);
        };
        history.push(ChatTurn { speaker: "user", text: message });
        history.push(ChatTurn { speaker: "assistant", text: reply.report });
    }
    history.length() / 2
}
