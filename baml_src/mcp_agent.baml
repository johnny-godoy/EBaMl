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

// The tool-calling loop on tools that the caller supplies. `explain_with_mcp` passes the tools of the server, and the
// offline tests pass tools with fixed results.
function explain_with_tools(
    question: string,
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
    let base = McpExplainer@spec(question);
    let spec = ai.FunctionSpec { ...base, toolbox: ai.tools.Toolbox.new(tools) };
    let result = agent.run(spec);
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
