// Port of TalkToEBM's t2ebm/prompts.py and t2ebm/llm.py: prompts, chat clients, CoT pipelines, token accounting.

// Some models behind `openrouter/free` reason mandatorily (reasoning cannot be disabled), and the hidden reasoning tokens
// count against `max_tokens`, so the caps are generous.
// Transient failures (rate limits, network, 5xx) are retried; rejected requests are not.
client RouterBase = openai.OpenRouterClient.new(model = "openrouter/free", api_key = env.OPENROUTER_API_KEY, temperature = 0.7, max_tokens = 6000);

// First assistant turn of each conversation allows a longer answer.
client RouterLongBase = openai.OpenRouterClient.new(model = "openrouter/free", api_key = env.OPENROUTER_API_KEY, temperature = 0.7, max_tokens = 8000);

client Router = ai.clients.Retry.new(inner = RouterBase, max_attempts = 3);

// The benchmark is scored on one answer per question, so it samples greedily, as TalkToEBM's benchmark does.
client RouterBenchmarkBase = openai.OpenRouterClient.new(model = "openrouter/free", api_key = env.OPENROUTER_API_KEY, temperature = 0.0, max_tokens = 8000);

client RouterBenchmark = ai.clients.Retry.new(inner = RouterBenchmarkBase, max_attempts = 3);

client RouterLong = ai.clients.Retry.new(inner = RouterLongBase, max_attempts = 3);

class GraphPromptOptions {
    expert_description: string,
    graph_description: string,
    dataset_description: string,
    task_description: string,
}

function default_graph_options() -> GraphPromptOptions {
    GraphPromptOptions {
        expert_description: "an expert statistician and data scientist",
        graph_description: "",
        dataset_description: "",
        task_description: "Describe the general pattern of the graph.",
    }
}

// Prompt text helpers. Pure functions, so the Jinja `{% if %}` blocks become plain BAML.

function graph_system_msg(expert_description: string) -> string {
    `You are ${expert_description}. You interpret global explanations produced by a Generalized Additive Model (GAM). Base every answer on the graph from the user, on any other information you are given, and on your knowledge of the real world. Answer in plain prose paragraphs, because a person reads your answer.`
}

function describe_graph_msg(graph: string, opts: GraphPromptOptions) -> string {
    let msg = `
        <graph_format>
        The graph is a JSON object. Its keys are the x-axis and its values are the y-axis. For a continuous feature, each key is an interval where the function predicts the same value. For a categorical feature, each key is a possible value of the feature.
        The graph contains these items:
        1. The name of the feature.
        2. The type of the feature (continuous, categorical, or boolean).
        3. The mean values.
        4. The lower bounds of the confidence interval (optional).
        5. The upper bounds of the confidence interval (optional).
        </graph_format>

        <graph>
        ${graph}
        </graph>
    `
        + "\n\n";
    if (opts.graph_description.length() > 0) {
        msg = msg + `<graph_description>\n${opts.graph_description}\n</graph_description>\n\n`;
    };
    if (opts.dataset_description.length() > 0) {
        msg = msg + `<dataset_description>\n${opts.dataset_description}\n</dataset_description>\n\n`;
    };
    msg + `<task>\n${opts.task_description}\n</task>`
}

function summarize_ebm_system_msg(expert_description: string) -> string {
    `You are ${expert_description}. You provide an overall summary of a Generalized Additive Model (GAM). The model consists of different graphs, and each graph shows the effect of one input feature. Answer in plain prose paragraphs, because a person reads your answer.`
}

function summarize_ebm_msg(
    feature_importances: string,
    graph_descriptions: string,
    dataset_description: string,
) -> string {
    let msg = `
        <feature_importances>
        ${feature_importances}
        </feature_importances>

        <graph_descriptions>
        ${graph_descriptions}
        </graph_descriptions>
    `
        + "\n\n";
    if (dataset_description.length() > 0) {
        msg = msg + `<dataset_description>\n${dataset_description}\n</dataset_description>\n\n`;
    };
    msg
        + `
            <instructions>
            Summarize the model. The feature importances are global, and there is exactly one graph description for each feature.
            Cover the most important features and their effect on the outcome. Leave out unimportant features and effects, so the summary stays short.
            Include any surprising pattern, because the reader wants to know where the model differs from expectation.
            </instructions>
        `
}

// Request text shared by the second turn and the replay of that turn in the third.
function surprises_request() -> string {
    `
        <instructions>
                Study the graph carefully. Highlight any region that you find surprising or counterintuitive, and name its range or category so the reader can find it. Suggest why the behavior is surprising and what may have caused it.
                </instructions>
    `
}

function final_description_request(num_sentences: int) -> string {
    `
        <instructions>
                Describe the graph in at most ${num_sentences} sentences.
                Include the important surprising patterns, because the reader uses them to decide where to look closer.
                The reader knows that the graph is from a Generalized Additive Model (GAM), so leave that background out.
                </instructions>
    `
}

// describe_graph_cot, turn 1: system + describe_graph prompt.
function GraphFirstPass(graph: string, opts: GraphPromptOptions) -> string {
    client: RouterLong
    prompt: `
        ${role("system")}
        ${graph_system_msg(opts.expert_description)}
        ${role("user")}
        ${describe_graph_msg(graph, opts)}
    `
}

// Turn 2: surprising regions. `first_pass` is the assistant reply from turn 1.
function GraphSurprises(graph: string, first_pass: string, opts: GraphPromptOptions) -> string {
    client: Router
    prompt: `
        ${role("system")}
        ${graph_system_msg(opts.expert_description)}
        ${role("user")}
        ${describe_graph_msg(graph, opts)}
        ${role("assistant")}
        ${first_pass}
        ${role("user")}
        ${surprises_request()}
    `
}

// Turn 3: brief final description.
function GraphFinalDescription(
    graph: string,
    first_pass: string,
    surprises: string,
    num_sentences: int,
    opts: GraphPromptOptions,
) -> string {
    client: Router
    prompt: `
        ${role("system")}
        ${graph_system_msg(opts.expert_description)}
        ${role("user")}
        ${describe_graph_msg(graph, opts)}
        ${role("assistant")}
        ${first_pass}
        ${role("user")}
        ${surprises_request()}
        ${role("assistant")}
        ${surprises}
        ${role("user")}
        ${final_description_request(num_sentences)}
    `
}

// summarize_ebm, turn 1.
function EbmSummaryDraft(
    feature_importances: string,
    graph_descriptions: string,
    opts: GraphPromptOptions,
) -> string {
    client: RouterLong
    prompt: `
        ${role("system")}
        ${summarize_ebm_system_msg(opts.expert_description)}
        ${role("user")}
        ${summarize_ebm_msg(feature_importances, graph_descriptions, opts.dataset_description)}
    `
}

// summarize_ebm, turn 2: shorten the draft.
function EbmSummaryShorten(
    feature_importances: string,
    graph_descriptions: string,
    draft: string,
    num_sentences: int,
    opts: GraphPromptOptions,
) -> string {
    client: Router
    prompt: `
        ${role("system")}
        ${summarize_ebm_system_msg(opts.expert_description)}
        ${role("user")}
        ${summarize_ebm_msg(feature_importances, graph_descriptions, opts.dataset_description)}
        ${role("assistant")}
        ${draft}
        ${role("user")}
        <instructions>
        Shorten your summary to at most ${num_sentences} sentences. Keep the most important information, because the reader wants the main effects and surprises.
        </instructions>
    `
}

// Token accounting.

class TokenUsage {
    input_tokens: int,
    output_tokens: int,
}

// An LLM result plus the tokens the provider reported for producing it.
class Counted {
    text: string,
    usage: TokenUsage,
}

class TokenBudget {
    estimated_tokens: int,
    status: "ok" | "warn" | "over",
}

function add_usage(a: TokenUsage, b: TokenUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: a.input_tokens + b.input_tokens,
        output_tokens: a.output_tokens + b.output_tokens,
    }
}

// Rough estimate (~3 chars/token): graph text is mostly digits and punctuation, which tokenize worse than prose.
// No tokenizer is available before a call, so this is a guard rail; the provider's own error is the real limit.
function estimate_tokens(text: string) -> int {
    (text.length() + 2) / 3
}

// "ok" within `max_tokens`; "warn" up to `max_tokens` plus `overbudget_percent`; "over" beyond that.
function check_token_budget(text: string, max_tokens: int, overbudget_percent: int) -> TokenBudget {
    let estimated = estimate_tokens(text);
    let hard_limit = max_tokens * (100 + overbudget_percent) / 100;
    if (estimated > hard_limit) {
        return TokenBudget { estimated_tokens: estimated, status: "over" };
    };
    if (estimated > max_tokens) {
        return TokenBudget { estimated_tokens: estimated, status: "warn" };
    };
    TokenBudget { estimated_tokens: estimated, status: "ok" }
}

// Runs one text-producing LLM function and keeps the provider's token usage.
function run_counted(spec: ai.FunctionSpec<string>, client: ai.Client? = null) -> Counted {
    let result = ai.Agent.new(client = client).run(spec);
    Counted {
        text: result.value,
        usage: TokenUsage {
            input_tokens: result.usage.input_tokens,
            output_tokens: result.usage.output_tokens,
        },
    }
}

// Entry points called from Python.

function describe_graph_cot(
    graph: string,
    num_sentences: int,
    opts: GraphPromptOptions,
    client: ai.Client? = null,
) -> Counted {
    //# Three sequential assistant turns, each seeing the previous replies
    let first_pass = run_counted(GraphFirstPass@spec(graph, opts), client = client);
    let surprises = run_counted(GraphSurprises@spec(graph, first_pass.text, opts), client = client);
    let final = run_counted(
        GraphFinalDescription@spec(graph, first_pass.text, surprises.text, num_sentences, opts),
        client = client,
    );
    Counted {
        text: final.text,
        usage: add_usage(add_usage(first_pass.usage, surprises.usage), final.usage),
    }
}

// `num_sentences` null keeps the draft summary as is.
function summarize_ebm(
    feature_importances: string,
    graph_descriptions: string,
    num_sentences: int?,
    opts: GraphPromptOptions,
    client: ai.Client? = null,
) -> Counted {
    let draft = run_counted(EbmSummaryDraft@spec(feature_importances, graph_descriptions, opts), client = client);
    if (num_sentences == null) {
        return draft;
    };
    let shortened = run_counted(
        EbmSummaryShorten@spec(feature_importances, graph_descriptions, draft.text, num_sentences, opts),
        client = client,
    );
    Counted { text: shortened.text, usage: add_usage(draft.usage, shortened.usage) }
}

// Full EBM description: one chain per feature graph (concurrently), then the summary.
// `graphs[i]` is the text of the graph of `feature_names[i]`.
function describe_ebm(
    feature_names: string[],
    graphs: string[],
    feature_importances: string,
    num_sentences: int,
    graph_sentences: int,
    max_concurrency: int,
    opts: GraphPromptOptions,
    client: ai.Client? = null,
) -> Counted {
    if (feature_names.length() != graphs.length()) {
        throw baml.errors.InvalidArgument { message: "feature_names and graphs must have the same length" };
    };
    //# One description chain per graph, at most `max_concurrency` at a time
    let limit = baml.spawn.Limit.new(max_concurrency);
    let futures = graphs.map((graph) -> {
        spawn with limit { describe_graph_cot(graph, graph_sentences, opts, client = client) }
    });
    let described = await baml.future.all(futures);

    let usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
    let lines: string[] = [];
    let i = 0;
    while (i < described.length()) {
        let item = described[i];
        usage = add_usage(usage, item.usage);
        lines.push(`${feature_names[i]}: ${item.text}`);
        i = i + 1;
    }
    //# Summarize all the per-graph descriptions
    let summary = summarize_ebm(feature_importances, lines.join("\n\n"), num_sentences, opts, client = client);
    Counted { text: summary.text, usage: add_usage(usage, summary.usage) }
}

// Benchmark (benchmark/): the questions end with "Think step by step.", so turn 1 reasons freely and turn 2
// extracts the final answer in the format the scorer parses.

function BenchmarkReasoning(question: string) -> string {
    client: RouterBenchmark
    prompt: `
        ${role("system")}
        ${graph_system_msg("an expert statistician and data scientist")}
        ${role("user")}
        ${question}
    `
}

function BenchmarkAnswer(question: string, reasoning: string, answer_format: string) -> string {
    client: RouterBenchmark
    prompt: `
        ${role("system")}
        ${graph_system_msg("an expert statistician and data scientist")}
        ${role("user")}
        ${question}
        ${role("assistant")}
        ${reasoning}
        ${role("user")}
        <instructions>
        Summarize your response by answering with ${answer_format}. Reply with the answer only, because a program parses your reply.
        </instructions>
    `
}

function answer_benchmark_question(
    question: string,
    answer_format: string,
    client: ai.Client? = null,
) -> Counted {
    let reasoning = run_counted(BenchmarkReasoning@spec(question), client = client);
    let answer = run_counted(BenchmarkAnswer@spec(question, reasoning.text, answer_format), client = client);
    Counted { text: answer.text, usage: add_usage(reasoning.usage, answer.usage) }
}
