// Port of TalkToEBM's t2ebm/prompts.py and t2ebm/llm.py: prompts, chat clients, CoT pipelines, token accounting.

// Some models behind `openrouter/free` reason mandatorily (reasoning cannot be disabled), and the hidden reasoning tokens
// count against `max_tokens`, so the caps are generous. 16000 stays below the output cap of 16384 tokens of common OpenAI
// models, because these limits are also the defaults of a ModelSpec for any provider.
// Transient failures (rate limits, network, 5xx) are retried; rejected requests are not.
client RouterBase = openai.OpenRouterClient.new(model = "openrouter/free", api_key = env.OPENROUTER_API_KEY, temperature = 0.7, max_tokens = 12000);

// First assistant turn of each conversation allows a longer answer.
client RouterLongBase = openai.OpenRouterClient.new(model = "openrouter/free", api_key = env.OPENROUTER_API_KEY, temperature = 0.7, max_tokens = 16000);

client Router = ai.clients.Retry.new(inner = RouterBase, max_attempts = 3);

// The benchmark is scored on one answer per question, so it samples greedily, as TalkToEBM's benchmark does.
client RouterBenchmarkBase = openai.OpenRouterClient.new(model = "openrouter/free", api_key = env.OPENROUTER_API_KEY, temperature = 0.0, max_tokens = 16000);

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

// How to read the series of the classes. Only the graph of a multiclass classifier has "Class: " lines, so the prompt
// of a binary classifier or a regressor gets no text about classes.
function multiclass_format(graph: string) -> string {
    if (graph.includes("Class: ")) {
        return "\nA graph of a multiclass classifier repeats items 3 to 5 once for each class, after a line \"Class: <name>\". In every bin, the scores have the same mean over all the classes of the model, including classes that the graph leaves out. Only the differences between the classes change the predicted probabilities, so compare the classes with each other within each bin.";
    };
    ""
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
        5. The upper bounds of the confidence interval (optional).${multiclass_format(graph)}
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

// The parts of the model that a description leaves out. The reader of a description cannot see a log warning, so the
// text itself must name them. An empty list gives no block, so a model without such parts gets no text about them.
function omitted_parts_msg(omitted_terms: string[], omitted_features: string[]) -> string {
    if (omitted_terms.length() + omitted_features.length() == 0) {
        return "";
    };
    let msg = "<omitted_parts>\nThe graph descriptions cover the main effect of one feature at a time.\n";
    if (omitted_terms.length() > 0) {
        msg
            = msg
                + `The model also has these interaction terms, and no graph description covers them: ${omitted_terms.join(", ")}.\n`;
    };
    if (omitted_features.length() > 0) {
        msg
            = msg
                + `These features have no main effect in the model, and no graph description covers them: ${omitted_features.join(", ")}.\n`;
    };
    msg
        + "Say in one sentence that your answer leaves these parts out, and name each of them, because the reader must know that the answer does not cover the whole model.\n</omitted_parts>\n\n"
}

function summarize_ebm_system_msg(expert_description: string) -> string {
    `You are ${expert_description}. You provide an overall summary of a Generalized Additive Model (GAM). The model consists of different graphs, and each graph shows the effect of one input feature. Answer in plain prose paragraphs, because a person reads your answer.`
}

function summarize_ebm_msg(
    feature_importances: string,
    graph_descriptions: string,
    dataset_description: string,
    omitted_terms: string[] = [],
    omitted_features: string[] = [],
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
        + omitted_parts_msg(omitted_terms, omitted_features)
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

function final_description_request(num_sentences: int, omitted_terms: string[] = []) -> string {
    omitted_parts_msg(omitted_terms, [])
        + `
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
    omitted_terms: string[] = [],
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
        ${final_description_request(num_sentences, omitted_terms = omitted_terms)}
    `
}

// summarize_ebm, turn 1.
function EbmSummaryDraft(
    feature_importances: string,
    graph_descriptions: string,
    opts: GraphPromptOptions,
    omitted_terms: string[] = [],
    omitted_features: string[] = [],
) -> string {
    client: RouterLong
    prompt: `
        ${role("system")}
        ${summarize_ebm_system_msg(opts.expert_description)}
        ${role("user")}
        ${summarize_ebm_msg(feature_importances, graph_descriptions, opts.dataset_description, omitted_terms = omitted_terms, omitted_features = omitted_features)}
    `
}

// summarize_ebm, turn 2: shorten the draft.
function EbmSummaryShorten(
    feature_importances: string,
    graph_descriptions: string,
    draft: string,
    num_sentences: int,
    opts: GraphPromptOptions,
    omitted_terms: string[] = [],
    omitted_features: string[] = [],
) -> string {
    client: Router
    prompt: `
        ${role("system")}
        ${summarize_ebm_system_msg(opts.expert_description)}
        ${role("user")}
        ${summarize_ebm_msg(feature_importances, graph_descriptions, opts.dataset_description, omitted_terms = omitted_terms, omitted_features = omitted_features)}
        ${role("assistant")}
        ${draft}
        ${role("user")}
        <instructions>
        Shorten your summary to at most ${num_sentences} sentences. Keep the most important information, because the reader wants the main effects and surprises. If the input has an <omitted_parts> block, keep the sentence that names the parts that the summary leaves out, because the reader must know that the summary does not cover the whole model.
        </instructions>
    `
}

// Token accounting.

// One request that the provider counted. Python reads the text of the messages in `body`, takes the prose away and
// fits the weights of its token estimate to the tokens of the graph that is left. The sample holds the request body
// only: the headers carry the API key, so they never leave the runtime.
class TokenSample {
    provider: string,
    model: string,
    body: string,
    input_tokens: int,
}

class TokenUsage {
    input_tokens: int,
    output_tokens: int,
    samples: TokenSample[],
}

// An LLM result plus the tokens the provider reported for producing it.
class Counted {
    text: string,
    usage: TokenUsage,
}

function add_usage(a: TokenUsage, b: TokenUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: a.input_tokens + b.input_tokens,
        output_tokens: a.output_tokens + b.output_tokens,
        samples: a.samples.concat(b.samples),
    }
}

// The input tokens of one call, with the cached and the cache-write tokens included, or null when the provider
// reported no count. A count that is missing is not zero: a zero would pull the weights of the estimate toward zero.
function input_total(usage: ai.events.Usage) -> int? {
    let uncached = usage.uncached_input_tokens;
    let cached = usage.cached_input_tokens;
    let written = usage.cache_write_input_tokens;
    if (uncached != null || cached != null || written != null) {
        return (uncached ?? 0) + (cached ?? 0) + (written ?? 0);
    };
    usage.input_tokens
}

// The sample of one model call. A call with no request, no usage, no model name or no input count gives no sample.
function sample_of(call: ai.events.ModelCall) -> TokenSample? {
    let usage = call.usage;
    let request = call.http_request;
    let model = call.resolved_model.model;
    if (usage == null || request == null || model == null) {
        return null;
    };
    let total = input_total(usage);
    if (total == null || total < 1) {
        return null;
    };
    TokenSample {
        provider: call.resolved_model.provider,
        model: model,
        body: request.body,
        input_tokens: total,
    }
}

// The samples of every model call in the events of a run, retries included.
function samples_of(events: ai.events.Event[]) -> TokenSample[] {
    let samples: TokenSample[] = [];
    for (let event in events) {
        if let call: ai.events.ModelCall = event {
            let sample = sample_of(call);
            if (sample != null) {
                samples.push(sample);
            };
        };
    }
    samples
}

// Runs one text-producing LLM function and keeps the provider's token usage. A count that the provider did not report
// is zero.
function run_counted(spec: ai.FunctionSpec<string>, client: ai.Client? = null) -> Counted {
    let result = ai.Agent.new(client = client).run(spec);
    let usage = TokenUsage { input_tokens: 0, output_tokens: 0, samples: samples_of(result.journal.entries()) };
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
    Counted { text: result.value, usage: usage }
}

// Entry points called from Python.

// Sends the first request of the graph description chain and returns the token samples of the requests. The sample
// holds the tokens of the input, so a short reply limit in `model` (`max_output_tokens`) keeps the cost low. A reply
// that the limit cuts off is not an error here, and the samples of the requests that were sent still come back. Any
// other error propagates. The scripts use the function to fit the weights of the token estimate of a model.
function sample_graph_tokens(
    graph: string,
    opts: GraphPromptOptions,
    client: ai.Client? = null,
    model: ModelSpec? = null,
) -> TokenSample[] {
    let samples: TokenSample[] = [];
    let agent = ai.Agent.new(client = select_client(client, model, long_defaults()), on_event = (event) -> {
        if let call: ai.events.ModelCall = event {
            let sample = sample_of(call);
            if (sample != null) {
                samples.push(sample);
            };
        };
    });
    //# A reply that hits the limit raises. The input counts of the request are what this function returns.
    //# Any other error (a rate limit, a refused key) is not a cut-off reply, so it propagates.
    let finished = {
        agent.run(GraphFirstPass@spec(graph, opts));
        true
    } catch (e) {
        ai.errors.FinishReasonError => false
    };
    samples
}

function describe_graph_cot(
    graph: string,
    num_sentences: int,
    opts: GraphPromptOptions,
    client: ai.Client? = null,
    model: ModelSpec? = null,
    omitted_terms: string[] = [],
) -> Counted {
    //# Three sequential assistant turns, each seeing the previous replies
    let long = select_client(client, model, long_defaults());
    let short = select_client(client, model, short_defaults());
    let first_pass = run_counted(GraphFirstPass@spec(graph, opts), client = long);
    let surprises = run_counted(GraphSurprises@spec(graph, first_pass.text, opts), client = short);
    let final = run_counted(
        GraphFinalDescription@spec(graph, first_pass.text, surprises.text, num_sentences, opts, omitted_terms = omitted_terms),
        client = short,
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
    model: ModelSpec? = null,
    omitted_terms: string[] = [],
    omitted_features: string[] = [],
) -> Counted {
    let draft = run_counted(
        EbmSummaryDraft@spec(
            feature_importances,
            graph_descriptions,
            opts,
            omitted_terms = omitted_terms,
            omitted_features = omitted_features,
        ),
        client = select_client(client, model, long_defaults()),
    );
    if (num_sentences == null) {
        return draft;
    };
    let shortened = run_counted(
        EbmSummaryShorten@spec(
            feature_importances,
            graph_descriptions,
            draft.text,
            num_sentences,
            opts,
            omitted_terms = omitted_terms,
            omitted_features = omitted_features,
        ),
        client = select_client(client, model, short_defaults()),
    );
    Counted { text: shortened.text, usage: add_usage(draft.usage, shortened.usage) }
}

// Full EBM description: one chain per feature graph (concurrently), then the summary.
// `graphs[i]` is the text of the graph of `feature_names[i]`. The summary names `omitted_terms` and `omitted_features`.
function describe_ebm(
    feature_names: string[],
    graphs: string[],
    feature_importances: string,
    num_sentences: int,
    graph_sentences: int,
    max_concurrency: int,
    opts: GraphPromptOptions,
    client: ai.Client? = null,
    model: ModelSpec? = null,
    omitted_terms: string[] = [],
    omitted_features: string[] = [],
) -> Counted {
    if (feature_names.length() != graphs.length()) {
        throw baml.errors.InvalidArgument { message: "feature_names and graphs must have the same length" };
    };
    //# One description chain per graph, at most `max_concurrency` at a time
    let limit = baml.spawn.Limit.new(max_concurrency);
    let futures = graphs.map((graph) -> {
        spawn with limit {
            describe_graph_cot(graph, graph_sentences, opts, client = client, model = model)
        }
    });
    let described: Counted[] = await baml.future.all(futures);

    let usage = TokenUsage { input_tokens: 0, output_tokens: 0, samples: [] };
    let lines: string[] = [];
    let i = 0;
    while (i < described.length()) {
        let item = described[i];
        usage = add_usage(usage, item.usage);
        lines.push(`${feature_names[i]}: ${item.text}`);
        i = i + 1;
    }
    //# Summarize all the per-graph descriptions
    let summary = summarize_ebm(
        feature_importances,
        lines.join("\n\n"),
        num_sentences,
        opts,
        client = client,
        model = model,
        omitted_terms = omitted_terms,
        omitted_features = omitted_features,
    );
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
    model: ModelSpec? = null,
) -> Counted {
    let selected = select_client(client, model, benchmark_defaults());
    let reasoning = run_counted(BenchmarkReasoning@spec(question), client = selected);
    let answer = run_counted(BenchmarkAnswer@spec(question, reasoning.text, answer_format), client = selected);
    Counted { text: answer.text, usage: add_usage(reasoning.usage, answer.usage) }
}
