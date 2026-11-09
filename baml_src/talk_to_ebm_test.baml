// Offline tests for talk_to_ebm.baml: pure helpers and the pipelines on a scripted client.

// A model turn that holds `text` and reports `input_tokens` and `output_tokens`. The runner reads the usage of a turn
// from its calls.
function scripted_turn(text: string, input_tokens: int, output_tokens: int) -> ai.ModelTurn {
    ai.ModelTurn {
        parsed_output: null,
        content: [ai.content.Text { text: text }],
        stop_reason: ai.content.StopReason.Complete,
        resolved_model: null,
        requested_model: null,
        calls: [
            ai.events.ModelCall {
                resolved_model: ai.events.ModelKey { provider: "scripted", model: null },
                usage: ai.events.Usage {
                    input_tokens: input_tokens,
                    output_tokens: output_tokens,
                    cached_input_tokens: null,
                    reasoning_tokens: null,
                    cache_write_input_tokens: null,
                    uncached_input_tokens: null,
                },
                timing: ai.events.Timing { start_time: null, duration: null, ttft: null },
                http_request: null,
                http_response: null,
                sse_responses: null,
            },
        ],
        child_turns: [],
        error: null,
    }
}

// Answers every call with "R" and reports 10 input / 5 output tokens; counts the calls.
class ScriptedClient {
    calls: int,

    implements ai.Client {
        function provider(self) -> string {
            "scripted"
        }

        function requested_model(self) -> string {
            "scripted"
        }

        function render(self, input: ai.ModelTurnInput) -> baml.http.Request {
            baml.http.Request.from_url("http://scripted.invalid")
        }

        function invoke(self, input: ai.ModelTurnInput) -> ai.ModelTurn {
            self.calls += 1;
            scripted_turn("R", 10, 5)
        }
    }
}

test "add_usage sums both counters" {
    let total = add_usage(
        TokenUsage { input_tokens: 1, output_tokens: 2, samples: [] },
        TokenUsage { input_tokens: 10, output_tokens: 20, samples: [] },
    );
    assert.equal(total.input_tokens, 11);
    assert.equal(total.output_tokens, 22)
}

test "describe_graph_msg includes optional sections only when set" {
    let bare = describe_graph_msg("GRAPH", default_graph_options());
    assert.is_true(bare.includes("GRAPH"));
    assert.is_true(bare.includes("Describe the general pattern of the graph."));
    assert.is_true(!bare.includes("<dataset_description>"));

    let opts = GraphPromptOptions { ...default_graph_options(), dataset_description: "A pneumonia dataset" };
    assert.is_true(describe_graph_msg("GRAPH", opts).includes("A pneumonia dataset"))
}

test "describe_graph_msg says how to read the classes of a multiclass graph" {
    let msg = describe_graph_msg("Feature Name: a\nClass: high\nMeans: {}", default_graph_options());
    assert.is_true(
        msg.includes("(optional).\nA graph of a multiclass classifier repeats items 3 to 5"),
    );
    assert.is_true(msg.includes("including classes that the graph leaves out"));
    assert.is_true(msg.includes("compare the classes with each other within each bin."));
    assert.is_true(position_of(msg, "Class: <name>") < position_of(msg, "</graph_format>"))
}

test "describe_graph_msg of a graph with one output says nothing about classes" {
    let msg = describe_graph_msg("Feature Name: a\nMeans: {}", default_graph_options());
    assert.is_true(!msg.includes("multiclass"));
    assert.is_true(
        msg.includes("5. The upper bounds of the confidence interval (optional).\n</graph_format>"),
    )
}

test "describe_graph_cot makes three calls and sums usage" {
    let scripted = ScriptedClient { calls: 0 };
    let result = describe_graph_cot("GRAPH", 7, default_graph_options(), client = scripted);
    assert.equal(result.text, "R");
    assert.equal(scripted.calls, 3);
    assert.equal(result.usage.input_tokens, 30);
    assert.equal(result.usage.output_tokens, 15)
}

test "summarize_ebm shortens only when num_sentences is set" {
    let draft_only = ScriptedClient { calls: 0 };
    summarize_ebm("IMP", "DESC", null, default_graph_options(), client = draft_only);
    assert.equal(draft_only.calls, 1);

    let shortened = ScriptedClient { calls: 0 };
    let result = summarize_ebm("IMP", "DESC", 30, default_graph_options(), client = shortened);
    assert.equal(shortened.calls, 2);
    assert.equal(result.usage.input_tokens, 20)
}

test "describe_ebm: one chain per graph plus the summary" {
    let scripted = ScriptedClient { calls: 0 };
    let result = describe_ebm(["age", "bmi"], ["G1", "G2"], "IMP", 30, 7, 2, default_graph_options(), client = scripted);
    // 2 graphs x 3 turns + draft + shorten. The two chains run at the same time, so `self.calls += 1` of the
    // scripted client can lose an increment; each call reports 10 input and 5 output tokens, so the usage counts the
    // calls without a shared counter.
    assert.equal(result.usage.input_tokens, 8 * 10);
    assert.equal(result.usage.output_tokens, 8 * 5)
}

test "describe_ebm rejects mismatched names and graphs" {
    let scripted = ScriptedClient { calls: 0 };
    let outcome = describe_ebm(["age"], [], "IMP", 30, 7, 2, default_graph_options(), client = scripted) catch (e) {
        baml.errors.InvalidArgument => null,
    };
    assert.is_true(outcome == null);
    assert.equal(scripted.calls, 0)
}

test "answer_benchmark_question reasons, then extracts the answer" {
    let scripted = ScriptedClient { calls: 0 };
    let result = answer_benchmark_question("QUESTION", "'a' or 'b'", client = scripted);
    assert.equal(result.text, "R");
    assert.equal(scripted.calls, 2);
    assert.equal(result.usage.input_tokens, 20);
    assert.equal(result.usage.output_tokens, 10)
}

test "local_user_msg prepends the global explanation and puts in the guidance of the audience" {
    let plain = local_user_msg("GLOBAL", "PRED", "QUESTION", "FACTS", "GUIDANCE", "");
    assert.is_true(plain.includes("GLOBAL"));
    assert.is_true(plain.includes("FACTS"));
    assert.is_true(plain.includes("<audience>"));
    assert.is_true(plain.includes("GUIDANCE"));
    assert.is_true(!plain.includes("<dataset_description>"));

    let described = local_user_msg("GLOBAL", "PRED", "QUESTION", "FACTS", "GUIDANCE", "A bank dataset");
    assert.is_true(described.includes("A bank dataset"))
}

test "local_user_msg tells the model to write in the language" {
    assert.is_true(
        local_user_msg("GLOBAL", "PRED", "QUESTION", "FACTS", "GUIDANCE", "").includes(
            "Write the whole answer in English",
        ),
    );
    assert.is_true(
        local_user_msg("GLOBAL", "PRED", "QUESTION", "FACTS", "GUIDANCE", "", language = "German")
            .includes("Write the whole answer in German"),
    )
}

test "explain_local makes three calls and sums usage" {
    let scripted = ScriptedClient { calls: 0 };
    let facts = LocalFactTexts {
        prediction: "P",
        counterfactual: "C",
        sufficient_reason: "S",
        completions: "K",
    };
    let result = explain_local("GLOBAL", facts, "GUIDANCE", 5, default_graph_options(), client = scripted);
    assert.equal(scripted.calls, 3);
    assert.equal(result.counterfactual, "R");
    assert.equal(result.sufficient_reason, "R");
    assert.equal(result.completions, "R");
    assert.equal(result.usage.input_tokens, 30);
    assert.equal(result.usage.output_tokens, 15)
}

// Answers every call with its own rendered prompt, so a test can see what each call was given.
class EchoClient {
    implements ai.Client {
        function provider(self) -> string {
            "echo"
        }

        function requested_model(self) -> string {
            "echo"
        }

        function render(self, input: ai.ModelTurnInput) -> baml.http.Request {
            baml.http.Request.from_url("http://echo.invalid")
        }

        function invoke(self, input: ai.ModelTurnInput) -> ai.ModelTurn {
            let prompt = input.prompt(ai.internal.build_output_format(input.output_type));
            scripted_turn(prompt.text(), 1, 1)
        }
    }
}

// Index of `needle` in `text`, or -1 when absent, so positions can be ordered.
function position_of(text: string, needle: string) -> int {
    let found = text.index_of(needle);
    if (found == null) {
        return -1;
    };
    found
}

test "local_system_msg names the expert and forbids invented facts" {
    let msg = local_system_msg("a credit risk analyst");
    assert.is_true(msg.includes("You are a credit risk analyst."));
    assert.is_true(msg.includes("invented number"));
    assert.is_true(msg.includes("a question about the predicted value"));
    assert.is_true(msg.includes("another class becomes the predicted class"))
}

test "local_user_msg orders the global explanation, dataset, prediction, facts, guidance and question" {
    let msg = local_user_msg("GLOBAL", "PRED", "QUESTION", "FACTS", "GUIDANCE", "DATASET");
    let positions = [
        position_of(msg, "GLOBAL"),
        position_of(msg, "DATASET"),
        position_of(msg, "PRED"),
        position_of(msg, "FACTS"),
        position_of(msg, "GUIDANCE"),
        position_of(msg, "QUESTION"),
    ];
    assert.is_true(positions[0] >= 0);
    assert.is_true(positions[0] < positions[1]);
    assert.is_true(positions[1] < positions[2]);
    assert.is_true(positions[2] < positions[3]);
    assert.is_true(positions[3] < positions[4]);
    assert.is_true(positions[4] < positions[5])
}

test "question functions state the sentence limit and their own query" {
    assert.is_true(counterfactual_question(3).includes("at most 3 sentences"));
    assert.is_true(counterfactual_question(3).includes("minimum change required"));
    assert.is_true(counterfactual_question(3).includes("is_smallest_set"));
    assert.is_true(counterfactual_question(3).includes("keep its size out of the answer"));
    assert.is_true(sufficient_reason_question(4).includes("at most 4 sentences"));
    assert.is_true(sufficient_reason_question(4).includes("sufficient reason"));
    assert.is_true(sufficient_reason_question(4).includes("is_smallest_set"));
    assert.is_true(sufficient_reason_question(4).includes("keep its size out of the answer"));
    assert.is_true(completions_question(6).includes("at most 6 sentences"));
    assert.is_true(completions_question(6).includes("count completions"))
}

test "the questions explain the lower bound and why a set is not proved smallest" {
    assert.is_true(counterfactual_question(3).includes("smallest_possible_number_of_changes"));
    assert.is_true(
        counterfactual_question(3).includes("ended at its time limit or was a fast search"),
    );
    assert.is_true(counterfactual_question(3).includes("you may state the gap"));
    assert.is_true(sufficient_reason_question(3).includes("smallest_possible_number_of_features"));
    assert.is_true(sufficient_reason_question(3).includes("did not prove the size"));
    assert.is_true(sufficient_reason_question(3).includes("you may state the gap"))
}

test "the questions make no claim that the facts do not support" {
    // A failed search is not a proof: the counterfactual search covers realistic values only.
    assert.is_true(counterfactual_question(3).includes("within the realistic values"));
    assert.is_true(!counterfactual_question(3).includes("cannot be changed by altering"));
    assert.is_true(sufficient_reason_question(3).includes("found no set of inputs"));
    assert.is_true(!sufficient_reason_question(3).includes("every input matters"));
    // Completion shares are estimates and can fall as inputs become known.
    assert.is_true(completions_question(3).includes("when it falls"));
    assert.is_true(!completions_question(3).includes("how that share grows"))
}

test "each local function gets its own facts, question and system role" {
    let facts = LocalFactTexts {
        prediction: "PRED-FACT",
        counterfactual: "CF-FACT",
        sufficient_reason: "SR-FACT",
        completions: "CO-FACT",
    };
    let opts = GraphPromptOptions { ...default_graph_options(), expert_description: "an EXPERT" };

    let cf = LocalCounterfactual@spec("GLOBAL", facts, "GUIDANCE", 3, opts).prompt().text();
    let sr = LocalSufficientReason@spec("GLOBAL", facts, "GUIDANCE", 4, opts).prompt().text();
    let co = LocalCompletions@spec("GLOBAL", facts, "GUIDANCE", 6, opts).prompt().text();

    for (let prompt in [cf, sr, co]) {
        assert.is_true(prompt.includes("You are an EXPERT."));
        assert.is_true(prompt.includes("GLOBAL"));
        assert.is_true(prompt.includes("PRED-FACT"))
    }
    ;

    assert.is_true(
        cf.includes("CF-FACT")
            && cf.includes("at most 3 sentences")
            && cf.includes("minimum change required"),
    );
    assert.is_true(!cf.includes("SR-FACT") && !cf.includes("CO-FACT"));
    assert.is_true(
        sr.includes("SR-FACT")
            && sr.includes("at most 4 sentences")
            && sr.includes("sufficient reason"),
    );
    assert.is_true(!sr.includes("CF-FACT") && !sr.includes("CO-FACT"));
    assert.is_true(
        co.includes("CO-FACT")
            && co.includes("at most 6 sentences")
            && co.includes("count completions"),
    );
    assert.is_true(!co.includes("CF-FACT") && !co.includes("SR-FACT"))
}

test "explain_local routes each answer to its own field" {
    let facts = LocalFactTexts {
        prediction: "PRED-FACT",
        counterfactual: "CF-FACT",
        sufficient_reason: "SR-FACT",
        completions: "CO-FACT",
    };
    let result = explain_local("GLOBAL", facts, "GUIDANCE", 5, default_graph_options(), client = EchoClient {});
    assert.is_true(result.counterfactual.includes("CF-FACT"));
    assert.is_true(
        !result.counterfactual.includes("SR-FACT") && !result.counterfactual.includes("CO-FACT"),
    );
    assert.is_true(result.sufficient_reason.includes("SR-FACT"));
    assert.is_true(
        !result.sufficient_reason.includes("CF-FACT")
            && !result.sufficient_reason.includes("CO-FACT"),
    );
    assert.is_true(result.completions.includes("CO-FACT"));
    assert.is_true(
        !result.completions.includes("CF-FACT") && !result.completions.includes("SR-FACT"),
    );
    assert.is_true(result.completions.includes("data scientist"));
    assert.equal(result.usage.input_tokens, 3);
    assert.equal(result.usage.output_tokens, 3)
}

// Answers call n with "REPLY-n" and counts the calls, so a test can see which reply reached which later prompt.
// Use it only for sequential pipelines: concurrent calls would race on the counter.
class SequenceClient {
    calls: int,

    implements ai.Client {
        function provider(self) -> string {
            "sequence"
        }

        function requested_model(self) -> string {
            "sequence"
        }

        function render(self, input: ai.ModelTurnInput) -> baml.http.Request {
            baml.http.Request.from_url("http://sequence.invalid")
        }

        function invoke(self, input: ai.ModelTurnInput) -> ai.ModelTurn {
            self.calls += 1;
            scripted_turn(`REPLY-${self.calls}`, 1, 1)
        }
    }
}

// The roles of a rendered prompt, joined, e.g. "system,user,assistant,user".
function roles_of(prompt: ai.Prompt) -> string {
    prompt.messages()
        .map((m) -> {
            m.role
        })
        .join(",")
}

// The content of message `index` of a rendered prompt.
function message_at(prompt: ai.Prompt, index: int) -> string {
    prompt.messages()[index].content
}

test "default_graph_options is a plain statistician with no extra context" {
    let opts = default_graph_options();
    assert.equal(opts.expert_description, "an expert statistician and data scientist");
    assert.equal(opts.graph_description, "");
    assert.equal(opts.dataset_description, "");
    assert.equal(opts.task_description, "Describe the general pattern of the graph.")
}

test "add_usage keeps the operands independent" {
    let zero = TokenUsage { input_tokens: 0, output_tokens: 0, samples: [] };
    let usage = TokenUsage { input_tokens: 3, output_tokens: 4, samples: [] };
    assert.equal(add_usage(zero, usage).input_tokens, 3);
    assert.equal(add_usage(zero, usage).output_tokens, 4);
    assert.equal(add_usage(usage, usage).input_tokens, 6);
    assert.equal(add_usage(usage, usage).output_tokens, 8)
}

test "graph_system_msg names the expert and the GAM" {
    let msg = graph_system_msg("a medical statistician");
    assert.is_true(msg.includes("You are a medical statistician."));
    assert.is_true(msg.includes("Generalized Additive Model"))
}

test "describe_graph_msg orders graph, graph description, dataset and task" {
    let opts = GraphPromptOptions {
        expert_description: "E",
        graph_description: "GRAPHDESC",
        dataset_description: "DATASET",
        task_description: "TASK",
    };
    let msg = describe_graph_msg("GRAPH", opts);
    assert.is_true(position_of(msg, "GRAPH") >= 0);
    assert.is_true(position_of(msg, "GRAPH") < position_of(msg, "GRAPHDESC"));
    assert.is_true(position_of(msg, "GRAPHDESC") < position_of(msg, "DATASET"));
    assert.is_true(position_of(msg, "DATASET") < position_of(msg, "TASK"));
    assert.is_true(msg.includes("<dataset_description>"))
}

test "describe_graph_msg leaves out each optional section on its own" {
    let only_graph = GraphPromptOptions { ...default_graph_options(), graph_description: "GRAPHDESC" };
    let msg = describe_graph_msg("GRAPH", only_graph);
    assert.is_true(msg.includes("GRAPHDESC"));
    assert.is_true(!msg.includes("<dataset_description>"));

    let only_dataset = GraphPromptOptions { ...default_graph_options(), dataset_description: "DATASET" };
    let other = describe_graph_msg("GRAPH", only_dataset);
    assert.is_true(other.includes("DATASET"));
    assert.is_true(!other.includes("GRAPHDESC"));

    let custom = GraphPromptOptions { ...default_graph_options(), task_description: "MY TASK" };
    let task = describe_graph_msg("GRAPH", custom);
    assert.is_true(task.includes("MY TASK"));
    assert.is_true(!task.includes("Describe the general pattern"))
}

test "summarize_ebm_system_msg names the expert" {
    let msg = summarize_ebm_system_msg("a risk analyst");
    assert.is_true(msg.includes("You are a risk analyst."));
    assert.is_true(msg.includes("overall summary"))
}

test "summarize_ebm_msg orders importances, descriptions and dataset, and the dataset is optional" {
    let msg = summarize_ebm_msg("IMPORTANCES", "DESCRIPTIONS", "DATASET");
    assert.is_true(position_of(msg, "IMPORTANCES") >= 0);
    assert.is_true(position_of(msg, "IMPORTANCES") < position_of(msg, "DESCRIPTIONS"));
    assert.is_true(position_of(msg, "DESCRIPTIONS") < position_of(msg, "DATASET"));
    assert.is_true(position_of(msg, "DATASET") < position_of(msg, "<instructions>"));

    let bare = summarize_ebm_msg("IMPORTANCES", "DESCRIPTIONS", "");
    assert.is_true(!bare.includes("<dataset_description>"));
    assert.is_true(bare.includes("<instructions>"))
}

test "describe_graph_cot prompts: roles, content and sentence limit per turn" {
    let opts = GraphPromptOptions { ...default_graph_options(), expert_description: "an EXPERT" };

    let first = GraphFirstPass@spec("GRAPH", opts).prompt();
    assert.equal(roles_of(first), "system,user");
    assert.is_true(message_at(first, 0).includes("You are an EXPERT."));
    assert.is_true(message_at(first, 1).includes("GRAPH"));

    let second = GraphSurprises@spec("GRAPH", "FIRST", opts).prompt();
    assert.equal(roles_of(second), "system,user,assistant,user");
    assert.is_true(message_at(second, 1).includes("GRAPH"));
    assert.equal(message_at(second, 2).trim(), "FIRST");
    assert.is_true(message_at(second, 3).includes("surprising or counterintuitive"));

    let third = GraphFinalDescription@spec("GRAPH", "FIRST", "SURPRISES", 7, opts).prompt();
    assert.equal(roles_of(third), "system,user,assistant,user,assistant,user");
    assert.equal(message_at(third, 2).trim(), "FIRST");
    assert.equal(message_at(third, 4).trim(), "SURPRISES");
    assert.is_true(message_at(third, 5).includes("at most 7 sentence"))
}

test "describe_graph_cot feeds each reply to the next turn in order" {
    let sequence = SequenceClient { calls: 0 };
    let result = describe_graph_cot("GRAPH", 7, default_graph_options(), client = sequence);
    assert.equal(sequence.calls, 3);
    assert.equal(result.text, "REPLY-3");
    assert.equal(result.usage.input_tokens, 3);
    assert.equal(result.usage.output_tokens, 3);

    // The echo client returns the last prompt, which must hold both earlier replies in order.
    let echo = describe_graph_cot("GRAPH", 7, default_graph_options(), client = EchoClient {});
    assert.is_true(echo.text.includes("GRAPH"));
    assert.is_true(echo.text.includes("at most 7 sentence"));
    assert.is_true(position_of(echo.text, "Highlight any region") >= 0);
    assert.is_true(
        position_of(echo.text, "Describe the graph in at most")
            > position_of(echo.text, "Highlight any region"),
    )
}

test "describe_graph_cot passes the second and third turns the replies of the earlier turns" {
    let sequence = SequenceClient { calls: 0 };
    let first = GraphFirstPass@spec("GRAPH", default_graph_options());
    let reply_one = run_counted(first, client = sequence);
    assert.equal(reply_one.text, "REPLY-1");
    let reply_two = run_counted(GraphSurprises@spec("GRAPH", reply_one.text, default_graph_options()), client = sequence);
    assert.equal(reply_two.text, "REPLY-2");
    let third = GraphFinalDescription@spec("GRAPH", reply_one.text, reply_two.text, 5, default_graph_options())
        .prompt();
    assert.equal(message_at(third, 2).trim(), "REPLY-1");
    assert.equal(message_at(third, 4).trim(), "REPLY-2")
}

test "summarize_ebm prompts: roles, content and sentence limit" {
    let opts = GraphPromptOptions {
        ...default_graph_options(),
        expert_description: "an EXPERT",
        dataset_description: "DATASET",
    };
    let draft = EbmSummaryDraft@spec("IMPORTANCES", "DESCRIPTIONS", opts).prompt();
    assert.equal(roles_of(draft), "system,user");
    assert.is_true(message_at(draft, 0).includes("You are an EXPERT."));
    assert.is_true(message_at(draft, 1).includes("IMPORTANCES"));
    assert.is_true(message_at(draft, 1).includes("DESCRIPTIONS"));
    assert.is_true(message_at(draft, 1).includes("DATASET"));

    let shorten = EbmSummaryShorten@spec("IMPORTANCES", "DESCRIPTIONS", "DRAFT", 12, opts).prompt();
    assert.equal(roles_of(shorten), "system,user,assistant,user");
    assert.is_true(message_at(shorten, 1).includes("IMPORTANCES"));
    assert.equal(message_at(shorten, 2).trim(), "DRAFT");
    assert.is_true(message_at(shorten, 3).includes("at most 12 sentences"))
}

test "summarize_ebm returns the draft when num_sentences is null, else the shortened reply" {
    let draft_only = SequenceClient { calls: 0 };
    let kept = summarize_ebm("IMP", "DESC", null, default_graph_options(), client = draft_only);
    assert.equal(kept.text, "REPLY-1");
    assert.equal(kept.usage.input_tokens, 1);
    assert.equal(kept.usage.output_tokens, 1);

    let shortened = SequenceClient { calls: 0 };
    let result = summarize_ebm("IMP", "DESC", 30, default_graph_options(), client = shortened);
    assert.equal(result.text, "REPLY-2");
    assert.equal(result.usage.input_tokens, 2);
    assert.equal(result.usage.output_tokens, 2);

    // The shorten prompt holds the draft and the limit.
    let echo = summarize_ebm("IMP", "DESC", 30, default_graph_options(), client = EchoClient {});
    assert.is_true(echo.text.includes("at most 30 sentences"));
    assert.is_true(echo.text.includes("IMP"));
    assert.is_true(echo.text.includes("DESC"))
}

test "describe_ebm pairs each feature name with its own description, in order" {
    let result = describe_ebm(
        ["age", "bmi"],
        ["G-AGE", "G-BMI"],
        "IMP",
        30,
        7,
        2,
        default_graph_options(),
        client = EchoClient {},
    );
    // The summary draft echoes the "name: description" lines; each description echoes its own graph.
    assert.is_true(position_of(result.text, "age: ") >= 0);
    assert.is_true(position_of(result.text, "age: ") < position_of(result.text, "bmi: "));
    assert.is_true(position_of(result.text, "G-AGE") < position_of(result.text, "bmi: "));
    assert.is_true(position_of(result.text, "bmi: ") < position_of(result.text, "G-BMI"));
    assert.is_true(result.text.includes("at most 30 sentences"));
    assert.is_true(result.text.includes("at most 7 sentence"));
    assert.is_true(result.text.includes("IMP"))
}

test "describe_ebm gives the same usage whatever the concurrency" {
    let one = describe_ebm(
        ["a", "b", "c"],
        ["G1", "G2", "G3"],
        "IMP",
        30,
        7,
        1,
        default_graph_options(),
        client = EchoClient {},
    );
    let many = describe_ebm(
        ["a", "b", "c"],
        ["G1", "G2", "G3"],
        "IMP",
        30,
        7,
        8,
        default_graph_options(),
        client = EchoClient {},
    );
    // 3 graphs x 3 turns + draft + shorten = 11 calls at 1 input and 1 output token each.
    assert.equal(one.usage.input_tokens, 11);
    assert.equal(one.usage.output_tokens, 11);
    assert.equal(many.usage.input_tokens, 11);
    assert.equal(many.usage.output_tokens, 11)
}

test "describe_ebm with no graphs only summarizes" {
    let scripted = ScriptedClient { calls: 0 };
    let result = describe_ebm([], [], "IMP", 30, 7, 2, default_graph_options(), client = scripted);
    assert.equal(scripted.calls, 2);
    assert.equal(result.usage.input_tokens, 20)
}

test "describe_ebm rejects more names than graphs and more graphs than names" {
    let scripted = ScriptedClient { calls: 0 };
    let more_graphs = describe_ebm([], ["G"], "IMP", 30, 7, 2, default_graph_options(), client = scripted) catch (e) {
        baml.errors.InvalidArgument => null,
    };
    assert.is_true(more_graphs == null);
    assert.equal(scripted.calls, 0)
}

test "benchmark prompts: roles, question, reasoning and answer format" {
    let reasoning = BenchmarkReasoning@spec("QUESTION").prompt();
    assert.equal(roles_of(reasoning), "system,user");
    assert.is_true(
        message_at(reasoning, 0).includes("You are an expert statistician and data scientist."),
    );
    assert.is_true(message_at(reasoning, 1).includes("QUESTION"));

    let answer = BenchmarkAnswer@spec("QUESTION", "REASONING", "'a' or 'b'").prompt();
    assert.equal(roles_of(answer), "system,user,assistant,user");
    assert.is_true(message_at(answer, 1).includes("QUESTION"));
    assert.equal(message_at(answer, 2).trim(), "REASONING");
    assert.is_true(message_at(answer, 3).includes("'a' or 'b'"));
    assert.is_true(message_at(answer, 3).includes("Reply with the answer only"))
}

test "answer_benchmark_question hands the reasoning to the answer turn" {
    let sequence = SequenceClient { calls: 0 };
    let result = answer_benchmark_question("QUESTION", "'a' or 'b'", client = sequence);
    assert.equal(sequence.calls, 2);
    assert.equal(result.text, "REPLY-2");
    assert.equal(result.usage.input_tokens, 2);
    assert.equal(result.usage.output_tokens, 2);

    let echo = answer_benchmark_question("QUESTION", "'a' or 'b'", client = EchoClient {});
    assert.is_true(echo.text.includes("'a' or 'b'"));
    assert.is_true(echo.text.includes("QUESTION"));
    assert.is_true(echo.text.includes("Summarize your response"))
}

// A model turn that holds `text` and reports 1 input and 1 output token.
function text_turn(text: string) -> ai.ModelTurn {
    scripted_turn(text, 1, 1)
}

// Answers call n with "Rn<<" + its rendered prompt + ">>". A later prompt then shows which earlier reply sits where.
// Use it only for sequential pipelines: concurrent calls would race on the counter.
class TaggedEchoClient {
    calls: int,

    implements ai.Client {
        function provider(self) -> string {
            "tagged-echo"
        }

        function requested_model(self) -> string {
            "tagged-echo"
        }

        function render(self, input: ai.ModelTurnInput) -> baml.http.Request {
            baml.http.Request.from_url("http://tagged-echo.invalid")
        }

        function invoke(self, input: ai.ModelTurnInput) -> ai.ModelTurn {
            self.calls += 1;
            let prompt = input.prompt(ai.internal.build_output_format(input.output_type));
            text_turn(`R${self.calls}<<${prompt.text()}>>`)
        }
    }
}

// Answers "R" to graph prompts and echoes the summary prompts, so the graph descriptions stay predictable.
class SummaryEchoClient {
    implements ai.Client {
        function provider(self) -> string {
            "summary-echo"
        }

        function requested_model(self) -> string {
            "summary-echo"
        }

        function render(self, input: ai.ModelTurnInput) -> baml.http.Request {
            baml.http.Request.from_url("http://summary-echo.invalid")
        }

        function invoke(self, input: ai.ModelTurnInput) -> ai.ModelTurn {
            let text = input.prompt(ai.internal.build_output_format(input.output_type)).text();
            if (text.includes("overall summary of a Generalized Additive Model")) {
                return text_turn(text);
            };
            text_turn("R")
        }
    }
}

test "describe_graph_cot puts the first reply in the second and third turns and the second reply in the third" {
    let tagged = TaggedEchoClient { calls: 0 };
    let result = describe_graph_cot("GRAPH", 7, default_graph_options(), client = tagged);
    assert.is_true(result.text.starts_with("R3<<"));
    // The third prompt holds R1 (with the first prompt inside it) before R2, and R2 holds R1 too.
    let r1 = position_of(result.text, "R1<<");
    let r2 = position_of(result.text, "R2<<");
    assert.is_true(r1 > 0);
    assert.is_true(r1 < r2);
    assert.is_true(position_of(result.text.slice(r2, result.text.length()), "R1<<") > 0)
}

test "describe_ebm joins the per-feature descriptions with a blank line" {
    let result = describe_ebm(
        ["age", "bmi"],
        ["G1", "G2"],
        "IMP",
        30,
        7,
        2,
        default_graph_options(),
        client = SummaryEchoClient {},
    );
    assert.is_true(result.text.includes("age: R\n\nbmi: R"))
}

test "answer_benchmark_question puts the reasoning reply in the answer prompt" {
    let tagged = TaggedEchoClient { calls: 0 };
    let result = answer_benchmark_question("QUESTION", "'a' or 'b'", client = tagged);
    assert.is_true(result.text.starts_with("R2<<"));
    assert.is_true(result.text.includes("R1<<"));
    assert.is_true(
        position_of(result.text, "R1<<") < position_of(result.text, "Summarize your response"),
    )
}

test "summarize_ebm puts the draft reply in the shorten prompt" {
    let tagged = TaggedEchoClient { calls: 0 };
    let result = summarize_ebm("IMP", "DESC", 30, default_graph_options(), client = tagged);
    assert.is_true(result.text.starts_with("R2<<"));
    assert.is_true(position_of(result.text, "R1<<") > 0);
    assert.is_true(
        position_of(result.text, "R1<<") < position_of(result.text, "at most 30 sentences"),
    )
}

// A usage record with the four input counts of a provider. A null count is one that the provider did not report.
function usage_with(
    input_tokens: int?,
    uncached: int?,
    cached: int?,
    written: int?,
) -> ai.events.Usage {
    ai.events.Usage {
        input_tokens: input_tokens,
        output_tokens: 5,
        cached_input_tokens: cached,
        reasoning_tokens: null,
        cache_write_input_tokens: written,
        uncached_input_tokens: uncached,
    }
}

// A model call to `model` with a request that holds `body` and a header with a secret, as a real call holds the key.
function call_with(model: string?, body: string?, usage: ai.events.Usage?) -> ai.events.ModelCall {
    let request: baml.http.Request? = null;
    if (body != null) {
        let built = baml.http.Request.from_url("http://sample.invalid");
        built.body = body;
        built.headers = { "Authorization": "Bearer SECRET-MARKER" };
        request = built;
    };
    ai.events.ModelCall {
        resolved_model: ai.events.ModelKey { provider: "scripted", model: model },
        usage: usage,
        timing: ai.events.Timing { start_time: null, duration: null, ttft: null },
        http_request: request,
        http_response: null,
        sse_responses: null,
    }
}

test "input_total adds the cache counts to the uncached count" {
    assert.equal(input_total(usage_with(900, 100, 700, 50)), 850);
    assert.equal(input_total(usage_with(null, 100, null, null)), 100);
    assert.equal(input_total(usage_with(null, null, 7, null)), 7);
    assert.equal(input_total(usage_with(null, null, null, 3)), 3)
}

test "input_total uses the input count when the provider reports no cache counts" {
    assert.equal(input_total(usage_with(233, null, null, null)), 233)
}

test "input_total is null, not zero, when the provider reports no count" {
    assert.equal(input_total(usage_with(null, null, null, null)), null)
}

test "sample_of keeps the provider, the model, the body and the total" {
    let sample = sample_of(call_with("some/model", "BODY", usage_with(null, 10, 20, 5)));
    assert.equal(
        sample,
        TokenSample { provider: "scripted", model: "some/model", body: "BODY", input_tokens: 35 },
    )
}

test "sample_of gives no sample for a call with null counts" {
    assert.equal(
        sample_of(call_with("some/model", "BODY", usage_with(null, null, null, null))),
        null,
    )
}

test "sample_of gives no sample for a zero count, a missing usage, a missing request or a missing model" {
    assert.equal(sample_of(call_with("some/model", "BODY", usage_with(0, null, null, null))), null);
    assert.equal(sample_of(call_with("some/model", "BODY", null)), null);
    assert.equal(sample_of(call_with("some/model", null, usage_with(5, null, null, null))), null);
    assert.equal(sample_of(call_with(null, "BODY", usage_with(5, null, null, null))), null)
}

test "a sample never holds a header of the request" {
    let sample = sample_of(call_with("some/model", "BODY", usage_with(5, null, null, null)));
    assert.not_null(sample);
    assert.is_true(!baml.json.stringify(sample).includes("SECRET-MARKER"));
    assert.is_true(!baml.json.stringify(sample).includes("Authorization"))
}

test "samples_of keeps one sample for each counted call, in order" {
    let events: ai.events.Event[] = [
        call_with("a", "ONE", usage_with(1, null, null, null)),
        call_with("b", "TWO", usage_with(null, null, null, null)),
        call_with("c", "THREE", usage_with(3, null, null, null)),
    ];
    let samples = samples_of(events);
    assert.equal(samples.length(), 2);
    assert.equal(samples[0].body, "ONE");
    assert.equal(samples[1].body, "THREE")
}

test "add_usage joins the samples of both operands in order" {
    let one = TokenSample { provider: "p", model: "m", body: "1", input_tokens: 1 };
    let two = TokenSample { provider: "p", model: "m", body: "2", input_tokens: 2 };
    let total = add_usage(
        TokenUsage { input_tokens: 1, output_tokens: 1, samples: [one] },
        TokenUsage { input_tokens: 1, output_tokens: 1, samples: [two] },
    );
    assert.equal(total.samples, [one, two])
}

// Answers with a model call that holds a request body, a model name and counts, as a real provider call does.
class SampledClient {
    implements ai.Client {
        function provider(self) -> string {
            "sampled"
        }

        function requested_model(self) -> string {
            "sampled"
        }

        function render(self, input: ai.ModelTurnInput) -> baml.http.Request {
            baml.http.Request.from_url("http://sampled.invalid")
        }

        function invoke(self, input: ai.ModelTurnInput) -> ai.ModelTurn {
            let turn = scripted_turn("ANSWER", 40, 6);
            turn.calls
                = [call_with("sampled/model", "REQUEST-BODY", usage_with(40, null, null, null))];
            turn
        }
    }
}

test "run_counted returns the sample of each counted call of the run" {
    let result = run_counted(GraphFirstPass@spec("GRAPH", default_graph_options()), client = SampledClient {});
    assert.equal(result.text, "ANSWER");
    assert.equal(
        result.usage.samples,
        [
            TokenSample {
                provider: "scripted",
                model: "sampled/model",
                body: "REQUEST-BODY",
                input_tokens: 40,
            },
        ],
    );
    assert.is_true(!baml.json.stringify(result).includes("SECRET-MARKER"))
}

test "describe_graph_cot returns one sample for each of its three turns" {
    let result = describe_graph_cot("GRAPH", 7, default_graph_options(), client = SampledClient {});
    assert.equal(result.usage.samples.length(), 3)
}

test "a run with no request in its calls returns no sample" {
    let result = describe_graph_cot("GRAPH", 7, default_graph_options(), client = SequenceClient { calls: 0 });
    assert.equal(result.usage.samples.length(), 0)
}

test "sample_graph_tokens returns the sample of the request" {
    let samples = sample_graph_tokens("GRAPH", default_graph_options(), client = SampledClient {});
    assert.equal(
        samples,
        [
            TokenSample {
                provider: "scripted",
                model: "sampled/model",
                body: "REQUEST-BODY",
                input_tokens: 40,
            },
        ],
    )
}

test "sample_graph_tokens returns no sample when the request has no counts" {
    let samples = sample_graph_tokens("GRAPH", default_graph_options(), client = SequenceClient { calls: 0 });
    assert.equal(samples.length(), 0)
}

// Fails the request as a provider does: with a reply that the output limit cut off, or with another error.
class FailingClient {
    cut_off: bool,

    implements ai.Client {
        function provider(self) -> string {
            "failing"
        }

        function requested_model(self) -> string {
            "failing"
        }

        function render(self, input: ai.ModelTurnInput) -> baml.http.Request {
            baml.http.Request.from_url("http://failing.invalid")
        }

        function invoke(self, input: ai.ModelTurnInput) -> ai.ModelTurn {
            if (self.cut_off) {
                throw ai.errors.FinishReasonError {
                    provider: "failing",
                    finish_reason: ai.content.StopReason.MaxTokens,
                    raw_output: "",
                };
            };
            throw ai.errors.RateLimited { provider: "failing", retry_after_ms: null, status_code: 429, raw_body: null };
        }
    }
}

test "sample_graph_tokens treats a reply that the limit cut off as no error" {
    let samples = sample_graph_tokens("GRAPH", default_graph_options(), client = FailingClient { cut_off: true });
    assert.equal(samples.length(), 0)
}

test "sample_graph_tokens lets another error propagate" {
    let marker = TokenSample { provider: "caught", model: "caught", body: "", input_tokens: 1 };
    let outcome = sample_graph_tokens("GRAPH", default_graph_options(), client = FailingClient { cut_off: false }) catch (e) {
        ai.errors.RateLimited => [marker]
    };
    assert.equal(outcome, [marker])
}
