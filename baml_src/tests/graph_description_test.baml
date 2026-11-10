// Offline tests of the graph description chain, the summary, describe_ebm and the benchmark answer in talk_to_ebm.baml, on scripted clients.

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

test "omitted_parts_msg names each omitted part and says why, and is empty without one" {
    assert.equal(omitted_parts_msg([], []), "");
    let both = omitted_parts_msg(["age & income", "age & city"], ["zip"]);
    assert.is_true(both.includes("<omitted_parts>") && both.includes("</omitted_parts>"));
    assert.is_true(
        both.includes(
            "interaction terms, and no graph description covers them: age & income, age & city.",
        ),
    );
    assert.is_true(
        both.includes("no main effect in the model, and no graph description covers them: zip."),
    );
    assert.is_true(both.includes("Say in one sentence that your answer leaves these parts out"));
    assert.is_true(both.includes("does not cover the whole model"));
    let terms = omitted_parts_msg(["age & income"], []);
    assert.is_true(terms.includes("age & income") && !terms.includes("no main effect"));
    let features = omitted_parts_msg([], ["zip"]);
    assert.is_true(features.includes("zip") && !features.includes("interaction terms"))
}

test "the summary prompts name the omitted parts, and a model without them gets no such text" {
    // Contract 4: a description that leaves out a model part says so in its text, not only in a log.
    let plain = summarize_ebm_msg("IMPORTANCES", "DESCRIPTIONS", "");
    assert.is_true(!plain.includes("omitted_parts"));
    let msg = summarize_ebm_msg(
        "IMPORTANCES",
        "DESCRIPTIONS",
        "",
        omitted_terms = ["age & income"],
        omitted_features = ["zip"],
    );
    assert.is_true(position_of(msg, "DESCRIPTIONS") < position_of(msg, "<omitted_parts>"));
    assert.is_true(position_of(msg, "</omitted_parts>") < position_of(msg, "<instructions>"));

    let result = summarize_ebm(
        "IMPORTANCES",
        "DESCRIPTIONS",
        3,
        default_graph_options(),
        client = EchoClient {},
        omitted_terms = ["age & income"],
        omitted_features = ["zip"],
    );
    // the echo of the shorten turn holds the draft turn too, so both turns got the block
    assert.is_true(result.text.includes("age & income") && result.text.includes("zip"));
    assert.is_true(
        result.text.includes("keep the sentence that names the parts that the summary leaves out"),
    );
    let draft = summarize_ebm(
        "IMPORTANCES",
        "DESCRIPTIONS",
        null,
        default_graph_options(),
        client = EchoClient {},
        omitted_terms = ["age & income"],
    );
    assert.is_true(draft.text.includes("age & income") && !draft.text.includes("no main effect"))
}

test "describe_graph_cot names the omitted interaction terms in its final turn only" {
    assert.is_true(!final_description_request(3).includes("omitted_parts"));
    assert.is_true(
        final_description_request(3, omitted_terms = ["age & income"]).includes("age & income"),
    );
    let opts = default_graph_options();
    let first = GraphFirstPass@spec("GRAPH", opts).prompt().text();
    let final = GraphFinalDescription@spec("GRAPH", "FIRST", "SURPRISES", 3, opts, omitted_terms = ["age & income"])
        .prompt()
        .text();
    assert.is_true(!first.includes("age & income"));
    assert.is_true(final.includes("age & income") && final.includes("at most 3 sentences"));
    let result = describe_graph_cot("GRAPH", 3, opts, client = EchoClient {}, omitted_terms = ["age & income"]);
    assert.is_true(result.text.includes("age & income"))
}

test "describe_ebm hands the omitted parts to the summary" {
    let result = describe_ebm(
        ["age"],
        ["GRAPH"],
        "IMPORTANCES",
        3,
        2,
        1,
        default_graph_options(),
        client = EchoClient {},
        omitted_terms = ["age & income"],
        omitted_features = ["zip"],
    );
    assert.is_true(result.text.includes("<omitted_parts>"));
    assert.is_true(result.text.includes("age & income") && result.text.includes("zip"))
}
