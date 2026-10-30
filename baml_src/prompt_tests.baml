// Live prompt tests: real requests to the default Router clients, so they need a real OPENROUTER_API_KEY.
// They check properties that a good answer must have (no model numbers for a layperson, a sentence limit, the
// requested answer format), not exact wording, because the free router is not deterministic. A failure on one run can
// be noise: rerun it before you change a prompt. Offline runs exclude them: `baml test -x root::Live`.

// Number of sentences in `text`: a run of closing punctuation followed by a space or the end of the text ends a sentence.
// The period of a decimal (0.62) and the abbreviations "e.g." and "i.e." do not end a sentence.
function count_sentences(text: string) -> int {
    let ends = baml.regex.new("[.!?]+[\"')\\]]*(\\s|$)").match_all(text.trim()).length();
    let abbreviations = baml.regex.new("(?i)\\b(e\\.g|i\\.e)\\.\\s").match_all(text).length();
    ends - abbreviations
}

testset "PromptChecks" {
    test "count_sentences: the period of a decimal does not end a sentence" {
        assert.equal(count_sentences("The debt ratio is 0.62. Income is low."), 2)
    }

    test "count_sentences: an abbreviation does not end a sentence" {
        assert.equal(
            count_sentences("Young people, e.g. those under 30, score higher. Older people score lower!"),
            2,
        )
    }

    test "count_sentences: a closing quote or bracket after the end counts once" {
        assert.equal(count_sentences("He said \"no.\" Then (later.) she left..."), 3)
    }

    test "count_sentences: text with no closing punctuation has no sentence end" {
        assert.equal(count_sentences("Income matters most"), 0);
        assert.equal(count_sentences(""), 0)
    }
}

function live_local_facts() -> LocalFactTexts {
    LocalFactTexts {
        prediction: "The model predicts 'loan denied' with probability 0.83 (log-odds +1.37).",
        counterfactual: "Smallest change that flips the prediction: income from 21000 to 48000 (score change -2.91 log-odds). The flip holds at the unfavourable edge of the confidence interval.",
        sufficient_reason: "Sufficient reason: income = 21000 and debt_ratio = 0.62 fix the prediction. Age (score +0.04) cannot change it.",
        completions: "Share of people with the same prediction: knowing only income 21000: 71%; income 21000 and debt_ratio 0.62: 100%.",
    }
}

testset "Live" {
    test "layperson counterfactual: no model numbers, within the sentence limit, names the input" {
        let text = LocalCounterfactual(
            "The model denies loans mainly because of low income and a high debt ratio.",
            live_local_facts(),
            "layperson",
            3,
            default_graph_options(),
        );
        assert.is_true(text.includes("income"));
        assert.is_true(!text.includes("0.83") && !text.includes("1.37") && !text.includes("2.91"));
        assert.is_true(!text.includes("%"));
        assert.is_true(count_sentences(text) <= 3)
    }

    test "layperson sufficient reason: says the other inputs cannot change the outcome" {
        let text = LocalSufficientReason(
            "The model denies loans mainly because of low income and a high debt ratio.",
            live_local_facts(),
            "layperson",
            3,
            default_graph_options(),
        );
        assert.is_true(text.to_lower_case().includes("income"));
        assert.is_true(text.to_lower_case().includes("age"));
        assert.is_true(!text.includes("0.04"));
        assert.is_true(count_sentences(text) <= 3)
    }

    test "layperson completions: approximate wording and no percentages" {
        let text = LocalCompletions(
            "The model denies loans mainly because of low income and a high debt ratio.",
            live_local_facts(),
            "layperson",
            3,
            default_graph_options(),
        );
        assert.is_true(!text.includes("71") && !text.includes("%"));
        assert.is_true(count_sentences(text) <= 3)
    }

    test "data scientist counterfactual: keeps the figures from the facts" {
        let text = LocalCounterfactual(
            "The model denies loans mainly because of low income and a high debt ratio.",
            live_local_facts(),
            "data_scientist",
            3,
            default_graph_options(),
        );
        assert.is_true(text.includes("48000") || text.includes("2.91"))
    }

    test "explain_local: three explanations, each non-empty, usage reported" {
        let result = explain_local(
            "The model denies loans mainly because of low income and a high debt ratio.",
            live_local_facts(),
            "layperson",
            3,
            default_graph_options(),
        );
        assert.is_true(result.counterfactual.length() > 0);
        assert.is_true(result.sufficient_reason.length() > 0);
        assert.is_true(result.completions.length() > 0);
        assert.is_true(result.usage.input_tokens > 0);
        assert.is_true(result.usage.output_tokens > 0)
    }

    test "describe_graph_cot: names the feature and respects the sentence limit" {
        let graph = "{\"feature_name\": \"age\", \"type\": \"continuous\", \"mean\": {\"18-30\": 0.8, \"30-50\": 0.0, \"50-90\": -0.6}}";
        let result = describe_graph_cot(graph, 3, default_graph_options());
        assert.is_true(result.text.to_lower_case().includes("age"));
        assert.is_true(count_sentences(result.text) <= 4);
        assert.is_true(result.usage.input_tokens > 0)
    }

    test "describe_graph_cot: a multiclass graph is read class against class" {
        // "young" leads at low ages and "old" at high ages; a good answer names both classes and the change of lead
        let graph = "Feature Name: age\nFeature Type: continuous\nClass: young\nMeans: {\"(18, 30)\": 1.5, \"(30, 50)\": 0.0, \"(50, 90)\": -1.5}\nClass: middle\nMeans: {\"(18, 30)\": 0.0, \"(30, 50)\": 0.5, \"(50, 90)\": 0.0}\nClass: old\nMeans: {\"(18, 30)\": -1.5, \"(30, 50)\": -0.5, \"(50, 90)\": 1.5}";
        let result = describe_graph_cot(graph, 4, default_graph_options());
        let text = result.text.to_lower_case();
        assert.is_true(text.includes("young") && text.includes("old"));
        assert.is_true(count_sentences(result.text) <= 5)
    }

    test "describe_graph_cot: a ModelSpec sends the requests to the chosen model" {
        let graph = "{\"feature_name\": \"age\", \"type\": \"continuous\", \"mean\": {\"18-30\": 0.8, \"30-50\": 0.0, \"50-90\": -0.6}}";
        let model = ModelSpec {
            provider: "openrouter",
            model: "openrouter/free",
            api_key_env: "OPENROUTER_API_KEY",
            base_url: null,
            temperature: null,
            max_output_tokens: null,
            headers: null,
        };
        let result = describe_graph_cot(graph, 3, default_graph_options(), model = model);
        assert.is_true(result.text.to_lower_case().includes("age"));
        assert.is_true(result.usage.output_tokens > 0)
    }

    test "answer_benchmark_question: the final turn follows the requested format" {
        let question = "A model's graph for feature x has mean value 2.0 for x < 5 and -1.0 for x >= 5. Is the function decreasing in x? Think step by step.";
        let result = answer_benchmark_question(question, "'yes' or 'no'");
        let answer = result.text.trim().to_lower_case();
        assert.is_true(answer.includes("yes"));
        assert.is_true(answer.length() < 20)
    }
}
