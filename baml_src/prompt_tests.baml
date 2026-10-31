// Live prompt tests: real requests to the default Router clients, so they need a real OPENROUTER_API_KEY.
// They check properties that a good answer must have (no model numbers for a layperson, a sentence limit, the
// requested answer format), not exact wording, because the free router is not deterministic. Each test logs the reply
// of the model. If a test fails, read the reply and rerun the test before you change a prompt, because one failure can
// be noise. Offline runs exclude them: `baml test -x root::Live`.

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

// The facts of one loan prediction, as ebaml.local.fact_text.facts_to_texts renders them for each audience. A layperson
// gets words in place of the model numbers, so the layperson prompts see no score, probability or share.
function live_layperson_facts() -> LocalFactTexts {
    LocalFactTexts {
        prediction: `{"predicted_class": "loan denied", "contributions_largest_first": [{"feature": "income", "value": "[1.8e+04, 2.4e+04)", "effect": "strongly towards this prediction"}, {"feature": "debt_ratio", "value": "[0.55, 0.7)", "effect": "moderately towards this prediction"}, {"feature": "age", "value": "[35, 42)", "effect": "slightly towards this prediction"}], "likelihood": "very likely"}`,
        counterfactual: `{"flip_possible": true, "changes": [{"feature": "income", "from": "[1.8e+04, 2.4e+04)", "to": "[4.5e+04, 5e+04)"}], "minimum_number_of_changes": 1, "new_predicted_class": "loan approved", "flip_holds_even_if_the_model_is_somewhat_off": true, "likelihood_of_new_predicted_class": "very likely"}`,
        sufficient_reason: `{"exists": true, "features": ["income", "debt_ratio"], "features_not_needed": ["age"]}`,
        completions: `{"steps": [{"known_features": ["income"], "share_with_same_prediction": "likely"}, {"known_features": ["income", "debt_ratio"], "share_with_same_prediction": "almost certain"}]}`,
    }
}

function live_data_scientist_facts() -> LocalFactTexts {
    LocalFactTexts {
        prediction: `{"predicted_class": "loan denied", "contributions_largest_first": [{"feature": "income", "value": "[1.8e+04, 2.4e+04)", "score": 1.12, "std": 0.15}, {"feature": "debt_ratio", "value": "[0.55, 0.7)", "score": 0.31, "std": 0.08}, {"feature": "age", "value": "[35, 42)", "score": 0.04, "std": 0.03}], "probability": 0.83, "confidence_interval": [0.74, 0.89], "confidence_level": 0.95, "logit": 1.37}`,
        counterfactual: `{"flip_possible": true, "changes": [{"feature": "income", "from": "[1.8e+04, 2.4e+04)", "to": "[4.5e+04, 5e+04)", "score_change": -2.91}], "minimum_number_of_changes": 1, "new_predicted_class": "loan approved", "flip_holds_at_confidence": 0.95, "probability_of_new_predicted_class": 0.82}`,
        sufficient_reason: `{"exists": true, "features": ["income", "debt_ratio"], "features_not_needed": ["age"], "guaranteed_margin_logit": 0.42}`,
        completions: `{"steps": [{"known_features": ["income"], "share_with_same_prediction": 0.71}, {"known_features": ["income", "debt_ratio"], "share_with_same_prediction": 1.0}]}`,
    }
}

// Prints the reply of a live test. An assertion reports only "expected true", so the log is how a failed run shows
// whether the model or the check is wrong.
function log_reply(reply: string) -> null {
    log.info({ "reply": reply }, event_name = "live_reply")
}

testset "Live" {
    test "layperson counterfactual: no model numbers, within the sentence limit, names the input" {
        let text = LocalCounterfactual(
            "The model denies loans mainly because of low income and a high debt ratio.",
            live_layperson_facts(),
            "layperson",
            3,
            default_graph_options(),
        );
        log_reply(text);
        assert.is_true(text.includes("income"));
        assert.is_true(!text.includes("0.83") && !text.includes("1.37") && !text.includes("2.91"));
        assert.is_true(!text.includes("%"));
        assert.is_true(count_sentences(text) <= 3)
    }

    test "layperson sufficient reason: says the other inputs cannot change the outcome" {
        let text = LocalSufficientReason(
            "The model denies loans mainly because of low income and a high debt ratio.",
            live_layperson_facts(),
            "layperson",
            3,
            default_graph_options(),
        );
        log_reply(text);
        assert.is_true(text.to_lower_case().includes("income"));
        assert.is_true(text.to_lower_case().includes("age"));
        assert.is_true(!text.includes("0.04"));
        assert.is_true(count_sentences(text) <= 3)
    }

    test "layperson completions: approximate wording and no percentages" {
        let text = LocalCompletions(
            "The model denies loans mainly because of low income and a high debt ratio.",
            live_layperson_facts(),
            "layperson",
            3,
            default_graph_options(),
        );
        log_reply(text);
        assert.is_true(!text.includes("71") && !text.includes("%"));
        assert.is_true(count_sentences(text) <= 3)
    }

    test "data scientist counterfactual: keeps the figures from the facts" {
        let text = LocalCounterfactual(
            "The model denies loans mainly because of low income and a high debt ratio.",
            live_data_scientist_facts(),
            "data_scientist",
            3,
            default_graph_options(),
        );
        log_reply(text);
        // the target bin "[4.5e+04, 5e+04)" may be written as "45,000", "45 000" or "45k"
        assert.is_true(baml.regex.new("2\\.91|4\\.5e\\+?04|45[,. ]?000|45k").has_match(text))
    }

    test "explain_local: three explanations, each non-empty, usage reported" {
        let result = explain_local(
            "The model denies loans mainly because of low income and a high debt ratio.",
            live_layperson_facts(),
            "layperson",
            3,
            default_graph_options(),
        );
        log_reply(result.counterfactual);
        log_reply(result.sufficient_reason);
        log_reply(result.completions);
        assert.is_true(result.counterfactual.length() > 0);
        assert.is_true(result.sufficient_reason.length() > 0);
        assert.is_true(result.completions.length() > 0);
        assert.is_true(result.usage.input_tokens > 0);
        assert.is_true(result.usage.output_tokens > 0)
    }

    test "describe_graph_cot: names the feature and respects the sentence limit" {
        let graph = "{\"feature_name\": \"age\", \"type\": \"continuous\", \"mean\": {\"18-30\": 0.8, \"30-50\": 0.0, \"50-90\": -0.6}}";
        let result = describe_graph_cot(graph, 3, default_graph_options());
        log_reply(result.text);
        assert.is_true(result.text.to_lower_case().includes("age"));
        assert.is_true(count_sentences(result.text) <= 4);
        assert.is_true(result.usage.input_tokens > 0)
    }

    test "describe_graph_cot: a multiclass graph is read class against class" {
        // "young" leads at low ages and "old" at high ages; a good answer names both classes and the change of lead
        let graph = "Feature Name: age\nFeature Type: continuous\nClass: young\nMeans: {\"(18, 30)\": 1.5, \"(30, 50)\": 0.0, \"(50, 90)\": -1.5}\nClass: middle\nMeans: {\"(18, 30)\": 0.0, \"(30, 50)\": 0.5, \"(50, 90)\": 0.0}\nClass: old\nMeans: {\"(18, 30)\": -1.5, \"(30, 50)\": -0.5, \"(50, 90)\": 1.5}";
        let result = describe_graph_cot(graph, 4, default_graph_options());
        log_reply(result.text);
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
        log_reply(result.text);
        assert.is_true(result.text.to_lower_case().includes("age"));
        assert.is_true(result.usage.output_tokens > 0)
    }

    test "answer_benchmark_question: the final turn follows the requested format" {
        let question = "A model's graph for feature x has mean value 2.0 for x < 5 and -1.0 for x >= 5. Is the function decreasing in x? Think step by step.";
        let result = answer_benchmark_question(question, "'yes' or 'no'");
        log_reply(result.text);
        // The step function is decreasing but not strictly, so a careful model may answer either way. The LiveBenchmark
        // tests check the answers; this test checks only that the final turn is the bare answer.
        let answer = result.text.trim().to_lower_case();
        assert.is_true(baml.regex.new("^\\W*(yes|no)\\W*$").has_match(answer))
    }
}
