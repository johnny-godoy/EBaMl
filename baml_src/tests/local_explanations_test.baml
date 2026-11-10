// Offline tests of local_explanations.baml: the prompts of the three kinds of local explanation, and how explain_local routes their answers.

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
    assert.is_true(
        sufficient_reason_question(4).includes("name the inputs of \"features_not_needed\""),
    );
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

test "the questions state a condition of the facts only through its key" {
    // Each condition depends on an option, so a fixed sentence about it is false for some option value.
    let counterfactual = counterfactual_question(3);
    let sufficient = sufficient_reason_question(3);
    let completions = completions_question(3);
    // confidence_level=0 moves no score
    assert.is_true(
        counterfactual.includes("\"flip_holds_even_if_the_model_is_somewhat_off\" is true"),
    );
    assert.is_true(counterfactual.includes("\"flip_holds_at_confidence\" is above 0"));
    assert.is_true(
        sufficient.includes("\"guarantee_holds_even_if_the_model_is_somewhat_off\" is true"),
    );
    assert.is_true(sufficient.includes("\"guarantee_holds_at_confidence\" is above 0"));
    for (let question in [counterfactual, sufficient]) {
        assert.is_true(question.includes("make no statement about the model's uncertainty"));
        // min_support=0 leaves out no value
        assert.is_true(question.includes("If \"rare_values_left_out\" is true"));
        assert.is_true(!question.includes("(rare extreme values are left out)"))
    }
    ;
    // only the changed features of a counterfactual move by their uncertainty
    assert.is_true(counterfactual.includes("the effects of the changed features"));
    assert.is_true(!counterfactual.includes("holds even if the model's effects"));
    // n_samples accepts any count
    assert.is_true(completions.includes("\"sample_size\""));
    assert.is_true(completions.includes("\"number_of_samples\""));
    assert.is_true(!completions.includes("a large random sample"));
    assert.is_true(!completions.includes("gets more certain"))
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
