// Offline tests of recourse.baml: the prompt of the difficulty proposal, and the parse of a reply with its usage.

// Answers every call with `reply` and reports 10 input / 5 output tokens.
class ReplyClient {
    reply: string,

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
            scripted_turn(self.reply, 10, 5)
        }
    }
}

const INPUTS = `
    - "income" (continuous), current value 40
    - "age" (continuous), current value 35
`;

test "propose_difficulty parses one rating for each input and counts the tokens" {
    let reply = `[{"feature": "income", "difficulty": "Easy", "reason": "A raise is possible."}, {"feature": "age", "difficulty": "VeryHard", "reason": "Age cannot change, so mark it as immutable."}]`;
    let proposal = propose_difficulty(INPUTS, default_graph_options(), client = ReplyClient {
        reply: reply,
    });
    assert.equal(
        proposal.difficulties.map((rating) -> { rating.difficulty }),
        [Difficulty.Easy, Difficulty.VeryHard],
    );
    assert.equal(proposal.difficulties[0].feature, "income");
    assert.equal(proposal.usage.input_tokens, 10);
    assert.equal(proposal.usage.output_tokens, 5)
}

test "the difficulty prompt holds the data before the instructions and the scale" {
    let prompt = ProposeFeatureDifficulty@spec(INPUTS, default_graph_options())
        .prompt()
        .text();
    assert.is_true(prompt.includes("\"income\" (continuous), current value 40"));
    assert.is_true(
        position_of(prompt, "<inputs>") < position_of(prompt, "<instructions>"),
    );
    for (let level in ["- Easy:", "- Moderate:", "- Hard:", "- VeryHard:"]) {
        assert.is_true(prompt.includes(level));
    }
    assert.is_true(prompt.includes("VeryHard"))
}

test "the difficulty prompt rates the effort for the person and leaves immutability to the user" {
    // The exact computation chooses the changes, so the model must not rate how much a change helps the prediction.
    let prompt = ProposeFeatureDifficulty@spec(INPUTS, default_graph_options())
        .prompt()
        .text();
    assert.is_true(
        prompt.includes("not how much the change helps the prediction"),
    );
    assert.is_true(prompt.includes("the user should mark it as immutable"));
    assert.is_true(prompt.includes(
        "give its name exactly as the text between the quotes, without the quotes and without the type",
    ));
    assert.is_true(prompt.includes("corrects it before any plan is computed"))
}

test "the difficulty prompt asks for the name between the quotes, with no type, in the instructions and the examples" {
    // A live model copied "monthly savings (continuous)" from a list with no quotes, and the program matched no name.
    let prompt = ProposeFeatureDifficulty@spec(INPUTS, default_graph_options())
        .prompt()
        .text();
    assert.is_true(prompt.includes(
        "gives the name of an input in quotes, then its type in brackets",
    ));
    assert.is_true(prompt.includes(
        "Input: \"savings\" (continuous), current value 1200. Name: savings.",
    ));
    assert.is_true(prompt.includes(
        "the text between the quotes in the list of inputs, without the quotes and without the type",
    ))
}
