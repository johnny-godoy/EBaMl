// Live benchmark tests: a fixed sample of the questions in benchmark/data/, asked through answer_benchmark_question.
// Each question has one correct answer, so a failure means a wrong answer, not a strict check. The full benchmark
// (benchmark/run.py) measures the accuracy over every question; these tests only catch a prompt change that breaks
// the answers. The selector root::Live also matches the testset "LiveBenchmark", so offline runs exclude these tests
// and live runs include them. `baml test` runs from the repository root, and the paths of the data files are relative
// to it.
//
// The sample takes the first questions of each solution in the file, not the questions that a model answers well,
// because a sample of easy wins cannot catch a regression. The answer formats and the parsers follow `_SPECS` in
// src/ebaml/benchmark.py. Keep them equal, so that these tests score an answer as the benchmark does.

// The questions of `task` at `indices`, as [question, solution] pairs.
function benchmark_sample(task: string, indices: int[]) -> string[][] {
    let questions = baml.json.decode<string[][]>(baml.fs.read(`benchmark/data/${task}.json`));
    indices.map((i) -> {
        questions[i]
    })
}

// The letter of the chosen option, as `_parse_choice` reads it: "a)" anywhere, or a bare "a" at the start.
function benchmark_choice(answer: string) -> string? {
    let lowered = answer.to_lower_case();
    let found = baml.regex.new("\\b([a-e])\\)").match_once(lowered)
        ?? baml.regex.new("^\\W*([a-e])\\b").match_once(lowered);
    if (found == null) {
        return null;
    };
    found.groups[1]?.text
}

// The monotonicity label, as `_parse_monotonicity` reads it. "not monotone" goes first, because an answer may say
// "not increasing".
function benchmark_monotonicity(answer: string) -> string? {
    let lowered = answer.to_lower_case();
    for (let label in ["not monotone", "increasing", "decreasing"]) {
        if (lowered.includes(label)) {
            return label;
        }
    }
    ;
    null
}

// The answer of the model to one question, or "" if the model stops without a finished answer, for example when a
// reasoning model uses up the token limit. The question then counts as wrong, as an answer that the benchmark cannot
// parse does, and the other questions of the sample still count. Other errors (a missing key, a rate limit) still fail
// the test, because they say nothing about the answers.
function benchmark_answer(question: string, answer_format: string, client: ai.Client?) -> string {
    answer_benchmark_question(question, answer_format, client = client).text catch (e) {
        ai.errors.FinishReasonError => ""
    }
}

// Asks every question of `sample`, two at a time to stay below the rate limit of the free router, and returns the
// number of answers that `parse` reads as the solution. `client` serves the offline test; the live tests omit it.
function benchmark_correct(
    sample: string[][],
    answer_format: string,
    parse: (string) -> string?,
    client: ai.Client? = null,
) -> int {
    let correct = 0;
    let i = 0;
    while (i < sample.length()) {
        let batch = sample.slice(i, i + 2);
        let replies: string[] = await baml.future.all(batch.map((pair) => { spawn { benchmark_answer(pair[0], answer_format, client) } }));
        let j = 0;
        while (j < batch.length()) {
            let answer = replies[j];
            let ok = parse(answer) == parse(batch[j][1]);
            log.info(
                { "answer": answer, "solution": batch[j][1], "correct": ok },
                event_name = "benchmark_answer",
            );
            if (ok) {
                correct = correct + 1;
            };
            j = j + 1;
        }
        ;
        i = i + 2;
    }
    ;
    correct
}

testset "LiveBenchmark" {
    // chance is 2 of 6 (three labels); 4 of 6 lets one or two hard questions fail
    test "benchmark monotonicity: at least 4 of 6 sample questions are correct" {
        // 2 "Decreasing", 2 "Increasing", 2 "Not monotone"
        let sample = benchmark_sample("monotonicity", [17, 37, 4, 7, 0, 8]);
        let correct = benchmark_correct(sample, "'Increasing', 'Decreasing' or 'Not monotone'", benchmark_monotonicity);
        assert.is_true(correct >= 4)
    }

    // chance is about 1 of 6 (five options); 4 of 6 lets one or two hard questions fail
    test "benchmark function recognition: at least 4 of 6 sample questions are correct" {
        // the first question of each solution a) to e), and a second c)
        let sample = benchmark_sample("function-recognition", [0, 1, 2, 7, 10, 14]);
        let correct = benchmark_correct(sample, "the letter of the function, for example 'a)'", benchmark_choice);
        assert.is_true(correct >= 4)
    }
}

testset "BenchmarkChecks" {
    test "benchmark_sample reads the question and the solution at each index" {
        let sample = benchmark_sample("monotonicity", [17, 4]);
        assert.equal(sample[0][1], "Decreasing");
        assert.equal(sample[1][1], "Increasing");
        assert.is_true(sample[0][0].includes("Think step by step."))
    }

    test "benchmark_choice reads the letter like the benchmark scorer" {
        assert.equal(benchmark_choice("**C)**"), "c");
        assert.equal(benchmark_choice("(b)"), "b");
        assert.equal(benchmark_choice("a"), "a");
        assert.equal(benchmark_choice("none of them"), null)
    }

    test "benchmark_monotonicity reads the label like the benchmark scorer" {
        assert.equal(benchmark_monotonicity("Not monotone."), "not monotone");
        assert.equal(benchmark_monotonicity("Monotone decreasing"), "decreasing");
        assert.equal(benchmark_monotonicity("It goes up and down"), null)
    }

    test "benchmark_correct counts the answers that match the solution" {
        // the sample holds 2 "Decreasing" questions and an odd count, so the last batch has one question
        let sample = benchmark_sample("monotonicity", [17, 4, 0, 37, 7]);
        let correct = benchmark_correct(
            sample,
            "'Increasing', 'Decreasing' or 'Not monotone'",
            benchmark_monotonicity,
            client = ConstantClient { answer: "Decreasing." },
        );
        assert.equal(correct, 2)
    }
    test "benchmark_correct counts an answer that runs out of tokens as wrong" {
        let sample = benchmark_sample("monotonicity", [17, 37]);
        let correct = benchmark_correct(
            sample,
            "'Increasing', 'Decreasing' or 'Not monotone'",
            benchmark_monotonicity,
            client = ExhaustedClient {},
        );
        assert.equal(correct, 0)
    }
}

// Ends every call at the token limit with no text, as a reasoning model does when its reasoning uses up the limit.
class ExhaustedClient {
    implements ai.Client {
        function provider(self) -> string {
            "exhausted"
        }

        function requested_model(self) -> string {
            "exhausted"
        }

        function render(self, input: ai.ModelTurnInput) -> baml.http.Request {
            baml.http.Request.from_url("http://exhausted.invalid")
        }

        function invoke(self, input: ai.ModelTurnInput) -> ai.ModelTurn {
            let turn = scripted_turn("", 1, 1);
            turn.stop_reason = ai.content.StopReason.MaxTokens;
            turn
        }
    }
}

// Answers every call with `answer`, so the score of a sample is the number of questions with that solution.
class ConstantClient {
    answer: string,

    implements ai.Client {
        function provider(self) -> string {
            "constant"
        }

        function requested_model(self) -> string {
            "constant"
        }

        function render(self, input: ai.ModelTurnInput) -> baml.http.Request {
            baml.http.Request.from_url("http://constant.invalid")
        }

        function invoke(self, input: ai.ModelTurnInput) -> ai.ModelTurn {
            scripted_turn(self.answer, 1, 1)
        }
    }
}
