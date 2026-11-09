// Offline tests of the token accounting in talk_to_ebm.baml: usage sums, the samples of the requests, run_counted and sample_graph_tokens.

test "add_usage sums both counters" {
    let total = add_usage(
        TokenUsage { input_tokens: 1, output_tokens: 2, samples: [] },
        TokenUsage { input_tokens: 10, output_tokens: 20, samples: [] },
    );
    assert.equal(total.input_tokens, 11);
    assert.equal(total.output_tokens, 22)
}

test "add_usage keeps the operands independent" {
    let zero = TokenUsage { input_tokens: 0, output_tokens: 0, samples: [] };
    let usage = TokenUsage { input_tokens: 3, output_tokens: 4, samples: [] };
    assert.equal(add_usage(zero, usage).input_tokens, 3);
    assert.equal(add_usage(zero, usage).output_tokens, 4);
    assert.equal(add_usage(usage, usage).input_tokens, 6);
    assert.equal(add_usage(usage, usage).output_tokens, 8)
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
