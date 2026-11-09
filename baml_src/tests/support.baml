// Fakes and prompt readers that the offline tests of several files share: a scripted model turn, scripted clients, and readers of the messages of a rendered prompt.

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
