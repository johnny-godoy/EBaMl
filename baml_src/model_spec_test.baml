// Offline tests for model_spec.baml: building a client from a `ModelSpec` sends no request.

function spec_for(provider: string) -> ModelSpec {
    ModelSpec {
        provider: provider,
        model: "m",
        api_key_env: null,
        base_url: null,
        temperature: null,
        max_output_tokens: null,
        headers: null,
    }
}

test "every provider of the table builds a client with no network access" {
    let providers = client_builders().keys();
    assert.equal(providers.length(), 9);
    for (let provider in providers) {
        assert.is_true(build_client(spec_for(provider), short_defaults()).id().includes("m"));
    }
}

test "the table lists the BAML provider prefixes" {
    let providers = client_builders().keys();
    for (
        let prefix in [
            "openai",
            "openai-generic",
            "azure",
            "openrouter",
            "ollama",
            "anthropic",
            "google",
            "vertex",
            "bedrock",
        ]
    ) {
        assert.is_true(providers.includes(prefix));
    }
}

test "a client with every optional field set builds" {
    let spec = ModelSpec {
        provider: "openai-generic",
        model: "m",
        api_key_env: "GATEWAY_KEY",
        base_url: "http://gateway.invalid/v1",
        temperature: 0.2,
        max_output_tokens: 100,
        headers: { "x-team": "ebm" },
    };
    assert.is_true(build_client(spec, short_defaults()).id().includes("m"))
}

test "resolve_settings lets the user values win over the defaults of the call" {
    let spec = ModelSpec {
        provider: "openai",
        model: "m",
        api_key_env: null,
        base_url: null,
        temperature: 0.1,
        max_output_tokens: 50,
        headers: null,
    };
    let settings = resolve_settings(spec, long_defaults());
    assert.equal(settings.temperature, 0.1);
    assert.equal(settings.max_output_tokens, 50)
}

test "resolve_settings takes the defaults of the call for null fields" {
    let settings = resolve_settings(spec_for("openai"), benchmark_defaults());
    assert.equal(settings.temperature, 0.0);
    assert.equal(settings.max_output_tokens, 8000)
}

test "the call defaults differ by call as the declared clients do" {
    assert.equal(short_defaults().max_output_tokens, 6000);
    assert.equal(long_defaults().max_output_tokens, 8000);
    assert.equal(short_defaults().temperature, 0.7);
    assert.equal(benchmark_defaults().temperature, 0.0)
}

// The message of the error that `build_client` throws, or "built" if it throws none.
function failure_message(provider: string) -> string {
    let built = build_client(spec_for(provider), short_defaults()) catch (e) {
        baml.errors.InvalidArgument => {
            return e.message;
        },
    };
    "built"
}

function client_id(client: ai.Client?) -> string {
    if (client == null) {
        return "null";
    };
    client.id()
}

test "an unknown provider throws InvalidArgument that lists the supported providers" {
    let message = failure_message("nope");
    assert.is_true(message.includes("nope"));
    assert.is_true(message.includes("anthropic"));
    assert.is_true(message.includes("openai-generic"))
}

test "a known provider throws no error" {
    assert.equal(failure_message("ollama"), "built")
}

test "check_model accepts a known provider and rejects an unknown one" {
    assert.is_true(check_model(spec_for("ollama")));
    let outcome = check_model(spec_for("nope")) catch (e) {
        baml.errors.InvalidArgument => false,
    };
    assert.equal(outcome, false)
}

test "select_client prefers the client over the model" {
    let scripted = ScriptedClient { calls: 0 };
    let selected = select_client(scripted, spec_for("ollama"), short_defaults());
    assert.equal(client_id(selected), "scripted")
}

test "select_client builds the client of the model" {
    let selected = select_client(null, spec_for("ollama"), short_defaults());
    assert.is_true(client_id(selected).includes("ollama"))
}

test "select_client returns null with no client and no model, so the declared client runs" {
    assert.equal(client_id(select_client(null, null, short_defaults())), "null")
}

test "describe_graph_cot sends every turn to the client when both client and model are set" {
    let scripted = ScriptedClient { calls: 0 };
    let result = describe_graph_cot("G", 7, default_graph_options(), client = scripted, model = spec_for("ollama"));
    assert.equal(result.text, "R");
    assert.equal(scripted.calls, 3)
}
