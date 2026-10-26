// A model chosen at run time. Python sends a `ModelSpec` (plain data) and BAML builds the client, because a client
// object cannot cross the Python/BAML boundary.

class ModelSpec {
    provider: string,
    model: string,
    api_key_env: string?,
    base_url: string?,
    temperature: float?,
    max_output_tokens: int?,
    headers: map<string, string>?,
}

// The settings of one call: used for each field that the `ModelSpec` leaves null.
class CallDefaults {
    temperature: float,
    max_output_tokens: int,
}

// These three must equal the `temperature` and `max_tokens` of `RouterBase`, `RouterLongBase` and
// `RouterBenchmarkBase` in talk_to_ebm.baml.
function short_defaults() -> CallDefaults {
    CallDefaults { temperature: 0.7, max_output_tokens: 6000 }
}

// The first turn of each conversation allows a longer answer.
function long_defaults() -> CallDefaults {
    CallDefaults { temperature: 0.7, max_output_tokens: 8000 }
}

function benchmark_defaults() -> CallDefaults {
    CallDefaults { temperature: 0.0, max_output_tokens: 8000 }
}

// A field that the user set wins. Otherwise the default of the call applies.
function resolve_settings(spec: ModelSpec, defaults: CallDefaults) -> CallDefaults {
    CallDefaults {
        temperature: spec.temperature ?? defaults.temperature,
        max_output_tokens: spec.max_output_tokens ?? defaults.max_output_tokens,
    }
}

// The key is read from the environment when the request is sent. Null selects the default variable of the provider.
function credential(name: string?) -> ai.Credential? {
    if (name == null) {
        return null;
    };
    baml.env.ref(name)
}

function build_openai(spec: ModelSpec, settings: CallDefaults) -> ai.Client {
    openai.ChatClient.new(
        model = spec.model,
        api_key = credential(spec.api_key_env),
        base_url = spec.base_url,
        headers = spec.headers,
        temperature = settings.temperature,
        max_tokens = settings.max_output_tokens,
    )
}

function build_openai_generic(spec: ModelSpec, settings: CallDefaults) -> ai.Client {
    openai.GenericClient.new(
        model = spec.model,
        api_key = credential(spec.api_key_env),
        base_url = spec.base_url,
        headers = spec.headers,
        temperature = settings.temperature,
        max_tokens = settings.max_output_tokens,
    )
}

// Azure serves deployments under names that the owner chose, so `model` names the deployment. `base_url` names it too
// and wins. Without it, the deployment is built on the endpoint in `AZURE_OPENAI_ENDPOINT`.
function build_azure(spec: ModelSpec, settings: CallDefaults) -> openai.AzureClient {
    openai.AzureClient.new(
        model = spec.model,
        deployment_id = spec.model,
        api_key = credential(spec.api_key_env),
        base_url = spec.base_url,
        headers = spec.headers,
        temperature = settings.temperature,
        max_tokens = settings.max_output_tokens,
    )
}

function build_openrouter(spec: ModelSpec, settings: CallDefaults) -> ai.Client {
    openai.OpenRouterClient.new(
        model = spec.model,
        api_key = credential(spec.api_key_env),
        base_url = spec.base_url,
        headers = spec.headers,
        temperature = settings.temperature,
        max_tokens = settings.max_output_tokens,
    )
}

function build_ollama(spec: ModelSpec, settings: CallDefaults) -> ai.Client {
    openai.OllamaClient.new(
        model = spec.model,
        api_key = credential(spec.api_key_env),
        base_url = spec.base_url,
        headers = spec.headers,
        temperature = settings.temperature,
        max_tokens = settings.max_output_tokens,
    )
}

function build_anthropic(spec: ModelSpec, settings: CallDefaults) -> ai.Client {
    anthropic.Client.new(
        model = spec.model,
        api_key = credential(spec.api_key_env),
        base_url = spec.base_url,
        headers = spec.headers,
        temperature = settings.temperature,
        max_tokens = settings.max_output_tokens,
    )
}

// Gemini names its output limit `max_output_tokens`.
function build_google(spec: ModelSpec, settings: CallDefaults) -> ai.Client {
    google.GeminiClient.new(
        model = spec.model,
        api_key = credential(spec.api_key_env),
        base_url = spec.base_url,
        headers = spec.headers,
        temperature = settings.temperature,
        max_output_tokens = settings.max_output_tokens,
    )
}

function build_vertex(spec: ModelSpec, settings: CallDefaults) -> ai.Client {
    google.VertexClient.new(
        model = spec.model,
        api_key = credential(spec.api_key_env),
        base_url = spec.base_url,
        headers = spec.headers,
        temperature = settings.temperature,
        max_output_tokens = settings.max_output_tokens,
    )
}

// Bedrock signs requests with AWS credentials, so `api_key_env` has no effect.
function build_bedrock(spec: ModelSpec, settings: CallDefaults) -> ai.Client {
    aws.BedrockClient.new(
        model = spec.model,
        endpoint_url = spec.base_url,
        headers = spec.headers,
        temperature = settings.temperature,
        max_tokens = settings.max_output_tokens,
    )
}

// One row for each provider. A new provider is a new row. The keys are the BAML provider prefixes.
function client_builders() -> map<string, (ModelSpec, CallDefaults) -> ai.Client throws never> {
    {
        "openai": build_openai,
        "openai-generic": build_openai_generic,
        "azure": build_azure,
        "openrouter": build_openrouter,
        "ollama": build_ollama,
        "anthropic": build_anthropic,
        "google": build_google,
        "vertex": build_vertex,
        "bedrock": build_bedrock,
    }
}

// The provider prefixes that `ModelSpec.provider` accepts, in table order.
function supported_providers() -> string[] {
    client_builders().keys()
}

// Builds the client of `spec` with the retry policy of the default clients.
function build_client(spec: ModelSpec, defaults: CallDefaults) -> ai.Client {
    let builders = client_builders();
    let build = builders.get(spec.provider);
    if (build == null) {
        throw baml.errors.InvalidArgument {
            message: `unknown provider "${spec.provider}"; supported providers: ${supported_providers().join(", ")}`,
        };
    };
    ai.clients.Retry.new(inner = build(spec, resolve_settings(spec, defaults)), max_attempts = 3)
}

// Builds and discards the client, so that Python can reject a bad provider before the first paid call.
function check_model(spec: ModelSpec) -> bool {
    build_client(spec, short_defaults());
    true
}

// Precedence: `client` (the test hook) > `model` > null, which selects the client declared by the function.
function select_client(
    client: ai.Client?,
    model: ModelSpec?,
    defaults: CallDefaults,
) -> ai.Client? {
    if (client != null) {
        return client;
    };
    if (model == null) {
        return null;
    };
    build_client(model, defaults)
}
