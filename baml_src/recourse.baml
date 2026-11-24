// Proposals of how hard each change of a recourse plan is. Python turns the difficulty of each input into a cost
// multiplier (ebaml.local.costs.Difficulty) and never runs a search here: the caller reads the proposal first.

// The difficulty of a change of one input, on a fixed scale. The names render into the output format of the prompt.
enum Difficulty {
    Easy,
    Moderate,
    Hard,
    VeryHard,
}

class FeatureDifficulty {
    feature: string @description("The name of the input, exactly as the list of inputs gives it."),
    difficulty: Difficulty,
    reason: string @description("One sentence: why a person finds this change easy or hard."),
}

class DifficultyProposal {
    difficulties: FeatureDifficulty[],
    usage: TokenUsage,
}

function difficulty_system_msg(expert_description: string) -> string {
    `
        You are ${expert_description} You help a person plan which inputs of a prediction they could change, and you judge how hard each change is for that person in the real world.
    `
}

function difficulty_user_msg(features: string, dataset_description: string) -> string {
    `
        <dataset>
        ${dataset_description}
        </dataset>

        <inputs>
        ${features}
        </inputs>

        <instructions>
        Rate how hard it is for this person to change each input from its current value, on this scale:
        - Easy: a change that most people can make soon and at little cost, such as an amount that they choose themselves.
        - Moderate: a change that takes planning, money or some months.
        - Hard: a change that takes years, large costs or the help of other people.
        - VeryHard: a change that is possible only in rare cases, such as a long history or a fact of the past.
        Rate every input in the list once, and use its name exactly as the list gives it, because a program matches the names.
        Judge the effort of the change for a person, not how much the change helps the prediction, because a separate exact computation finds the changes that help.
        If an input cannot change at all, rate it VeryHard and say in the reason that the user should mark it as immutable, because the user decides what is impossible.
        Give one sentence of reason for each rating, because the user reads your proposal and corrects it before any plan is computed.
        </instructions>

        <examples>
        <example>
        Input: savings (continuous), current value 1200. Rating: Easy. Reason: A person can add to their savings within weeks.
        </example>
        <example>
        Input: education (nominal), current value "high school". Rating: Hard. Reason: A new degree takes years of study.
        </example>
        <example>
        Input: number of past defaults (continuous), current value 2. Rating: VeryHard. Reason: The past cannot change, so the user should mark this input as immutable.
        </example>
        </examples>
    `
}

function ProposeFeatureDifficulty(
    features: string,
    opts: GraphPromptOptions,
) -> FeatureDifficulty[] {
    client: Router
    prompt: `
        ${role("system")}
        ${difficulty_system_msg(opts.expert_description)}
        ${role("user")}
        ${difficulty_user_msg(features, opts.dataset_description)}

        ${ctx.output_format()}
    `
}

// Entry point called from Python: one request, with its token usage.
function propose_difficulty(
    features: string,
    opts: GraphPromptOptions,
    client: ai.Client? = null,
    model: ModelSpec? = null,
) -> DifficultyProposal {
    let selected = select_client(client, model, short_defaults());
    let result = ai.Agent.new(client = selected).run(ProposeFeatureDifficulty@spec(features, opts));
    DifficultyProposal { difficulties: result.value, usage: usage_of(result) }
}
