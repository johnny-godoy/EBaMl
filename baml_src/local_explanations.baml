// Local explanations: Python computes the facts of one prediction deterministically (ebaml.local), BAML turns each
// kind of fact into words. The three kinds are the explainability queries of Barceló et al., "Model Interpretability
// through the Lens of Computational Complexity" (arXiv:2010.12265): minimum change required, sufficient reason and
// count completions. The global explanation of the model (describe_ebm) is prepended to every prompt.

// Facts of one prediction, one JSON text per kind of explanation.
class LocalFactTexts {
    prediction: string,
    counterfactual: string,
    sufficient_reason: string,
    completions: string,
}

class LocalExplanationText {
    counterfactual: string,
    sufficient_reason: string,
    completions: string,
    usage: TokenUsage,
}

function local_system_msg(expert_description: string) -> string {
    `You are ${expert_description}. You explain individual predictions of a Generalized Additive Model (GAM). You receive an explanation of the whole model and facts about one prediction that were computed exactly from the model. State only effects, values and probabilities that appear in these facts, because the reader acts on your words and an invented number would mislead. Answer in plain prose paragraphs, because a person reads your answer.`
}

// The audience decides the vocabulary. The facts of a "layperson" carry no model numbers, but the instruction repeats it.
function audience_guidance(audience: "layperson" | "data_scientist") -> string {
    if (audience == "data_scientist") {
        return "The reader is a data scientist, so be precise. Quote the scores (log-odds contributions), probabilities and confidence intervals from the facts. Say how uncertain a statement is when the facts show wide confidence intervals.";
    };
    "The reader is not technical and knows no statistics, so write in everyday words. Express strength and likelihood in plain words such as 'strongly', 'slightly', 'very likely' or 'about as likely as not'. Mention only values of the person's own inputs (for example their age), because model numbers (scores, log-odds, probabilities, percentages, intervals, standard deviations) mean nothing to this reader."
}

function local_user_msg(
    global_explanation: string,
    prediction: string,
    question: string,
    facts: string,
    audience: "layperson" | "data_scientist",
    dataset_description: string,
) -> string {
    let msg = `
        <global_explanation>
        ${global_explanation}
        </global_explanation>
    `
        + "\n\n";
    if (dataset_description.length() > 0) {
        msg = msg + `<dataset_description>\n${dataset_description}\n</dataset_description>\n\n`;
    };
    msg
        + `
            <prediction>
            ${prediction}
            </prediction>

            <facts>
            ${facts}
            </facts>

            <audience>
            ${audience_guidance(audience)}
            </audience>

            <question>
            ${question}
            </question>
        `
}

function counterfactual_question(num_sentences: int) -> string {
    `
        Explain, in at most ${num_sentences} sentences, what would have to change for the prediction to be different.
                The facts list the smallest set of input changes that flips the prediction (a minimum change required), and whether the flip survives the model's uncertainty.
                This set is exact, not a guess: no smaller set of changes flips the prediction, because the model adds up one independent effect per input.
                Only realistic alternative values were considered (rare extreme values are left out), and the flip holds even if the model's effects are at the unfavourable edge of their confidence interval.
                Suggest only the changes that the facts list, because they are the only ones that the model supports.
                If no change flips the prediction, say that the prediction cannot be changed by altering the inputs.
    `
}

function sufficient_reason_question(num_sentences: int) -> string {
    `
        Explain, in at most ${num_sentences} sentences, which inputs are enough to settle the prediction.
                The facts list a smallest set of inputs (a sufficient reason) that fixes the prediction whatever the other inputs are. State clearly that the remaining inputs cannot change the outcome.
                This is exact: no smaller set of inputs guarantees the outcome, because the model adds up one independent effect per input.
                The guarantee covers every realistic value of the other inputs (rare extreme values are left out) and holds even when the model's effects are at the unfavourable edge of their confidence interval.
                Present it as a statement about this model, not about the real world.
                If there is no such set, say that every input matters.
    `
}

function completions_question(num_sentences: int) -> string {
    `
        Explain, in at most ${num_sentences} sentences, how the outcome gets more certain as the inputs become known.
                The facts give the share of people who would get the same prediction when only the first inputs are known and the rest are unknown (count completions).
                Describe how that share grows and which inputs make the biggest difference.
                Unlike the other two explanations, these shares are estimates from a large random sample that treats the unknown inputs as independent and distributed as in the training data. Describe them as approximate, because they are not guarantees.
    `
}

function LocalCounterfactual(
    global_explanation: string,
    facts: LocalFactTexts,
    audience: "layperson" | "data_scientist",
    num_sentences: int,
    opts: GraphPromptOptions,
) -> string {
    client: Router
    prompt: `
        ${role("system")}
        ${local_system_msg(opts.expert_description)}
        ${role("user")}
        ${local_user_msg(global_explanation, facts.prediction, counterfactual_question(num_sentences), facts.counterfactual, audience, opts.dataset_description)}
    `
}

function LocalSufficientReason(
    global_explanation: string,
    facts: LocalFactTexts,
    audience: "layperson" | "data_scientist",
    num_sentences: int,
    opts: GraphPromptOptions,
) -> string {
    client: Router
    prompt: `
        ${role("system")}
        ${local_system_msg(opts.expert_description)}
        ${role("user")}
        ${local_user_msg(global_explanation, facts.prediction, sufficient_reason_question(num_sentences), facts.sufficient_reason, audience, opts.dataset_description)}
    `
}

function LocalCompletions(
    global_explanation: string,
    facts: LocalFactTexts,
    audience: "layperson" | "data_scientist",
    num_sentences: int,
    opts: GraphPromptOptions,
) -> string {
    client: Router
    prompt: `
        ${role("system")}
        ${local_system_msg(opts.expert_description)}
        ${role("user")}
        ${local_user_msg(global_explanation, facts.prediction, completions_question(num_sentences), facts.completions, audience, opts.dataset_description)}
    `
}

// Entry point called from Python: the three explanations are independent, so they run concurrently.
function explain_local(
    global_explanation: string,
    facts: LocalFactTexts,
    audience: "layperson" | "data_scientist",
    num_sentences: int,
    opts: GraphPromptOptions,
    client: ai.Client? = null,
    model: ModelSpec? = null,
) -> LocalExplanationText {
    let selected = select_client(client, model, short_defaults());
    let counterfactual = spawn {
        run_counted(
            LocalCounterfactual@spec(global_explanation, facts, audience, num_sentences, opts),
            client = selected,
        )
    };
    let sufficient = spawn {
        run_counted(
            LocalSufficientReason@spec(global_explanation, facts, audience, num_sentences, opts),
            client = selected,
        )
    };
    let completions = spawn {
        run_counted(
            LocalCompletions@spec(global_explanation, facts, audience, num_sentences, opts),
            client = selected,
        )
    };
    let done = await baml.future.all([counterfactual, sufficient, completions]);
    LocalExplanationText {
        counterfactual: done[0].text,
        sufficient_reason: done[1].text,
        completions: done[2].text,
        usage: add_usage(add_usage(done[0].usage, done[1].usage), done[2].usage),
    }
}
