// Offline tests for mcp_agent.baml: a scripted model calls a fake tool, then answers.

// Asks for `get_model_summary` on the first turn, then answers with a text that holds the tool result.
class ToolCallingClient {
    turns: int,
    seen_tools: string[],

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
            self.turns += 1;
            if (self.turns == 1) {
                let turn = scripted_turn("", 10, 5);
                turn.content
                    = [ai.content.ToolUse { id: "call-1", name: "get_model_summary", args: {} }];
                turn.stop_reason = ai.content.StopReason.ToolUse;
                return turn;
            };
            scripted_turn("Report: 2 features.", 20, 8)
        }
    }
}

function fake_summary_tool() -> ai.tools.Tool {
    ai.tools.raw_tool("get_model_summary", "Summarize the model.", { "type": "object", "properties": {} }, (
        args,
    ) -> {
        `{"feature_count": 2}`
    })
}

test "explain_with_tools runs the tool the model asks for and reports it" {
    let scripted = ToolCallingClient { turns: 0, seen_tools: [] };
    let result = explain_with_tools("What matters?", [fake_summary_tool()], client = scripted);
    assert.equal(result.report, "Report: 2 features.");
    assert.equal(result.tool_calls, ["get_model_summary"]);
    assert.equal(scripted.turns, 2);
    assert.equal(result.usage.input_tokens, 30)
}

test "the user message holds the question and the system message forbids invented numbers" {
    assert.is_true(mcp_agent_user_msg("Why?").includes("Why?"));
    assert.is_true(mcp_agent_system_msg().includes("invented number"))
}

test "conversation_text keeps the order and the speaker of each message" {
    let text = conversation_text(
        [
            ChatTurn { speaker: "user", text: "Explain row 3." },
            ChatTurn { speaker: "assistant", text: "Age pushes it up." },
        ],
    );
    assert.equal(
        text,
        "<user>\nExplain row 3.\n</user>\n<assistant>\nAge pushes it up.\n</assistant>",
    )
}

test "the chat message holds the history and the new message" {
    let message = mcp_chat_user_msg("What if income were higher?", [ChatTurn { speaker: "user", text: "Explain row 3." }]);
    assert.is_true(message.includes("Explain row 3."));
    assert.is_true(message.includes("What if income were higher?"));
    assert.is_true(mcp_chat_system_msg().includes("follow-up"))
}

test "chat_with_tools answers a follow-up with the history in the prompt" {
    let scripted = ToolCallingClient { turns: 0, seen_tools: [] };
    let history = [ChatTurn { speaker: "user", text: "Explain row 3." }];
    let result = chat_with_tools("And row 4?", history, [fake_summary_tool()], client = scripted);
    assert.equal(result.report, "Report: 2 features.");
    assert.equal(result.tool_calls, ["get_model_summary"]);
    let prompt = McpChat@spec("And row 4?", history).prompt();
    assert.is_true(message_at(prompt, 1).includes("Explain row 3."))
}

test "the chat message asks for rounded numbers" {
    assert.is_true(mcp_chat_user_msg("Q", []).includes("round them to three digits"))
}
