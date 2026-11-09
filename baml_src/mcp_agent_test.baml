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
