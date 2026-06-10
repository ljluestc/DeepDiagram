import unittest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.mindmap import extract_current_code_from_messages


class MindmapIncrementalExtractionTests(unittest.TestCase):
    def test_extracts_latest_code_from_ai_steps(self):
        messages = [
            HumanMessage(content="请先生成一个学习计划思维导图"),
            AIMessage(
                content="",
                additional_kwargs={
                    "steps": [
                        {
                            "type": "tool_start",
                            "name": "create_mindmap",
                            "content": "{}",
                        },
                        {
                            "type": "tool_end",
                            "name": "create_mindmap",
                            "content": "# 学习计划\n## 基础\n- 语法",
                        },
                    ]
                },
            ),
            HumanMessage(content="在基础下面增加‘项目实战’节点"),
        ]

        code = extract_current_code_from_messages(messages)
        self.assertEqual(code, "# 学习计划\n## 基础\n- 语法")

    def test_extracts_code_from_execution_trace_fallback(self):
        messages = [
            HumanMessage(content="请生成一个产品路线图思维导图"),
            AIMessage(
                content=(
                    "### Execution Trace:\n"
                    "agentName: mindmap\n"
                    "toolName: create_mindmap, toolArgs: {}, toolsOutput: "
                    "# 产品路线图\n## Q1\n- 需求调研\n"
                ),
                additional_kwargs={},
            ),
            HumanMessage(content="新增一个 Q2 分支"),
        ]

        code = extract_current_code_from_messages(messages)
        self.assertEqual(code, "# 产品路线图\n## Q1\n- 需求调研")

    def test_returns_empty_when_no_mindmap_code_present(self):
        messages = [
            HumanMessage(content="你好"),
            AIMessage(content="这是纯文本回复"),
            HumanMessage(content="继续"),
        ]
        self.assertEqual(extract_current_code_from_messages(messages), "")


if __name__ == "__main__":
    unittest.main()
