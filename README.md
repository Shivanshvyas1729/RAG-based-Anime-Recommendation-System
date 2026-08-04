"# RAG-based-Anime-Recommendation-System" 
LangChain Tool Calling Flow (Short Summary)
1. Bind the tool to the model
model_with_tools = model.bind_tools([anime_retriever_tool])

This tells the LLM what tools are available.

2. User asks a question
messages = [
    HumanMessage(content="Recommend anime like Naruto")
]
3. Model decides whether to use a tool
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

The model doesn't execute the tool. It only returns something like:

ai_msg.tool_calls
# [{
#   "name": "anime_retriever_tool",
#   "args": {"query": "Recommend anime like Naruto"},
#   "id": "call_123"
# }]
4. Execute the tool yourself
for tool_call in ai_msg.tool_calls:
    result = anime_retriever_tool.invoke(tool_call["args"])

    messages.append(
        ToolMessage(
            content=result,
            tool_call_id=tool_call["id"]
        )
    )

Here your Python code runs the tool and sends its output back to the conversation.

5. Ask the model again
final_response = model_with_tools.invoke(messages)

print(final_response.content)

Now the LLM has the retrieved information and can generate the final answer.

Complete Flow
User
   │
   ▼
LLM
   │
   ▼
tool_calls
   │
   ▼
Python executes tool
   │
   ▼
ToolMessage
   │
   ▼
LLM
   │
   ▼
Final Answer

Remember: The model requests tool usage; your Python code executes the tool and returns the result to the model.