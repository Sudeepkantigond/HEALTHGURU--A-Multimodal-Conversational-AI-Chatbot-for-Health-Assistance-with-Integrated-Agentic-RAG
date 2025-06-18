from chains.rag_chain import *
from .state import AgentState
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document

class GradeDocument(BaseModel):
    score: str = Field(
        ...,
        description=(
            "Does this document fully or partially answer the user's query?\n"
            "Answer 'Yes' if the document contains relevant, useful, or directly related information.\n"
            "Answer 'No' if the document does not help answer the query or is off-topic.\n"
        )
    )


def retrieve(state: AgentState):
    print("Entering retrieve")
    documents = retriever.invoke(state["enhanced_query"])
    # print(f"retrieve: Retrieved {len(documents)} documents")
    state["documents"] = documents
    return state

def retrieval_grader(state: AgentState):
    print("Entering retrieval_grader")
    system_message = SystemMessage(
    content="""
    You are a grader assessing the relevance of a retrieved document to a user question.
    Respond only with 'Yes' or 'No'.

    Respond 'Yes' only if the document directly answers the user's question AND is on the correct topic.

    Do NOT respond 'Yes' if:
    - The document discusses unrelated diseases or topics (e.g., HIV vs COVID-19).
    - The symptoms mentioned are general and not clearly tied to the topic in the question.
    - The document content is ambiguous or off-topic.

    Be strict. Respond 'No' if unsure.
    """
)

    structured_llm = llm.with_structured_output(GradeDocument)

    relevant_docs = []
    for doc in state["documents"]:
        human_message = HumanMessage(
            content=f"User question: {state['enhanced_query']}\n\nRetrieved document:\n{doc.page_content}"
        )
        grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        grader_llm = grade_prompt | structured_llm
        result = grader_llm.invoke({})
        # print(
        #     f"Grading document: {doc.page_content[:30]}... Result: {result.score.strip()}"
        # )
        if result.score.strip().lower() == "yes":
            relevant_docs.append(doc)
    state["documents"] = relevant_docs
    state["proceed_to_generate"] = len(relevant_docs) > 0
    print(f"retrieval_grader: proceed_to_generate = {state['proceed_to_generate']}")
    return state

def websearch(state: AgentState):
    print("Entering web_search fallback")
    
    try:
        results = tavily_search.invoke({"query": state["enhanced_query"]})
    except Exception as e:
        print(f"web_search: Tavily API failed - {e}")
        state["documents"] = []
        state["proceed_to_generate"] = False
        return state

    if not results:
        print("web_search: No results returned from Tavily.")
        state["documents"] = []
        state["proceed_to_generate"] = False
        return state

    docs = [
        Document(
            page_content=res["content"],
            metadata={"source": res["url"], "source_type": "websearch"}
        )
        for res in results if res.get("content")
    ]

    print(f"From the websearch")
    # print(docs)

    state["documents"] = docs
    state["proceed_to_generate"] = len(docs) > 0
    return state
