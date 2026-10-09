import os
from typing import Any, List, Dict, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from retrieval.fusion import HybridRetriever

# 1. Custom LangChain Retriever wrapping our Day 2 Hybrid Engine
class LangChainClinicalRetriever(BaseRetriever):
    hybrid_engine: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves top chunks from our custom HybridRetriever (Dense + BM25 + RRF)
        and converts them into LangChain Document objects.
        """
        results = self.hybrid_engine.search_hybrid(query=query, top_k=4)
        docs = []
        for r in results:
            doc = Document(
                page_content=r["text"],
                metadata={
                    "drug": r["drug"],
                    "section": r["section"],
                    "page": r["page"],
                    "source": r["source"],
                    "rrf_score": r.get("rrf_score", 0.0),
                    "rrf_rank": r.get("rrf_rank", 0)
                }
            )
            docs.append(doc)
        return docs


# 2. Helper to format LangChain Documents into a clean prompt evidence block
def format_docs_with_metadata(docs: List[Document]) -> str:
    formatted_blocks = []
    for idx, doc in enumerate(docs, start=1):
        block = (
            f"[Evidence #{idx}]\n"
            f"Drug: {doc.metadata.get('drug')}\n"
            f"Section: {doc.metadata.get('section')}\n"
            f"Page: {doc.metadata.get('page')}\n"
            f"Source File: {doc.metadata.get('source')}\n"
            f"Text:\n{doc.page_content}\n"
        )
        formatted_blocks.append(block)
    return "\n----------------------------------------\n".join(formatted_blocks)


# 3. Main Clinical Answer Agent using LCEL (LangChain Expression Language)
class ClinicalAnswerAgent:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Run: $env:GEMINI_API_KEY = 'your_key' in PowerShell."
            )

        print(f"[*] Initializing Clinical LLM ({model_name})...")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0  # Zero temperature for deterministic clinical grounding
        )

        # Initialize our hybrid retrieval engine
        self.hybrid_engine = HybridRetriever()
        self.retriever = LangChainClinicalRetriever(hybrid_engine=self.hybrid_engine)

        # Strict Clinical Grounding Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert Clinical and Pharmaceutical Evidence Assistant.\n"
                "Your role is to answer questions strictly using the provided FDA drug labeling evidence.\n\n"
                "STRICT RULES:\n"
                "1. ONLY state facts directly supported by <CLINICAL_EVIDENCE>.\n"
                "2. Every clinical assertion MUST cite: [Drug, Section, Page X].\n"
                "3. If the evidence is insufficient, state:\n"
                "   'Based on the provided FDA labeling documentation, this information is not specified.'\n"
                "4. Distinguish between INDICATIONS, CONTRAINDICATIONS, and ADVERSE REACTIONS."
            )),
            ("human", (
                "<CLINICAL_EVIDENCE>\n"
                "{context}\n"
                "</CLINICAL_EVIDENCE>\n\n"
                "QUESTION: {question}\n\n"
                "Provide a concise, medically accurate, cited response:"
            ))
        ])

        # Modern LCEL Chain
        self.chain = self.prompt | self.llm | StrOutputParser()
        print("[✓] LangChain Clinical Agent ready!")

    def answer_question(self, question: str) -> Dict[str, Any]:
        print(f"\n[*] Retrieving evidence for: '{question}'...")
        docs = self.retriever.invoke(question)
        context_str = format_docs_with_metadata(docs)

        print("[*] Generating grounded answer with Gemini via LCEL...")
        response_text = self.chain.invoke({
            "context": context_str,
            "question": question
        })

        return {
            "question": question,
            "answer": response_text,
            "evidence": docs
        }


if __name__ == "__main__":
    agent = ClinicalAnswerAgent()

    test_q = "What is the recommended target INR for a patient with mechanical heart valves taking Warfarin?"
    result = agent.answer_question(test_q)

    print("\n" + "=" * 60)
    print("CLINICAL QUESTION:")
    print(result["question"])
    print("=" * 60)
    print("\nAGENT ANSWER:")
    print(result["answer"])
    print("=" * 60)