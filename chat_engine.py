from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import Config

class ChatEngine:
    """Manages the RAG pipeline using the Groq API and a vector retriever."""
    
    def __init__(self, retriever):
        self.retriever = retriever
        # Initialize Groq LLM
        self.llm = ChatGroq(
            model_name=Config.LLM_MODEL,
            groq_api_key=Config.GROQ_API_KEY,
            temperature=0.3 # low temperature for more grounded answers
        )
        
        # Create the specific prompt for simplifying research papers
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert academic research assistant tasked with helping users understand complex research papers.
            Use the provided context from the research paper to answer the user's question. 
            
            Crucially, you must:
            1. Simplify complex concepts so they are easy to understand for a layperson.
            2. Use analogies if helpful.
            3. Keep your answers concise but comprehensive.
            4. If the answer is not in the provided context, state that you don't know rather than hallucinating.
            
            Context: {context}"""),
            ("human", "{input}")
        ])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": self.retriever | format_docs, "input": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
    def query(self, user_query: str) -> str:
        """
        Sends the user query through the RAG pipeline and returns the simplified answer.
        """
        return self.rag_chain.invoke(user_query)
