import os
import logging
from typing import TypedDict, Annotated, Dict
from operator import itemgetter

from dotenv import load_dotenv
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableParallel
from langgraph.graph import StateGraph, END
import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize language model
llm = ChatOpenAI(temperature=0.7, model=os.getenv("OPENAI_MODEL", "gpt-4"))

# Initialize search tool
search = DuckDuckGoSearchResults()

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


class ResearchTool(BaseTool):
    name = "Research"
    description = "Use this tool to gather information about a given topic"

    def _run(self, query: str) -> str:
        try:
            search_results = search.run(query)
            urls = [r for r in search_results.split() if r.startswith("http")]
            loader = WebBaseLoader(urls[:5])  # Load top 5 search results
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
            return qa_chain.invoke(query)["result"]
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred during the research: {str(e)}")
            return f"An error occurred during the research: {str(e)}"


class WriterTool(BaseTool):
    name = "Writer"
    description = "Use this tool to write content based on research for specific platforms"

    def _run(self, research: str, platform: str = "Medium") -> str:
        prompt_templates = {
            "LinkedIn": "Based on the following research, write a professional and insightful LinkedIn post (max 3000 characters):\n\n{research}",
            "Medium": "Based on the following research, write a comprehensive and engaging Medium article:\n\n{research}",
            "Twitter": "Based on the following research, write a concise and impactful tweet thread (max 280 characters per tweet, up to 5 tweets):\n\n{research}"
        }
        prompt = PromptTemplate(
            input_variables=["research"],
            template=prompt_templates.get(platform, prompt_templates["Medium"])
        )
        return llm.invoke(prompt.format(research=research)).content


class CritiqueTool(BaseTool):
    name = "Critique"
    description = "Use this tool to critique and improve content for specific platforms"

    def _run(self, content: str, platform: str = "Research") -> str:
        prompt_templates = {
            "LinkedIn": "Critique the following LinkedIn post and suggest improvements, focusing on professional tone and engagement:\n\n{content}",
            "Medium": "Critique the following Medium article and suggest improvements, focusing on depth, structure, and reader engagement:\n\n{content}",
            "Twitter": "Critique the following tweet thread and suggest improvements, focusing on conciseness and impact:\n\n{content}",
            "Research": "Critique the following research and suggest improvements, focusing on comprehensiveness and relevance:\n\n{content}"
        }
        prompt = PromptTemplate(
            input_variables=["content"],
            template=prompt_templates.get(platform, prompt_templates["Research"])
        )
        return llm.invoke(prompt.format(content=content)).content


class PublishTool(BaseTool):
    name = "Publish"
    description = "Use this tool to publish content to different platforms"

    def _run(self, content: Dict[str, str]) -> dict:
        # Simulate publishing to different platforms
        results = {}
        for platform, platform_content in content.items():
            if platform == "Twitter":
                # Split content into tweets
                tweets = [platform_content[i:i + 280] for i in range(0, len(platform_content), 280)]
                results[platform] = f"Tweet thread published: {len(tweets)} tweets"
            else:
                results[platform] = f"Content published to {platform}"
        return results


# Define state types
class State(TypedDict):
    topic: str
    research_results: str
    draft_content: Dict[str, str]
    research_critique: str
    improved_research: str
    content_critique: Dict[str, str]
    improved_content: Dict[str, str]
    publish_result: dict
    final_content: Dict[str, str]


# Define input schemas
class ResearchInput(BaseModel):
    topic: str = Field(description="The topic to research")


class WriteInput(BaseModel):
    research_results: str = Field(description="The research results to base the content on")


class CritiqueInput(BaseModel):
    content: Dict[str, str] = Field(description="The content to critique and improve for each platform")


class PublishInput(BaseModel):
    improved_content: Dict[str, str] = Field(description="The improved content to publish for each platform")


# Define the workflow functions
def perform_research(state: State, research_tool: ResearchTool) -> dict:
    topic = state["topic"]
    research_results = research_tool.run(topic)
    return {"research_results": research_results}


def critique_research(state: Annotated[State, RunnableParallel(research_results=itemgetter("research_results"))],
                      critique_tool: CritiqueTool) -> dict:
    research_critique = critique_tool.run(state["research_results"], platform="Research")
    return {"research_critique": research_critique}


def improve_research(state: Annotated[State, RunnableParallel(research_results=itemgetter("research_results"),
                                                              research_critique=itemgetter("research_critique"))],
                     llm: ChatOpenAI) -> dict:
    prompt = PromptTemplate(
        input_variables=["research", "critique"],
        template="Improve the following research based on the critique:\n\nResearch:\n{research}\n\nCritique:\n{critique}"
    )
    improved_research = llm.invoke(
        prompt.format(research=state["research_results"], critique=state["research_critique"])).content
    return {"improved_research": improved_research}


def write_content(state: Annotated[State, RunnableParallel(improved_research=itemgetter("improved_research"))],
                  writer_tool: WriterTool) -> dict:
    platforms = ["LinkedIn", "Medium", "Twitter"]
    draft_content = {platform: writer_tool.run(state["improved_research"], platform=platform) for platform in platforms}
    return {"draft_content": draft_content}


def critique_content(state: Annotated[State, RunnableParallel(draft_content=itemgetter("draft_content"))],
                     critique_tool: CritiqueTool) -> dict:
    content_critique = {platform: critique_tool.run(content, platform=platform)
                        for platform, content in state["draft_content"].items()}
    return {"content_critique": content_critique}


def improve_content(state: Annotated[State, RunnableParallel(draft_content=itemgetter("draft_content"),
                                                             content_critique=itemgetter("content_critique"))],
                    llm: ChatOpenAI) -> dict:
    improved_content = {}
    for platform in state["draft_content"].keys():
        prompt = PromptTemplate(
            input_variables=["content", "critique"],
            template=f"Improve the following {platform} content based on the critique:\n\nContent:\n{{content}}\n\nCritique:\n{{critique}}"
        )
        improved_content[platform] = llm.invoke(
            prompt.format(content=state["draft_content"][platform],
                          critique=state["content_critique"][platform])).content
    return {"improved_content": improved_content}


def publish_content(state: Annotated[State, RunnableParallel(improved_content=itemgetter("improved_content"))],
                    publish_tool: PublishTool) -> dict:
    publish_result = publish_tool.run(state["improved_content"])
    return {"publish_result": publish_result, "final_content": state["improved_content"]}


# Create the graph
workflow = StateGraph(State)

# Initialize tools
research_tool = ResearchTool()
writer_tool = WriterTool()
critique_tool = CritiqueTool()
publish_tool = PublishTool()

# Add nodes to the graph
workflow.add_node("perform_research", lambda state: perform_research(state, research_tool))
workflow.add_node("critique_research", lambda state: critique_research(state, critique_tool))
workflow.add_node("improve_research", lambda state: improve_research(state, llm))
workflow.add_node("write_content", lambda state: write_content(state, writer_tool))
workflow.add_node("critique_content", lambda state: critique_content(state, critique_tool))
workflow.add_node("improve_content", lambda state: improve_content(state, llm))
workflow.add_node("publish_content", lambda state: publish_content(state, publish_tool))

# Define edges
workflow.add_edge("perform_research", "critique_research")
workflow.add_edge("critique_research", "improve_research")
workflow.add_edge("improve_research", "write_content")
workflow.add_edge("write_content", "critique_content")
workflow.add_edge("critique_content", "improve_content")
workflow.add_edge("improve_content", "publish_content")
workflow.add_edge("publish_content", END)

# Set the entrypoint
workflow.set_entry_point("perform_research")

# Compile the graph
app = workflow.compile()


# Run the workflow
def run_workflow(topic: str) -> State:
    try:
        result = app.invoke({"topic": topic})
        logger.info(f"Workflow completed successfully for topic: {topic}")
        return result
    except Exception as e:
        logger.error(f"An error occurred during workflow execution: {str(e)}")
        raise


# Evaluation function
def evaluate_solution(topic: str, content: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    evaluation_results = {}
    for platform, platform_content in content.items():
        evaluation_prompt = PromptTemplate(
            input_variables=["topic", "content", "platform"],
            template="""Evaluate the following {platform} content on a scale of 1-10 for relevance, coherence, engagement, and platform-specific suitability, given the topic '{topic}'. 
            Provide only numerical scores for each category in the following format:
            Relevance: [score]
            Coherence: [score]
            Engagement: [score]
            Platform Suitability: [score]

            Content to evaluate:
            {content}"""
        )
        try:
            evaluation_result = llm.invoke(
                evaluation_prompt.format(topic=topic, content=platform_content, platform=platform))

            # Parse the evaluation result
            scores = {}
            for line in evaluation_result.content.split('\n'):
                parts = line.split(':')
                if len(parts) == 2:
                    key, value = parts
                    try:
                        scores[key.strip()] = float(value.strip())
                    except ValueError:
                        logger.warning(f"Could not parse score for {key}: {value}")

            if not scores:
                logger.warning(f"No valid scores were parsed from the evaluation result for {platform}.")

            evaluation_results[platform] = scores
        except Exception as e:
            logger.error(f"An error occurred during evaluation for {platform}: {str(e)}")
            evaluation_results[platform] = {}

    return evaluation_results


# Example usage
if __name__ == "__main__":
    topic = "The impact of artificial intelligence on job markets"
    try:
        result = run_workflow(topic)
        for platform, content in result['final_content'].items():
            print(f"\nFinal content for {platform}:")
            print(content)
        print(f"\nPublish result: {result['publish_result']}")

        # Run evaluation
        evaluation_scores = evaluate_solution(topic, result['final_content'])
        if evaluation_scores:
            print("\nEvaluation scores:")
            for platform, scores in evaluation_scores.items():
                print(f"\n{platform}:")
                for category, score in scores.items():
                    print(f"  {category}: {score}")
        else:
            print("Evaluation failed to produce any scores.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
