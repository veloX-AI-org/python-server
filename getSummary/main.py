from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# Load Model
model = ChatOpenAI(
    model='gpt-5-nano'
)

# Create Output Schema
class output_structure(BaseModel):
    summary: str = Field(
        description=(
            "A clear, well-structured summary of the provided context (of might be URL or DOCUMENT content), written in approximately 100-150 words and MUST be a string."
        )
    )

    questions: List[str] = Field(
        description=(
            "A list of three thoughtful follow-up questions derived from the summary and the original context."
        )
    )

parser = PydanticOutputParser(pydantic_object=output_structure)

structuredOutputModel = model.with_structured_output(output_structure)

template = """
You are a helpful assistant that analyzes the given context, produces a concise, accurate summary, and generates three relevant follow-up questions.
Sometimes to won't context properly or it may be empty. In such case, simply response with an little summary and explan why can cannot able to create and and empty simple question.

Format Instructions:
{format_instructions}

The following are some chucks of original document use them to summarize:
{context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=['context'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | structuredOutputModel

# Function to get structured reponse
def getResponse(context):
    return chain.invoke({
        'context': context
    })