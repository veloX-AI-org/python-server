from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from schema.output.quizSchema import QuizStructure

load_dotenv()

model = ChatOpenAI(
    model='gpt-5-nano'
)

parser = PydanticOutputParser(pydantic_object=QuizStructure)

structuredOutput = model.with_structured_output(QuizStructure)

template = """
Generate EXACTLY ONE quiz question strictly from the provided transcript.

RULES:
- Use ONLY information explicitly in the transcript.
- Do NOT add external knowledge, assumptions, or inferences.
- Do NOT hallucinate any content.
- Output MUST strictly follow format_instructions and be parseable by PydanticOutputParser.
- Do NOT include markdown, comments, or extra text.
- Do NOT return multiple quizzes or lists.

MISSING / INVALID TRANSCRIPT:
If the transcript is missing, empty, or invalid, respond ONLY with:
"Unable to generate quiz questions as no valid transcript was provided for the video."

QUIZ REQUIREMENTS:
- Exactly ONE question assessing transcript understanding.
- Exactly four answer options with only one correct.
- Answer index must match the correct option.
- Explanation must be 40-50 words and based only on the transcript.

DUPLICATE PREVENTION:
Do NOT repeat, rephrase, or generate semantically similar questions from the list below.
If no distinct question can be generated, respond with the missing-transcript message.

Transcript:
{transcript}

Format Instructions:
{format_instructions}

Previously Generated Questions:
{previous_questions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["transcript", "previous_questions"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | structuredOutput

def generate_quiz(transcript):
    generated_quizzes = []
    previous_questions = []

    MAX_QUIZZES = 5
    MAX_RETRIES = 3

    for i in range(MAX_QUIZZES):
        for attempt in range(MAX_RETRIES):
            try:
                result = chain.invoke({
                    "transcript": transcript,
                    "previous_questions": previous_questions or "None"
                })

                generated_quizzes.append(result.dict())
                previous_questions.append(result.question)

                break 

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to generate unique quiz {i+1}: {e}")
                continue
    print(f"{i+1} Quiz generated")
    
    return generated_quizzes