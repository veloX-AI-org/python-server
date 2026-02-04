from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

class QuizStructure(BaseModel):
    question: str = Field(
        description=(
            "A clear, specific multiple-choice question derived strictly from the provided YouTube transcript. The question must test understanding of the transcript content and must not rely on external knowledge."
        )
    )

    options: List[str] = Field(
        description=(
            "A list of EXACTLY four answer options for the question. Each option must be a string formatted exactly as: 'A. option text', 'B. option text', 'C. option text', 'D. option text'. Only ONE option must be correct according to the transcript."
        ),
        min_length=4,
        max_length=4
    )

    answer: int = Field(
        description=(
            "An integer representing the index of the correct answer option. The value must be between 0 and 3 inclusive, where: 0 = option A, 1 = option B, 2 = option C, 3 = option D."
        ),
        ge=0,
        le=3
    )

    explanation: str = Field(
        description=(
            "A concise explanation JUSTIFYING why the selected answer is correct (between 40 and 50 words)."
        )
    )

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