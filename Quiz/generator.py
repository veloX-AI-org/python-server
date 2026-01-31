import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-4-Scout-17B-16E-Instruct"
)

model = ChatHuggingFace(llm=llm)
print("Model Loaded")
print("-"*20)


class QuizStructure(BaseModel):
    question: str = Field(
        description=(
            "A clear, specific multiple-choice question derived strictly from the "
            "provided YouTube transcript. The question must test understanding of "
            "the transcript content and must not rely on external knowledge."
        )
    )

    options: List[str] = Field(
        description=(
            "A list of EXACTLY four answer options for the question. "
            "Each option must be a string formatted exactly as: "
            "'A. option text', 'B. option text', 'C. option text', 'D. option text'. "
            "Only ONE option must be correct according to the transcript."
        ),
        min_length=4,
        max_length=4
    )

    answer: int = Field(
        description=(
            "An integer representing the index of the correct answer option. "
            "The value must be between 0 and 3 inclusive, where: "
            "0 = option A, 1 = option B, 2 = option C, 3 = option D. "
            "This index must correctly correspond to the correct option."
        ),
        ge=0,
        le=3
    )

    explanation: str = Field(
        description=(
            "A concise explanation JUSTIFYING why the selected answer is correct. "
            "The explanation MUST be between 40 and 50 words in length, "
            "and it must reference only information explicitly stated in the transcript. "
            "Do not introduce new facts or external context."
        )
    )

parser = PydanticOutputParser(pydantic_object=QuizStructure)

template = """
You are an AI assistant whose task is to generate exactly ONE quiz question based strictly on a provided YouTube video transcript.

You will be given:
1. A video transcript (may be empty or missing)
2. format_instructions that define the exact structure and format of the output

------------------------------------
STRICT RULES (MUST FOLLOW):
------------------------------------

- Generate EXACTLY ONE quiz.
- Use ONLY information explicitly present in the transcript.
- Do NOT add external knowledge, assumptions, or inferred facts.
- Do NOT hallucinate questions, options, answers, or explanations.
- The final output MUST strictly follow the provided format_instructions.
- The output MUST be directly parseable by the PydanticOutputParser.
- Do NOT include markdown, comments, headings, or extra text outside the required format.
- Do NOT return a list or array of quizzes.

------------------------------------
MISSING OR INVALID TRANSCRIPT HANDLING:
------------------------------------

If the transcript is missing, empty, or invalid:
- Do NOT generate a quiz.
- Do NOT follow format_instructions.
- Respond ONLY with the following formal sentence:

"Unable to generate quiz questions as no valid transcript was provided for the video."

------------------------------------
QUIZ GENERATION REQUIREMENTS:
------------------------------------

If a valid transcript is provided:
- Generate exactly ONE quiz question.
- The question must assess understanding of the transcript.
- There must be exactly four answer options, with only one correct answer.
- The answer index must correctly match the correct option.
- The explanation must be between 40 and 50 words and must rely only on the transcript.
- Follow format_instructions exactly with no deviations.

------------------------------------
INPUTS:
------------------------------------

Transcript:
{transcript}

Format Instructions:
{format_instructions}

------------------------------------
DUPLICATE PREVENTION RULE (MANDATORY):
------------------------------------

Previously generated quiz questions are listed below.
You MUST NOT repeat, rephrase, or generate a semantically similar question.

If you cannot generate a new, distinct question based on the transcript,
respond with the formal missing-transcript response.

Previously Generated Questions:
{previous_questions}

"""

prompt = PromptTemplate(
    template=template,
    input_variables=["transcript", "previous_questions"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = prompt | model | parser

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
    
    return generated_quizzes