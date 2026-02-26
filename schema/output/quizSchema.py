from pydantic import BaseModel, Field
from typing import List

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