from .basic import router as basic_router
from .cv import router as cv_router
from .vacancy import router as vacancy_router
from .question import router as question_router

__all__ = ["basic_router", "cv_router", "vacancy_router", "question_router"]