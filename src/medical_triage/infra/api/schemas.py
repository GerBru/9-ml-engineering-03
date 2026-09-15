from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Payload de entrada para classificação de laudo médico."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"text": "patient with cardiac chest pain and heart failure"}
            ]
        }
    )

    text: str = Field(..., description="Texto do laudo médico para classificação")
    

class PredictResponse(BaseModel):
    """Resultado da classificação de triagem médica."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"label": "cardiovascular"}
            ]
        }
    )

    label: str = Field(..., description="Classe de triagem predita pelo modelo")