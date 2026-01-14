from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Union


class PredictRequest(BaseModel):
    Air_temperature_K: float = Field(
        ..., 
        alias="Air temperature [K]", 
        gt=0,
        description="Air temperature in Kelvin. Must be a positive value.",
        json_schema_extra={"example": 300.0}
    )
    Process_temperature_K: float = Field(
        ..., 
        alias="Process temperature [K]", 
        gt=0,
        description="Process temperature in Kelvin. Must be a positive value.",
        json_schema_extra={"example": 310.0}
    )
    Rotational_speed_rpm: float = Field(
        ..., 
        alias="Rotational speed [rpm]", 
        ge=0,
        description="Rotational speed of the tool in RPM.",
        json_schema_extra={"example": 1500}
    )
    Torque_Nm: float = Field(
        ..., 
        alias="Torque [Nm]", 
        ge=0,
        description="Torque applied to the tool in Nm.",
        json_schema_extra={"example": 40.0}
    )
    Tool_wear_min: float = Field(
        ..., 
        alias="Tool wear [min]", 
        ge=0,
        description="Tool wear in minutes.",
        json_schema_extra={"example": 100}
    )
    Type: Literal["L", "M", "H"] = Field(
        ..., 
        description="Quality class of the machine (Low, Medium, High)."
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "Air temperature [K]": 300.8,
                "Process temperature [K]": 310.3,
                "Rotational speed [rpm]": 1538,
                "Torque [Nm]": 41.2,
                "Tool wear [min]": 118,
                "Type": "L"
            }
        }
    }


class PredictResponse(BaseModel):
    proba_failure: float = Field(..., description="Failure probability (0.0 to 1.0)")
    alert: int = Field(..., description="1 if failure is likely, 0 otherwise")
    threshold: float = Field(..., description="The decision threshold used by the model")


class BatchPredictRequest(BaseModel):
    records: List[PredictRequest] = Field(..., min_length=1)


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


class ShapContributor(BaseModel):
    feature: str = Field(..., description="Feature name")
    value: Optional[Union[float, str]] = Field(None, description="Actual feature value")
    shap_value: float = Field(..., description="SHAP contribution (impact) on the prediction")
    direction: Literal["increases_risk", "decreases_risk"]

class ExplainResponse(BaseModel):
    proba_failure: float
    alert: int
    threshold: float
    top_contributors: List[ShapContributor]
