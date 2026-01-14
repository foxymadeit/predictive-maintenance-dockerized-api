from fastapi import (FastAPI, Depends,
                    HTTPException, Query,
                    Response, Request)
from pathlib import Path
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uuid
import time
import pandas as pd
import numpy as np
import io
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from api.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    ShapContributor, ExplainResponse
)
from api.deps import get_model, get_background_df
from src.predictive_model import PredictiveMaintenanceModel
from src.logging_config import setup_json_logger
from src.paths import PROJECT_ROOT
from src.config import load_config

cfg = load_config()
STATIC_DIR = PROJECT_ROOT / cfg["paths"]["static_dir"]





logger = setup_json_logger("api")

app = FastAPI(
    title="Predictive Maintenance API",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail=f"Missing {index_path}")
    return index_path.read_text(encoding="utf-8")

@app.get("/health")
def health(model: PredictiveMaintenanceModel = Depends(get_model)):
    return {
        "status": "ok",
        "threshold": model.threshold,
        "features": list(model.feature_cols),
    }


@app.post("/predict", response_model=PredictResponse)
def predict_one(
    request: Request,
    req: PredictRequest,
    model: PredictiveMaintenanceModel = Depends(get_model),
):
    """
    Performs a single equipment failure prediction.

    Takes sensor readings, transforms them through the model pipeline, 
    and returns the failure probability along with a binary alert flag.
    The prediction is logged with a unique request ID for audit purposes.

    Args:
        req: A PredictRequest containing feature values for a single timestamp.
        
    Returns:
        A PredictResponse object including 'proba_failure', 'alert', 
        and the 'threshold' used for decision making.
    """
    try:
        # transform request to dict with original column names
        record = req.model_dump(by_alias=True)
        out = model.predict(record)  # returns pd.DataFrame
        row = out.iloc[0].to_dict()

        proba = float(row["proba_failure"])
        alert = int(row["alert"])
        thr = float(row["threshold"])

        logger.info(
            "prediction made",
            extra={
                "extra": {
                    "request_id": request.state.request_id,
                    "model": model.artifacts_dir.name,
                    "proba_failure": proba,
                    "alert": alert,
                    "threshold": thr,
                }
            },
        )

        
        return PredictResponse(**row)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    req: BatchPredictRequest,
    model: PredictiveMaintenanceModel = Depends(get_model)
):
    """
    Performs batch predictions for multiple sensor records at once.

    Converts a list of records into a pandas DataFrame, runs a single vectorized inference through 
    the pipeline, and returns a list of results.

    Args:
        req: A BatchPredictRequest containing a list of records.

    Returns:
        A BatchPredictResponse containing a 'results' list, where each 
        element is a standard PredictResponse.
    """
    try:
        records = [r.model_dump(by_alias=True) for r in req.records]
        X = pd.DataFrame(records)
        out = model.predict(X)
        results = [PredictResponse(**row) for row in out.to_dict(orient="records")]
        return BatchPredictResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.post("/explain", response_model=ExplainResponse)
def explain_one(
    req: PredictRequest,
    model: PredictiveMaintenanceModel = Depends(get_model),
    top_k: int = Query(8, ge=1, le=30),
    X_background: pd.DataFrame = Depends(get_background_df),
):
    """
    Provides a detailed explanation for a single prediction using SHAP.

    Returns the top K features that most significantly increased or 
    decreased the failure risk for this specific observation.
    """
    try:
        record = req.model_dump(by_alias=True)
        X = pd.DataFrame([record])

        # Predict
        pred_row = model.predict(X).iloc[0]
        proba = float(pred_row["proba_failure"])
        alert = int(pred_row["alert"])
        thr = float(pred_row["threshold"])

        shap_values = model.explain(X, X_background=X_background, check_additivity=False)


        sv = shap_values[0].values
        fn = list(shap_values.feature_names)
        fv = shap_values[0].data

        order = np.argsort(np.abs(sv))[::-1][:top_k]

        top = []
        for i in order:
            val = None
            if fv is not None and fv[i] is not None and not (isinstance(fv[i], float) and np.isnan(fv[i])):
                # fv after preprocessing is numeric; keep float
                val = float(fv[i])

            s = float(sv[i])
            top.append(
                ShapContributor(
                    feature=fn[i],
                    value=val,
                    shap_value=s,
                    direction="increases_risk" if s > 0 else "decreases_risk",
                )
            )

        
        return ExplainResponse(
            proba_failure=proba,
            alert=alert,
            threshold=thr,
            top_contributors=top
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    


@app.post("/explain/plot", response_class=Response)
def explain_plot(
    req: PredictRequest,
    model: PredictiveMaintenanceModel = Depends(get_model),
    X_background: pd.DataFrame = Depends(get_background_df),
):
    """
    Generates a SHAP waterfall plot as a PNG image.

    This visual representation shows the contribution of each feature to 
    the final prediction. Best used for embedding in dashboards or reports.
    """
    try:
        record = req.model_dump(by_alias=True)
        X = pd.DataFrame([record])

        shap_values = model.explain(X, X_background=X_background, check_additivity=False)

        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        fig = plt.gcf()

        buf = io.BytesIO()
        plt.gcf().savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close("all")

        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    start = time.time()

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        raise e
    finally:
        latency_ms = (time.time() - start) * 1000

        logger.info(
            "request completed",
            extra={
                "extra": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": round(latency_ms, 2),
                }
            },
        )

    response.headers["X-Request-ID"] = request_id
    return response