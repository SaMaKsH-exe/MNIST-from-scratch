from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from train import predict_digit_from_pixels

app = FastAPI(title="MNIST Predictor API")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=False,
	allow_methods=["*"],
	allow_headers=["*"],
)


class PredictRequest(BaseModel):
	pixels: list[int]


class PredictResponse(BaseModel):
	digit: int
	confidence: float


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
	if len(payload.pixels) != 784:
		raise HTTPException(status_code=400, detail="Expected 784 pixels")

	try:
		digit, confidence = predict_digit_from_pixels(payload.pixels)
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc

	return PredictResponse(digit=int(digit), confidence=float(confidence))


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)