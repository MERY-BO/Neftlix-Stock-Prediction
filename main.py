from fastapi import FastAPI
from fastapi.responses import JSONResponse
from ntflx_model import predict_stock  

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Netflix Stock Prediction API"}

@app.get("/predict/")
def predict_stock_endpoint(ticker: str = "NFLX"):
    try:
        # Call the prediction workflow
        predictions, dates = predict_stock(ticker)

        # Format response
        results = [{"date": str(dates[i]), "predicted_close": float(predictions[i])} for i in range(len(predictions))]
        return JSONResponse(content={"ticker": ticker, "predictions": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
