from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from diagnosis_pred import single_description, df, model, embedding_matrix

app = FastAPI()

# Static and templates setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search_icd")
def search(description: str):
    result = single_description(description, df, model, embedding_matrix)
    return JSONResponse(content=result)
