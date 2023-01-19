
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from Service.ClassificationService import classify_image

app = FastAPI()


# MAIN PAGE - description
@app.get("/", response_class=HTMLResponse)
async def read_items():
    html_content = """
        <h1>Maracatu Recognition  --  about</h1>
    """
    return HTMLResponse(content=html_content, status_code=200)


# UPLOAD IMAGE
@app.post("/upload_image")
async def upload_image(file: UploadFile):
    scores = classify_image(file)
    return scores
