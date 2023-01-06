
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from Service.ClassificationService import classify_image # to do

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
    body = classify_image(file)
    return {"filename": file.filename}




# CLASSIFICATION RESULTS - SCORES AND AVALIATION
@app.get("/return_avaliation", methods=["GET"])
def return_avaliation():
    body = avaliation_return() # get the classification results to return
    return body
