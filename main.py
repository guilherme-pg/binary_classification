
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from Service.ClassificationService import classify_image

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def read_items():
    html_content = """
        <h1>Maracatu Recognition</h1>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.route("/send_image", methods=["POST"])
def send_image():
    img = request.files['image']
    body = classify_image(img)
    return body


@app.route("/return_avaliation", methods=["GET"])
def return_avaliation():
    body = avaliation_return() # get the classification results to return
    return body
