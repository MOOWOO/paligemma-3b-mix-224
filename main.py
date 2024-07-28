from fastapi import FastAPI, File, UploadFile
from paligemma import PaliGemma

app = FastAPI()
model = PaliGemma()

@app.post("/generate")
async def generate(task: str, image: UploadFile = File(...)):
    contents = await image.read()
    with open("input.jpg", "wb") as f:
        f.write(contents)
    output = model.run(task, "input.jpg")
    return {"output": output}