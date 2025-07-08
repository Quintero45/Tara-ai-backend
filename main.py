from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Inicializa la app
app = FastAPI()

# Carga el modelo (usando CPU)
generator = pipeline("text2text-generation", model="mrm8488/t5-small-finetuned-quora-for-paraphrasing")

# Modelo para recibir datos por POST
class PromptRequest(BaseModel):
    prompt: str

# Ruta raíz opcional
@app.get("/")
def read_root():
    return {"message": "Bienvenido al backend de TARA"}

# Ruta para generar texto
@app.post("/generate")
async def generate_text(req: PromptRequest):
    result = generator(req.prompt, max_new_tokens=100)
    return {"response": result[0]["generated_text"]}
