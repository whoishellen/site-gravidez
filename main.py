from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# 1) Libera todas as origens (para evitar erro CORS entre portas)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2) Carrega o modelo treinado
modelo = joblib.load("modelo.pkl")

# 3) Define como os dados chegam
class DadosClinicos(BaseModel):
    idade: float
    fc: float
    glicemia: float
    temperatura: float
    sistolica: float
    diastolica: float

# 4) Endpoint POST que retorna risco + mensagem
@app.post("/api/risco")
def classificar_risco(dados: DadosClinicos):
    entrada = np.array([[
        dados.idade,
        dados.sistolica,
        dados.diastolica,
        dados.glicemia,
        dados.temperatura,
        dados.fc
    ]])
    risco = modelo.predict(entrada)[0]
    mensagens = {
        "low risk":   "Parabéns! Sua gravidez apresenta baixo risco. Continue os acompanhamentos.",
        "mid risk":   "Atenção! Há risco moderado. Reforce cuidados e consulte seu médico.",
        "high risk":  "⚠️ Alto risco! Busque orientação médica imediatamente."
    }
    return {"risco": risco, "mensagem": mensagens.get(risco, "")}
