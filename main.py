from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import motor.motor_asyncio
from bson import ObjectId
from sklearn.feature_extraction.text import TfidfVectorizer

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB (usa Railway plugin para MongoDB y actualiza MONGO_URI si cambia)
MONGO_URI = "mongodb://mongo:YvjDmHBINTcvxYWvLCzHaNJGmeBTjZWc@mongodb.railway.internal:27017"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client.IA
coleccion = db.conversacions

# Modelo ligero
vectorizer = TfidfVectorizer()
preguntas = ["¿Cuál es tu nombre?", "¿Qué puedes hacer?", "¿Dónde estudias?"]
respuestas = ["Soy un bot.", "Puedo responder preguntas.", "Estudio en ESPOL."]
X = vectorizer.fit_transform(preguntas)

# Modelos
class Mensaje(BaseModel):
    rol: str
    contenido: str

@app.get("/")
def ping():
    return {"message": "Servidor ON 🚀"}

@app.get("/conversaciones")
async def obtener_conversaciones():
    conversaciones = []
    async for conv in coleccion.find({}, {"titulo": 1, "mensajes": 1}):
        conv["_id"] = str(conv["_id"])
        conversaciones.append(conv)
    return conversaciones

@app.post("/conversaciones/nuevo")
async def nueva_conversacion(primerMensaje: str = Body(..., embed=True)):
    nueva = {
        "titulo": primerMensaje[:30],
        "mensajes": [{"rol": "Estudiante", "contenido": primerMensaje}]
    }
    resultado = await coleccion.insert_one(nueva)
    nueva["_id"] = str(resultado.inserted_id)
    return {"conversacion": nueva}

@app.post("/conversaciones/{conv_id}/mensajes")
async def agregar_mensaje(conv_id: str, mensaje: Mensaje):
    res = await coleccion.update_one(
        {"_id": ObjectId(conv_id)},
        {"$push": {"mensajes": mensaje.dict()}}
    )
    if res.modified_count == 0:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    return {"message": "Mensaje guardado"}

@app.delete("/conversaciones/{conv_id}")
async def eliminar_conversacion(conv_id: str):
    res = await coleccion.delete_one({"_id": ObjectId(conv_id)})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    return {"message": "Conversación eliminada"}

@app.post("/buscar")
def buscar_similar(query: str = Body(..., embed=True)):
    query_vec = vectorizer.transform([query])
    similitudes = (X * query_vec.T).toarray().flatten()

    if max(similitudes) < 0.1:
        return {
            "respuesta": "No entiendo eso todavía 😅",
            "necesita_aprendizaje": True,
            "pregunta_original": query
        }

    idx = similitudes.argmax()
    return {
        "respuesta": respuestas[idx],
        "necesita_aprendizaje": False
    }
