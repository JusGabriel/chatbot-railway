from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import motor.motor_asyncio
from bson import ObjectId
from typing import Optional
import os

# Modelo simple de embeddings
class SimpleModel:
    def encode(self, texts):
        if isinstance(texts, str):
            return [self._text_to_vector(texts)]
        return [self._text_to_vector(t) for t in texts]
    
    def _text_to_vector(self, text):
        text = text.lower()
        vec = [0] * 26
        for ch in text:
            idx = ord(ch) - ord('a')
            if 0 <= idx < 26:
                vec[idx] += 1
        total = sum(vec)
        return [v / total for v in vec] if total > 0 else vec

def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

# App y CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# MongoDB (ajustado para Railway interno)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:YvjDmHBINTcvxYWvLCzHaNJGmeBTjZWc@mongodb.railway.internal:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client.IA
coleccion = db.conversacions

# Embedding + memoria
model = SimpleModel()
preguntas = [
    "hola", "¿cómo estás?", "¿qué puedes hacer?", "adiós", "gracias",
    "¿Cuántas horas de PPP y Servicio comunitario se necesitan?",
    "¿Con quién debo comunicarme en caso de dudas?",
    "¿Se puede registrar en un mismo formulario horas de PPP y servicio comunitario?",
    "¿Qué tiempo de validez tienen los certificados y las prácticas realizadas?",
    "¿Se puede registrar capacitaciones como prácticas de servicio comunitario?",
    "¿Las actividades de cooperación o voluntariado pueden ser diferentes a las del perfil de egreso de la carrera?",
    "¿En dónde puedo hacer vinculación?",
    "¿Hay alguna restricción de donde no puedo hacer prácticas o vinculación?",
    "¿Qué debo hacer para empezar mis prácticas pre-profesionales?",
    "¿Qué debo hacer para registrar la convalidación de mis actividades extracurriculares?",
    "¿Cómo se realiza el procedimiento de registro de prácticas pre-profesionales?",
    "¿Qué es la convalidación y cómo funciona?"
]
respuestas = [
    "Hola, ¿en qué puedo ayudarte?", "Estoy bien, gracias.", "Puedo responder tus preguntas.", "Adiós, cuídate.", "¡De nada!",
    "Tecnología Superior: 240 horas de Prácticas Laborales y 96 de Servicio Comunitario...",
    "Con tu tutor o escribe a vinculacion.esfot@epn.edu.ec",
    "No se pueden registrar ambas en un mismo formulario.",
    "6 meses desde la fecha de emisión.",
    "No, ya no se pueden registrar capacitaciones como servicio comunitario.",
    "Deben estar relacionadas al perfil de egreso.",
    "En entidades públicas o privadas que emitan certificado.",
    "No puedes en negocios propios ni de familiares.",
    "Debes tener un tutor y seguir el proceso.",
    "Debes enviar documentos a la subdirección.",
    "Certificado firmado, formularios y seguimiento con tutor.",
    "Validación de actividades extracurriculares con proceso administrativo."
]

qa_memory = []
for i, emb in enumerate(model.encode(preguntas)):
    qa_memory.append({
        "id": str(uuid.uuid4()),
        "pregunta": preguntas[i],
        "respuesta": respuestas[i],
        "embedding": emb,
        "metadata": {}
    })

# Modelos Pydantic
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
    query_embedding = model.encode(query)[0]
    mejores = [(cosine_similarity(query_embedding, item["embedding"]), item) for item in qa_memory]
    mejores.sort(reverse=True, key=lambda x: x[0])
    mejor_sim, mejor_item = mejores[0]
    if mejor_sim < 0.5:
        return {
            "respuesta": "Eso no lo entiendo aún 😅. ¿Qué debería responder a eso?",
            "necesita_aprendizaje": True,
            "pregunta_original": query
        }
    return {
        "respuesta": mejor_item["respuesta"],
        "necesita_aprendizaje": False
    }

@app.post("/corregir")
async def corregir_respuesta(
    pregunta: str = Body(..., embed=True),
    respuesta_correcta: str = Body(..., embed=True),
    metadata: Optional[dict] = Body(None, embed=True)
):
    if not respuesta_correcta or len(respuesta_correcta.strip().split()) < 5:
        raise HTTPException(status_code=400, detail="La respuesta correcta debe tener al menos 5 palabras.")
    embedding = model.encode(pregunta)[0]
    qa_memory.append({
        "id": str(uuid.uuid4()),
        "pregunta": pregunta,
        "respuesta": respuesta_correcta,
        "embedding": embedding,
        "metadata": metadata or {}
    })
    await coleccion.insert_one({
        "titulo": pregunta[:30],
        "mensajes": [
            {"rol": "Estudiante", "contenido": pregunta},
            {"rol": "IA", "contenido": respuesta_correcta}
        ]
    })
    return {"message": "Respuesta añadida correctamente"}

@app.patch("/conversaciones/{conv_id}")
async def actualizar_titulo(conv_id: str, nuevoTitulo: str = Body(..., embed=True)):
    res = await coleccion.update_one(
        {"_id": ObjectId(conv_id)},
        {"$set": {"titulo": nuevoTitulo}}
    )
    if res.modified_count == 0:
        raise HTTPException(status_code=404, detail="No se pudo actualizar el título")
    return {"message": "Título actualizado"}

# Solo para desarrollo local
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
