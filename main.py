from fastapi import FastAPI, Body, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import motor.motor_asyncio
from bson import ObjectId
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajustar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conexión MongoDB Railway (ajusta tu URI)
MONGO_URI = "mongodb://mongo:BHFQycLysgYtindKTQJOWyFJUyTNLxiv@mongodb.railway.internal:27017"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client['IA']
coleccion = db.conversacions
pendientes = db.aprendizaje
coleccion_clientes = db.clientes

# Preguntas y respuestas para chatbot
preguntas = [
    "hola",
    "¿cómo estás?",
    "¿qué puedes hacer?",
    "adiós",
    "gracias",
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
    "¿Qué es la convalidación y cómo funciona?",
]

respuestas = [
    "Hola, ¿en qué puedo ayudarte?",
    "Estoy bien, gracias por preguntar.",
    "Puedo responder tus preguntas básicas.",
    "Adiós, que tengas un buen día.",
    "De nada, estoy aquí para ayudarte.",
    "Tecnología Superior: 240 horas de Prácticas Laborales y 96 de Servicio Comunitario. Modalidad dual: 2000 horas de Prácticas Laborales y 100 de Servicio Comunitario.",
    "Con tu tutor. Si no cuentas con tutor, puedes escribir a vinculacion.esfot@epn.edu.ec",
    "No, no se pueden registrar en un mismo formulario horas de PPP y servicio comunitario.",
    "6 meses desde el último día de prácticas o desde la fecha de emisión del certificado.",
    "Ya no se pueden registrar capacitaciones como prácticas de servicio comunitario, regirse a la tabla de convalidación.",
    "No, las actividades de cooperación o voluntariado deben estar relacionadas al perfil de egreso de la carrera.",
    "En entidades públicas o privadas que entreguen certificado indicando las horas y actividades realizadas conforme al perfil de egreso.",
    "No podrás realizar prácticas ni vinculación en emprendimientos u organizaciones de compañeros o familiares directos.",
    "Debes solicitar un tutor antes de empezar las prácticas, asegurarte que la empresa te haya aceptado y seguir el procedimiento de registro.",
    "Debes enviar un correo a la subdirección con documentos, incluyendo formularios y certificados para validar las horas de tus actividades extracurriculares.",
    "Debes entregar un certificado firmado y llenar formularios específicos, además de seguir los pasos de designación de tutor y seguimiento.",
    "La convalidación es un procedimiento para validar actividades extracurriculares que puedes registrar como prácticas, siguiendo un proceso administrativo específico.",
]

# TF-IDF vectorizador
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preguntas)

# Modelos Pydantic
class Mensaje(BaseModel):
    rol: str
    contenido: str

class NuevaConversacion(BaseModel):
    primerMensaje: str

# Obtener usuario desde token en header Authorization: Bearer <token>
async def obtener_usuario(authorization: Optional[str] = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token requerido")
    token = authorization.split(" ")[1]
    usuario = await coleccion_clientes.find_one({"token": token})
    if usuario is None:
        raise HTTPException(status_code=401, detail="Token inválido")
    return usuario

@app.get("/")
def ping():
    return {"message": "Servidor ON 🚀"}

# Listar conversaciones del usuario
@app.get("/conversaciones")
async def obtener_conversaciones(usuario=Depends(obtener_usuario)):
    conversaciones = []
    cursor = coleccion.find({"usuario_id": usuario["_id"]})
    async for conv in cursor:
        conv["_id"] = str(conv["_id"])
        conv["usuario_id"] = str(conv["usuario_id"])
        conversaciones.append(conv)
    return conversaciones

# Crear conversación para usuario
@app.post("/conversaciones/nuevo")
async def nueva_conversacion(nueva: NuevaConversacion, usuario=Depends(obtener_usuario)):
    doc = {
        "usuario_id": usuario["_id"],
        "titulo": nueva.primerMensaje[:30],
        "mensajes": [{"rol": "Estudiante", "contenido": nueva.primerMensaje}]
    }
    result = await coleccion.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    doc["usuario_id"] = str(doc["usuario_id"])
    return {"conversacion": doc}

# Agregar mensaje solo si conversación pertenece a usuario
@app.post("/conversaciones/{conv_id}/mensajes")
async def agregar_mensaje(conv_id: str, mensaje: Mensaje, usuario=Depends(obtener_usuario)):
    conversacion = await coleccion.find_one({"_id": ObjectId(conv_id)})
    if conversacion is None:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    if str(conversacion["usuario_id"]) != str(usuario["_id"]):
        raise HTTPException(status_code=403, detail="No autorizado")
    res = await coleccion.update_one(
        {"_id": ObjectId(conv_id)},
        {"$push": {"mensajes": mensaje.dict()}}
    )
    if res.modified_count == 0:
        raise HTTPException(status_code=500, detail="Error al guardar mensaje")
    return {"message": "Mensaje guardado"}

# Eliminar conversación solo si pertenece al usuario
@app.delete("/conversaciones/{conv_id}")
async def eliminar_conversacion(conv_id: str, usuario=Depends(obtener_usuario)):
    conversacion = await coleccion.find_one({"_id": ObjectId(conv_id)})
    if conversacion is None:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    if str(conversacion["usuario_id"]) != str(usuario["_id"]):
        raise HTTPException(status_code=403, detail="No autorizado")
    res = await coleccion.delete_one({"_id": ObjectId(conv_id)})
    if res.deleted_count == 0:
        raise HTTPException(status_code=500, detail="Error al eliminar conversación")
    return {"message": "Conversación eliminada"}

# Buscar respuesta con TF-IDF
@app.post("/buscar")
async def buscar_similar(query: str = Body(..., embed=True), historial: list[str] = Body(default=[])):
    contexto = " ".join(historial[-3:])
    texto_total = contexto + " " + query if contexto else query

    query_vec = vectorizer.transform([texto_total])
    similitudes = (X * query_vec.T).toarray().flatten()

    if np.max(similitudes) < 0.1:
        await pendientes.insert_one({"pregunta": query, "contexto": historial})
        return {
            "respuesta": "Lo siento, aún no tengo información sobre eso 😅",
            "necesita_aprendizaje": True,
            "pregunta_original": query
        }

    idx = np.argmax(similitudes)
    return {
        "respuesta": respuestas[idx],
        "necesita_aprendizaje": False
    }
