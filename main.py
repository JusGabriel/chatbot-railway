from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

# MongoDB (ajusta MONGO_URI según tu configuración)
MONGO_URI = "mongodb://mongo:YvjDmHBINTcvxYWvLCzHaNJGmeBTjZWc@mongodb.railway.internal:27017"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client.IA
coleccion = db.conversacions

# Preguntas y respuestas ampliadas
preguntas = [
    "hola",
    "¿cómo estás?",
    "¿qué puedes hacer?",
    "adiós",
    "gracias",

    # Info prácticas pre-profesionales y convalidación
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

    # Respuestas basadas en el texto que enviaste
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

# Vectorizador TF-IDF y entrenamiento
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preguntas)

# Modelo para mensajes
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
