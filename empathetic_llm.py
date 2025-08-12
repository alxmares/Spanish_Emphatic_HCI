from typing import Optional, Literal, List
from pathlib import Path
import json
import requests
import chromadb
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.retrievers import VectorIndexRetriever


# ===== Esquema de respuesta validada =====
class Registro(BaseModel):
    nombre: Optional[str]
    edad: Optional[int]
    ubicacion: Optional[str]
    sexo: Optional[str]
    ideacion: Optional[bool]
    plan: Optional[bool]
    metodo: Optional[bool]


class RespuestaTexto(BaseModel):
    text: str                         # respuesta en español
    emocion: Literal["JOY", "NEUTRAL_NORMAL",
                     "ANGER", "DISGUST", "FEAR",
                     "SADNESS", "SURPRISE"]          # cómo responde el LLM
    kind: Literal["AFFIRMATIVE", "EXCLAMATORY",
                  "PARAGRAPH", "INTERROGATIVE"]       # estilo de la frase


class Protocolo(BaseModel):
    paso_actual: str
    pasos_completados: List[str]
    pasos_faltantes: List[str]


class SalidaEstructurada(BaseModel):
    registro: Registro
    respuesta: RespuestaTexto
    protocolo: Protocolo


# ===== Objeto general del asistente empático =====
class EmpatheticLLM:
    def __init__(self,
                 chroma_path: str,
                 collection_name: str,
                 modelo_llm: str = "qwen3:4b",
                 url_ollama: str = "http://localhost:11434",
                 embed_model: str = "qwen3:4b"):
        
        self.modelo = modelo_llm
        self.url = f"{url_ollama}/api/chat"
        self.historial_conversacion = []
        self.inicializado = False
        # --------------- Embedding model
        self.embedding_model = OllamaEmbedding(model_name=embed_model, base_url=url_ollama)
        self.index = self._load_index(chroma_path, collection_name)
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=2,
        )
        
        self.mensaje_inicial = "Hola, gracias por comunicarse con Soporte técnico. Soy un H C i empático. ¿Con quién tengo el gusto?"
    
    def _load_index(self, path: str, collection_name: str):
        db = chromadb.PersistentClient(path=path)
        chroma_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=self.embedding_model
        )

    def reset(self):
        self.historial_conversacion.clear()
        self.inicializado = False

    def get_prompt(self,
            protocol_flag: bool,       
            mensaje_usuario: str,
            emocion: str,
            certeza: float,
            registro: dict,
            protocolo: dict,
            mensajes: list):
        

        # --------------- Prompt
        if protocol_flag:
            # -------------- Retriever
            context_nodes = self.retriever.retrieve(protocolo["paso_actual"])
            context_text = "\n".join([n.text for n in context_nodes])

            prompt = f"""
                Responde solo en español. No traduzcas ni uses inglés.

                Eres un asistente virtual para la prevención del suicidio. NO derives al usuario: tú eres la ayuda. Sé empático, directo y formal con tono ligeramente amigable.

                Sigue este protocolo ESTRICTAMENTE y paso a paso, sin saltarte ninguno:
                1. saludo: En el saludo siempre preguntarás el nombre de la persona. Ejem: ¡Hola! Soy un asistente virtual de apoyo emocional. ¿Con quién tengo el gusto? No pases del saludo hasta que te diga su nombre.
                2. datos_generales: Luego del saludo, no importa lo que te diga. Siempre pregunta los datos personales, pues es lo más importante. Sé empático con tu respuesta solo si es necesario, pero requiero conocer los datos generales. Es muy importante que te des cuenta que antes de saber qué siente y por qué es necesario conocer su edad, ubicación y sexo. No pases hasta conseguir eso. 
                3. evaluacion_inicial: Evalua su sitación actual. La primera pregunta debería de ser ¿Cuál es el motivo de su llamada? ¿Cómo y qué siente? ¿Qué ocurrió para que se sintiera así? ¿Quiere realizar algo en concreto? 
                4. evaluacion_riesgo: Evalua más a fondo. ¿Tiene ideación, plan y/o método de querer hacer algo arriesgado con su vida? ¿Hay forma de hacer que se sienta mejor?
                5. intervencion
                6. escalamiento
                7. cierre

                Posteriormente, Siempre responderás con un JSON que contiene tres claves:
                1. "registro": un diccionario que acumula la información clave del usuario (nombre, edad, sexo, ubicación, emociones, riesgo).
                2. "respuesta": la respuesta que le darás al usuario en esta etapa del protocolo.
                3. "protocolo": un diccionario con los pasos del protocolo. Sigue estrictamente este flujo:
                    saludo → datos_generales (nombre, edad, ubicación y sexo) → evaluacion_inicial → evaluacion_riesgo → intervencion → escalamiento (si aplica) → cierre.

                ---
                EJEMPLOS DE CÓMO RESPONDER: 
                {context_text}
                
                ---
                ESTADO DEL PROTOCOLO:
                {json.dumps(protocolo, ensure_ascii=False)}

                ESTADO DEL REGISTRO:
                {json.dumps(registro, ensure_ascii=False)}

                Responde únicamente con un objeto JSON válido compatible con este esquema:
                En cada respuesta debes indicar dentro de "respuesta":
                - "emocion": JOY o NEUTRAL_NORMAL (son las únicas permitidas para tu tono).
                - "kind": AFFIRMATIVE, EXCLAMATORY, PARAGRAPH o INTERROGATIVE.

                {SalidaEstructurada.schema_json(indent=2)}

                NO uses Markdown. NO expliques. NO pienses en voz alta. Solo devuelve un JSON válido.
                """
        else:
            #
            prompt = (
                "Responde SOLO en español, de forma breve, alegre y empática.\n"
                f"El usuario parece sentirse {emocion.lower()} con certeza {round(certeza,2)}%.\n"
                "Al elegir tu emoción de respuesta **prefiere**: JOY > NEUTRAL_NORMAL > (ANGER, DISGUST, FEAR, SADNESS, SURPRISE).\n"
                "Usa emociones negativas solo si el usuario las pide explícitamente.\n"
                "Elige también el 'kind' que mejor describa tu frase: "
                "AFFIRMATIVE, EXCLAMATORY, PARAGRAPH o INTERROGATIVE.\n\n"
                "Devuelve EXCLUSIVAMENTE un JSON con esta forma exacta (sin Markdown):\n"
                "{{\n"
                '  "respuesta": {{\n'
                '    "text": "<tu respuesta en español>",\n'
                '    "emocion": "<JOY|NEUTRAL_NORMAL|ANGER|DISGUST|FEAR|SADNESS|SURPRISE>",\n'
                '    "kind": "<AFFIRMATIVE|EXCLAMATORY|PARAGRAPH|INTERROGATIVE>"\n'
                "  }}\n"
                "}}\n"
                "No añadas ninguna clave extra ni explicación."
            )

        
        # -------------- 3) Actualizar historial ----------
        #   ── posición 0 siempre es el "system" prompt dinámico ──
        if not mensajes:
            if protocol_flag:
                self.mensaje_inicial = "Hola, estás hablando con una H C I empático. ¿Con quién tengo el gusto?"
            else: 
                self.mensaje_inicial = "Hola, estás hablando con una H C I empático. ¿Cómo te encuentras hoy?"
            # Inserta el prompt como system
            mensajes.append({"role": "system", "content": prompt})
            # Inserta el mensaje inicial del asistente
            mensajes.append({"role": "assistant", "content": self.mensaje_inicial})

        else:
            mensajes[0]["content"] = prompt

        # -------------- 4) Añadir mensaje del usuario ----
        mensajes.append({"role": "user", "content": mensaje_usuario})

        #print("Historial: \n", mensajes)

        return mensajes
    
    def responder(self,
        protocol_flag: bool,
        mensaje_usuario: str,
        emocion: str,
        certeza: float,
        registro: dict,
        protocolo: dict,
        mensajes: list):

        yield ("⏳ Retrieving step...", mensajes)
        mensajes = self.get_prompt(protocol_flag,
                                    mensaje_usuario,
                                    emocion,
                                    certeza,
                                    registro,
                                    protocolo,
                                    mensajes,)
        
        # -------------- 5) Enviar a Ollama ---------------
        yield ("⏳ Thinking...", mensajes)
        payload = {
            "model": self.modelo,
            "messages": mensajes,
            "stream": False,
            "format": "json",
            "temperature": 0.7,
            "max_tokens": 500,
        }

        resp = requests.post(self.url, json=payload, timeout=120)
        resp.raise_for_status()
        respuesta_json = resp.json()["message"]["content"]

        # -------------- 6) Guardar respuesta en historial
        mensajes.append({"role": "assistant", "content": respuesta_json})

        yield (respuesta_json, mensajes)
        #return respuesta_json, mensajes