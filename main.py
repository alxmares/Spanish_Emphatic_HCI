from faster_whisper import WhisperModel
from ser_model import SERModel
from emotional_tts import EmotionalTTS
from empathetic_llm import EmpatheticLLM
import gradio as gr
import numpy as np
from pathlib import Path
import pandas as pd
import time


# Paths relativos a la raíz del proyecto (dentro del contenedor)
BASE_DIR = Path(__file__).resolve().parent

# ----------------- EMOTIONS & KIDNS
EMOTINS_BASE = ['ANGER', 'DISGUST', 'FEAR', 'JOY', 'NEUTRAL_NORMAL', 'SURPRISE', 'SADNESS']
EMOTIONS_PROTOCOL = ["NEUTRAL_NROMAL", "JOY"]
KINDS = ["AFFIRMATIVE", "EXCLAMATORY", "PARAGRAPH", "INTERROGATIVE"]
DF_SAMPLES = pd.read_csv(BASE_DIR / "models" / "elra-f" / "elra_f_examples.csv")



# ---------------------------------
# ------- F U N C I O N E S -------
# ---------------------------------

# --- Funciones de procesamiento ---
def detectar_emocion(audio_path):
    resultado = ser.predict(audio_path)
    #print(resultado)

    perfil = resultado["tarea_1"]["clase"]
    prob_perfil = resultado["tarea_1"]["probabilidades"].get(perfil, 0.0)

    region = resultado["tarea_2"]["clase"]
    prob_region = resultado["tarea_2"]["probabilidades"].get(region, 0.0)

    emocion = resultado["tarea_3"]["clase"]
    prob_emocion = resultado["tarea_3"]["probabilidades"].get(emocion, 0.0)

    return perfil, prob_perfil, region, prob_region, emocion, prob_emocion

def get_emotion(audio_path, umbral=0.0):
    resultado = ser.predict(audio_path)

    # EMOCIONES
    emociones = ['Anger','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']
    probs_emo = [round(resultado["tarea_3"]["probabilidades"].get(e, 0.0) * 100, 1) for e in emociones]
    idx_dom   = probs_emo.index(max(probs_emo))
    emocion_dom = emociones[idx_dom]
    certeza_dom = round(probs_emo[idx_dom] / 100, 2)

    df_emo = pd.DataFrame({
        "Etiqueta": emociones,
        "Porcentaje": probs_emo,
        "Grupo": ["Emoción"] * len(emociones)
    })

    # PERFIL
    perfiles = ['Child', 'Female', 'Male']
    probs_perf = [round(resultado["tarea_1"]["probabilidades"].get(e, 0.0) * 100, 1) for e in perfiles]
    df_perf = pd.DataFrame({
        "Etiqueta": perfiles,
        "Porcentaje": probs_perf,
        "Grupo": ["Perfil"] * len(perfiles)
    })

    # REGIÓN
    regiones = ['Spain', 'Mex']
    probs_reg = [round(resultado["tarea_2"]["probabilidades"].get(e, 0.0) * 100, 1) for e in regiones]
    df_reg = pd.DataFrame({
        "Etiqueta": regiones,
        "Porcentaje": probs_reg,
        "Grupo": ["Región"] * len(regiones)
    })

    return (
        df_emo, df_perf, df_reg
    )

def generar_audio_tts(texto, emotion, kind, ref_audio_path=None):

    options = {
        "temperature": 0.75,
        "length_penalty": 1,
        "repetition_penalty": 5.0,
        "top_k": 50,
        "top_p": 0.85,
        "speed": 1.0,
    }

    ruta = tts.synthesize(
        text=texto,
        emotion=emotion.upper(),
        kind=kind.upper(),
        output_path="output.wav",
        ref_audio_path=ref_audio_path,
        **options
    )
    return ruta

def get_tts(text, emotion, kind):
    if not text.strip():
        return None
    
    tts.emotions = EMOTINS_BASE
    tts.kinds = KINDS

    # ------------- GET SAMPLE
    if emotion not in tts.emotions: emotion = "NEUTRAL_NORMAL"
    if kind not in tts.kinds: kind = "AFFIRMATIVE"

    samples = DF_SAMPLES[(DF_SAMPLES["emotion"] == emotion) & (DF_SAMPLES["kind"] == kind)]
    
    audio_name = samples.sample(1)["name"].iloc[0]
    ref_audio_path = BASE_DIR / "models" / "elra-f" / "examples" / f"{audio_name}.wav"
    #print(ref_audio_path)

    # ------------- GENERATE AUDIO
    ruta_audio = generar_audio_tts(text, emotion, kind, ref_audio_path=ref_audio_path)

    return ruta_audio

def procesar_transcripcion_y_emocion(audio_path):
    if audio_path is None:
        return "", "", "", "", "", "✅ Esperando voz", ["etapa_1"]
    segments, info = modelo_whisper.transcribe(audio_path, language="es")
    texto = " ".join([seg.text for seg in segments])

    (df_emo, df_perf, df_reg) = get_emotion(audio_path)
    
    df_sorted = df_emo.sort_values(by="Porcentaje", ascending=False).reset_index(drop=True)
    emocion_detectada = df_sorted.at[0, "Etiqueta"]
    certeza_emocion = df_sorted.at[0, "Porcentaje"]

    return (
            texto,
            emocion_detectada, certeza_emocion,
            df_emo,
            df_perf,
            df_reg,
            "🟢 Voz procesada", ["etapa_1"]
        )

def generar_respuesta_llm(texto_usuario, emocion, certeza, historial):
    if not texto_usuario:
        return historial, "", "✅ Esperando solicitud", [], "", generar_panel_estado()

    registro_global["estado_emocional"] = emocion.upper()

    salida = asistente.responder(
        mensaje_usuario=texto_usuario,
        emocion=emocion,
        certeza=float(certeza),
        registro=registro_global,
        protocolo=protocolo_global,
    )

    # ───────── MODO CHAT NORMAL ─────────
    if isinstance(salida, dict) and salida.get("modo") == "chat":
        respuesta_texto = salida["text"]
        historial += [["🧑 Usuario", texto_usuario], ["🤖 Asistente", respuesta_texto]]
        return (
            historial,
            respuesta_texto,
            "💬 Respuesta generada (chat normal)",
            salida["emocion"],
            salida["kind"],
            generar_panel_estado(),
        )

    # ───────── MODO PROTOCOLO ─────────
    if salida is None:
        return historial, "⚠️ Error del LLM", "❌ Falló la generación", [], emocion, generar_panel_estado()

    protocolo_global.update(salida.protocolo.model_dump())
    registro_global.update(salida.registro.model_dump())
    respuesta_texto = salida.respuesta.text
    emo = salida.respuesta.emocion
    kind = salida.respuesta.kind
    historial += [["🧑 Usuario", texto_usuario], ["🤖 Asistente", respuesta_texto]]

    return (
        historial,
        respuesta_texto,
        "🧠 Respuesta de LLM en texto generada",
        ["etapa_2"],
        emo,
        kind,
        generar_panel_estado(),
    )

def sintetizar_respuesta_tts(respuesta_texto, emotion, kind):

    # ------------- GET SAMPLE
    if emotion not in tts.emotions: emotion = "NEUTRAL_NORMAL"
    if kind not in tts.kinds: kind = "AFFIRMATIVE"

    samples = DF_SAMPLES[(DF_SAMPLES["emotion"] == emotion) & (DF_SAMPLES["kind"] == kind)]
    
    audio_name = samples.sample(1)["name"].iloc[0]
    ref_audio_path = BASE_DIR / "models" / "elra-f" / "examples" / f"{audio_name}.wav"
    #print(ref_audio_path)

    # ------------- GENERATE AUDIO
    ruta_audio = generar_audio_tts(respuesta_texto, emotion, kind, ref_audio_path=ref_audio_path)

    return ruta_audio, "🟢 TTS generado", ["etapa_3"]

def actualizar_flujo_html(pasos_activos, emocion=None, duracion=None):
    def div(id, texto, subtitulo, paso_index):
        if id in pasos_activos:
            clase = "flujo-etapa flujo-verde"
            icono = "✅"
            info_extra = ""
            if paso_index == 0 and emocion:  # Paso 1 muestra emoción
                info_extra = (
                    f"<div class='info-extra'>"
                    f"<span class='tiempo'>⏱️ {duracion or '0.8s'}</span> "
                    f"<span class='emocion'>🙂 Emoción: <b>{emocion}</b></span>"
                    f"</div>"
                )
            else:
                info_extra = ""
        elif len(pasos_activos) > 0 and id == pasos_activos[-1]:
            clase = "flujo-etapa flujo-actual"
            icono = "🔵"
            info_extra = ""
        else:
            clase = "flujo-etapa flujo-pendiente"
            icono = "⚪"
            info_extra = ""

        return f"""
        <div class='{clase}' id='{id}'>
            <strong>{icono} {texto}</strong><br>
            <span class='subtitulo'>{subtitulo}</span>
            {info_extra}
        </div>"""

    html = (
        div("etapa_1", "1. Reconocimiento emocional y transcripción", "🔍 SER multitask + Whisper", 0) +
        div("etapa_2", "2. LLM", "📚 Retriever → 💭 Think → 💬 Response", 1) +
        div("etapa_3", "3. Síntesis de voz", "🗣️ TTS emocional", 2)
    )
    return html

def flujo_inicial_html():
    return actualizar_flujo_html([])

def generar_panel_estado():
    pasos_c = protocolo_global.get("pasos_completados", [])
    pasos_f = protocolo_global.get("pasos_faltantes", [])
    paso_actual = protocolo_global.get("paso_actual", "desconocido")
    emocion = registro_global.get("estado_emocional", "N/A")
    ideacion = registro_global.get("ideacion", "N/A")
    plan = registro_global.get("plan", "N/A")
    metodo = registro_global.get("metodo", "N/A")

    return f"""📋 Protocolo y Estado del Usuario
                ▸ Paso actual:         {paso_actual}
                ▸ Pasos completados:   {pasos_c}
                ▸ Pasos pendientes:    {pasos_f}
                ▸ Estado emocional:    {emocion}
                ▸ Ideación suicida:    {ideacion}
                ▸ Plan:                {plan}
                ▸ Método:              {metodo}"""

def cambiar_modo(modo_elegido):
    asistente.set_mode(modo_elegido)

    registro_global.update({k: None for k in registro_global})
    protocolo_global.update({
        "paso_actual": "saludo",
        "pasos_completados": [],
        "pasos_faltantes": [
            "saludo", "datos_generales", "evaluacion_inicial",
            "evaluacion_riesgo", "intervencion", "escalamiento", "cierre"
        ]
    })

    nuevo_chat = [["🤖 Asistente", "Hola, soy tu asistente empático. ¿Con quién tengo el gusto?"]]
    flujo_html = flujo_inicial_html()
    panel = generar_panel_estado()

    texto_modo = f"🛠️ Modo actual: {modo_elegido}"

    return nuevo_chat, flujo_html, "", [], panel, texto_modo

# ---------------------------------
# ---------------------------------


# -----------------------------
# ------- M Ó D U L O S -------
# -----------------------------

# ----------------------- MODELO WSHIPER
# Precarga el modelo 'large-v2' en GPU con precisión float16
modelo_whisper = WhisperModel("large-v2", device="cuda", compute_type="float16")

# ----------------------- MODELO SER
ser = SERModel(
    model_path = BASE_DIR / "models" / "multioutput_best_model.pkl",
    scaler_path= BASE_DIR / "models" / "scaler_multioutput_best_model.pkl"
)


# ----------------------- MODELO TTS
tts = EmotionalTTS(
    model_version="elra-f",
    model_base_path= BASE_DIR / "models",
    emotion_csv= "elra_f_examples.csv",
    emotions = EMOTIONS_PROTOCOL,
    kinds = KINDS,
)

# ----------------------- MODELO LLM
asistente = EmpatheticLLM(
    chroma_path     = str(BASE_DIR / "chroma_db"),
    collection_name = "ManualV2",
    modelo_llm      = "qwen3:4b"
)

# ---------------------------------
# ---------------------------------


# -----------------------------------------------
# ------- I N T E R F A Z  -  G R A D I O -------
# -----------------------------------------------

# Placeholder del historial
historial_estado = gr.State([])

# Estado global de memoria
registro_global = {
    "nombre": None,
    "edad": None,
    "ubicacion": None,
    "sexo": None,
    "estado_emocional": None,
    "ideacion": None,
    "plan": None,
    "metodo": None,
}

protocolo_global = {
    "paso_actual": "saludo",
    "pasos_completados": [],
    "pasos_faltantes": [
        "saludo",
        "datos_generales",
        "evaluacion_inicial",
        "evaluacion_riesgo",
        "intervencion",
        "escalamiento",
        "cierre",
    ],
}


theme = gr.themes.Ocean()
with gr.Blocks(theme=theme, css="""
                .boton-largo { height: 48%; min-height: 120px; }
                .flujo-etapa { font-weight: bold; padding: 6px; margin-bottom: 8px; transition: background-color 0.4s ease; }
                .flujo-verde { background-color: #DFFFD6; border-radius: 5px; }
                .flujo-normal { background-color: #f0f0f0; border-radius: 5px; }
                .flujo-etapa {
                    font-weight: bold;
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: 8px;
                    border-left: 5px solid #ccc;
                    background-color: #f9f9f9;
                    transition: all 0.3s ease;
                }
                .flujo-verde {
                    border-color: #4CAF50;
                    background-color: #e9fce9;
                }
                .flujo-actual {
                    border-color: #3f51b5;
                    background-color: #e3e7fd;
                }
                .flujo-pendiente {
                    border-color: #ccc;
                    background-color: #f0f0f0;
                }
                .subtitulo {
                    font-weight: normal;
                    color: #555;
                }
                .info-extra {
                    margin-top: 6px;
                    font-size: 0.85em;
                    color: #333;
                }
                .tiempo {
                    background-color: #dfe6e9;
                    padding: 3px 8px;
                    border-radius: 12px;
                    margin-right: 5px;
                }
                .emocion {
                    background-color: #dff9fb;
                    padding: 3px 8px;
                    border-radius: 12px;
                }
                .audio-grande {
                    min-height: 140px !important;
                }

                """) as interfaz:
    gr.Markdown("## 🤖 Spanish Empathic HCI System")

    # ----------------- TAB PRINCIPAL
    with gr.Tab("Empathic HCI System"):
        with gr.Row():

            # ----------------- PRIMERA COLUMNA
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(label="Conversación", value=[
                    ["🤖 Asistente", "Hola, soy tu asistente empático. ¿Con quién tengo el gusto?"]
                ])

                # ----------------- SUBIR AUDIO Y BOTONES
                with gr.Row():
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="🎤 Graba tu voz o sube un archivo",
                        elem_classes='audio-grande'
                    )
                    with gr.Column():
                        boton_procesar = gr.Button("🎙️ Procesar voz", elem_classes="boton-largo")
                        boton_llm = gr.Button("🧠 Obtener respuesta", elem_classes="boton-largo")
                        
                # ----------------- TRANSCRIPCIÓN
                with gr.Row():
                    texto_transcrito = gr.Textbox(label="📝 Transcription", interactive=False, )    
                    emocion_tts = gr.Textbox(visible=False)
                    kind_tts = gr.Textbox(visible=False)

                # ----------------- EMOCIÓN, PERFIL Y REGION
                with gr.Row():
                    emocion_detectada = gr.Textbox(visible=False)
                    certeza_emocion = gr.Textbox(visible=False)

                    with gr.Column(scale=2):
                        plot_emociones_main = gr.BarPlot(
                            x="Etiqueta",
                            y="Porcentaje",
                            label="🎭 Emotions (%)",
                            y_lim=(0, 100),
                            color="Grupo",
                            show_label=True
                        )
                                            
                    with gr.Column(scale=1):
                        plot_perfil_main = gr.BarPlot(
                                x="Etiqueta",
                                y="Porcentaje",
                                label="🧑‍🎤 Profile (%)",
                                y_lim=(0, 100),
                                color="Grupo",
                                show_label=True
                            )
                    
                    with gr.Column(scale=1):
                        plot_region_main = gr.BarPlot(
                            x="Etiqueta",
                            y="Porcentaje",
                            label="🌎 Region (%)",
                            y_lim=(0, 100),
                            color="Grupo",
                            show_label=True
                        )

                
                audio_tts = gr.Audio(label="🔊 Respuesta del asistente")

            # ----------------- SEGUNDA COLUMNA
            with gr.Column(scale=1):
                    # MODO DE CONVERSACIÓN (arriba del todo)
                    modo_selector = gr.Radio(
                        ["Protocolo", "Chat normal"],
                        value="Protocolo",
                        label="Modo de conversación"
                    )
                    boton_modo = gr.Button("✅ Aceptar modo")
                    modo_actual_texto = gr.Textbox(label="🛠️ Modo actual", interactive=False)
                    
                    flujo_vista = gr.HTML(value=flujo_inicial_html())
                    flujo_estado = gr.Textbox(label="🪄 Estado del flujo", interactive=False)
                    respuesta_texto_llm = gr.Textbox(visible=False)
                    panel_estado_llm = gr.Textbox(label="📋 Protocolo y estado del usuario", lines=10, interactive=False)


        pasos_flujo = gr.State([])

        # ----------------- SELECCIÓN DE MODO
        boton_modo.click(
            fn=cambiar_modo,
            inputs=[modo_selector],
            outputs=[
                chatbot,
                flujo_vista,
                flujo_estado,
                pasos_flujo,
                panel_estado_llm,
                modo_actual_texto  # nuevo output
            ]
        )

        # ----------------- TRANSCRIPCIÓN Y SER
        boton_procesar.click(
            fn=procesar_transcripcion_y_emocion,
            inputs=[audio_input],
            outputs=[
                texto_transcrito,
                emocion_detectada, certeza_emocion,
                plot_emociones_main,  # <-- nuevo
                plot_perfil_main,     # <-- nuevo
                plot_region_main,      # <-- nuevo
                flujo_estado,
                pasos_flujo
            ]
        ).then(
            fn=lambda pasos: actualizar_flujo_html(pasos, emocion=emocion_detectada.value, duracion="0.8 s"),
            inputs=[pasos_flujo],
            outputs=[flujo_vista]
        )

        # ----------------- LLM Y TTS
        boton_llm.click(
            fn=generar_respuesta_llm,
            inputs=[texto_transcrito, emocion_detectada, certeza_emocion, chatbot],
            outputs=[
                chatbot,                # historial
                respuesta_texto_llm,    # respuesta oculta
                flujo_estado,           # texto del estado
                pasos_flujo,            # lista de etapas
                emocion_tts,            # emoción (actualizada)
                kind_tts,
                panel_estado_llm        # protocolo
            ]
        ).then(
            fn=actualizar_flujo_html,
            inputs=[pasos_flujo],
            outputs=[flujo_vista]
        ).then(
            fn=sintetizar_respuesta_tts,
            inputs=[respuesta_texto_llm, emocion_tts, kind_tts],
            outputs=[audio_tts, flujo_estado, pasos_flujo]
        ).then(
            fn=actualizar_flujo_html,
            inputs=[pasos_flujo],
            outputs=[flujo_vista]
        )

    # ----------------- TAB SER
    with gr.Tab("Spanish SER"):
        gr.Markdown("### 🎭 Speech Emotion Recognition in Spanish")

        with gr.Row():
            audio_ser = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎙️ Entrada de voz para análisis SER",
                elem_classes='audio-grande'
            )
        with gr.Row():
            boton_analizar_ser = gr.Button("🔍 Analizar voz")

        with gr.Row():
            plot_emociones = gr.BarPlot(
                x="Etiqueta",
                y="Porcentaje",
                label="🎭 Emotions (%)",
                y_lim=(0, 100),
                color="Grupo",
                show_label=True
            )
            plot_perfil = gr.BarPlot(
                x="Etiqueta",
                y="Porcentaje",
                label="🧑‍🎤 Speaker profile (%)",
                y_lim=(0, 100),
                color="Grupo",
                show_label=True
            )
            plot_region = gr.BarPlot(
                x="Etiqueta",
                y="Porcentaje",
                label="🌎 Accent region (%)",
                y_lim=(0, 100),
                color="Grupo",
                show_label=True
            )

        boton_analizar_ser.click(
            fn=get_emotion,
            inputs=[audio_ser],
            outputs=[
                plot_emociones,
                plot_perfil,
                plot_region
            ]
        )

    # ----------------- TAB TTS
    with gr.Tab("Emotional TTS"):
        gr.Markdown("### 🗣️ Spanish Emotional Text-to-Speech")

        with gr.Row():
            emotion_tts = gr.Dropdown(
                label="Emotion",
                choices=['ANGER', 'DISGUST', 'FEAR', 'JOY', 'NEUTRAL_NORMAL', 'SURPRISE', 'SADNESS'],
                value="NEUTRAL_NORMAL"
            )
            kind_tts = gr.Dropdown(
                label="Kind",
                choices=["AFFIRMATIVE", "EXCLAMATORY", "PARAGRAPH", "INTERROGATIVE"],
                value="PARAGRAPH"
            )

        with gr.Row():
            input_text_tts = gr.Textbox(
                label="Enter the text to synthesize",
                placeholder="Type your message here...",
                lines=3
            )

        with gr.Row():
            button_tts = gr.Button("🗣️ Generate Voice")
            audio_output_tts = gr.Audio(label="🔊 Generated Audio")

        button_tts.click(
            fn=get_tts,
            inputs=[input_text_tts, emotion_tts, kind_tts],
            outputs=[audio_output_tts]
        )
            

interfaz.launch(
                #share=True
                )



