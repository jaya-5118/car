from __future__ import annotations

import os
from dotenv import load_dotenv
from typing import Tuple

import cv2
import gradio as gr
import numpy as np
import openrouteservice
import requests
import speech_recognition as sr
from gtts import gTTS
from folium import PolyLine
import folium
from youtubesearchpython import VideosSearch
from openai import OpenAI
from app.inference_service import SelfDrivingInferenceService, draw_detections_on_image
# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------
# Global config / clients
# ---------------------------------------------------------------------

DEFAULT_WEIGHTS = "yolov8n.pt"

# Load API keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ORS_API_KEY = os.getenv("ORS_API_KEY")

# Initialize API clients
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
ors_client = openrouteservice.Client(key=ORS_API_KEY) if ORS_API_KEY else None
def geocode_destination(name: str) -> tuple[float, float] | None:
    """Use OpenRouteService geocoding to convert a place name → (lon, lat)."""
    if not ors_client or not name.strip():
        return None
    try:
        res = ors_client.pelias_search(text=name, size=1)
        feats = res.get("features") or []
        if not feats:
            return None
        coords = feats[0]["geometry"]["coordinates"]
        lon, lat = float(coords[0]), float(coords[1])
        return lon, lat
    except Exception:
        return None
# ---------------------------------------------------------------------
# Shared model service
# ---------------------------------------------------------------------


def load_service(weights_path: str) -> SelfDrivingInferenceService:
    return SelfDrivingInferenceService(weights_path)


service_cache: SelfDrivingInferenceService | None = None


def get_service() -> SelfDrivingInferenceService:
    global service_cache
    if service_cache is None:
        service_cache = load_service(DEFAULT_WEIGHTS)
    return service_cache


def preprocess_image(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def postprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def tts_to_file(text: str, filename: str) -> str | None:
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(filename)
        return filename
    except Exception:
        return None


def transcribe_audio(audio_file) -> str:
    """Transcribe audio to text."""
    if not audio_file:
        return ""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text.lower()
    except Exception as e:
        return f"Transcription error: {e}"


# ---------------------------------------------------------------------
# NAVIGATION TAB
# ---------------------------------------------------------------------

def navigation_pipeline(
    origin_name: str,
    destination_name: str,
) -> tuple[str, str | None, str | None]:
    """
    - origin_name: free-form address (e.g. 'IIT Madras main gate')
    - destination_name: free-form address (e.g. 'Chennai Central Railway Station')

    Returns:
      - textual summary,
      - HTML map (string),
      - audio file path for TTS.
    """
    if not ors_client:
        msg = "OpenRouteService is not configured. Set ORS_API_KEY in your .env file."
        audio = tts_to_file(msg, "nav_error.mp3")
        return msg, "", audio or ""

    if not origin_name.strip():
        msg = "Enter your current location (for now, type it as text, e.g. 'IIT Delhi main gate')."
        audio = tts_to_file(msg, "nav_error.mp3")
        return msg, "", audio or ""

    if not destination_name.strip():
        msg = "Enter a destination name like 'IIT Delhi'."
        audio = tts_to_file(msg, "nav_error.mp3")
        return msg, "", audio

    # Geocode origin and destination
    origin_coords = geocode_destination(origin_name)
    dest_coords = geocode_destination(destination_name)
    if origin_coords is None:
        msg = f"Could not find location: '{origin_name}'. Try a more specific name."
        audio = tts_to_file(msg, "nav_error.mp3")
        return msg, "", audio
    if dest_coords is None:
        msg = f"Could not find destination: '{destination_name}'. Try a more specific name."
        audio = tts_to_file(msg, "nav_error.mp3")
        return msg, "", audio

    origin_lon, origin_lat = origin_coords
    dest_lon, dest_lat = dest_coords

    # Directions
    try:
        route = ors_client.directions(
            coordinates=[[origin_lon, origin_lat], [dest_lon, dest_lat]],
            profile="driving-car",
            format="geojson",
        )
    except Exception as e:
        msg = f"Could not fetch route from maps API: {e}."
        audio = tts_to_file(msg, "nav_error.mp3")
        return msg, "", audio

    feat = route["features"][0]
    props = feat["properties"]
    summary = props.get("summary", {})
    distance_m = summary.get("distance", 0.0)
    duration_s = summary.get("duration", 0.0)

    distance_km = distance_m / 1000.0
    duration_min = duration_s / 60.0

    # Very simple "traffic probability"
    ideal_time_min = (distance_km / 60.0) * 60.0 if distance_km > 0 else duration_min
    if ideal_time_min <= 0:
        traffic_level = "unknown"
    else:
        ratio = duration_min / ideal_time_min
        if ratio < 1.1:
            traffic_level = "low"
        elif ratio < 1.5:
            traffic_level = "moderate"
        else:
            traffic_level = "heavy"

    text_summary = (
        f"Best route from **{origin_name}** to **{destination_name}** "
        f"is **{distance_km:.1f} km** with estimated travel time **{duration_min:.0f} minutes**. "
        f"Estimated traffic level: **{traffic_level}**."
    )

    # Build map
    coords = feat["geometry"]["coordinates"]  # [lon, lat] pairs
    poly_latlon = [[c[1], c[0]] for c in coords]

    m = folium.Map(location=[origin_lat, origin_lon], zoom_start=13)
    PolyLine(poly_latlon, color="#2563eb", weight=5, opacity=0.9).add_to(m)
    folium.Marker(
        [origin_lat, origin_lon],
        popup=origin_name,
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)
    folium.Marker(
        [dest_lat, dest_lon],
        popup=destination_name,
        icon=folium.Icon(color="red", icon="flag"),
    ).add_to(m)

    map_html = m._repr_html_()

    audio_path = tts_to_file(
        f"Route from {origin_name} to {destination_name}, distance {distance_km:.1f} kilometers, "
        f"travel time about {duration_min:.0f} minutes. "
        f"Traffic appears {traffic_level}.",
        "nav_summary.mp3",
    )

    return text_summary, map_html, audio_path or ""

# ---------------------------------------------------------------------
# PERCEPTION TAB (YOLO)
# ---------------------------------------------------------------------


def perception_pipeline(
    image: np.ndarray,
    show_heatmap: bool,
    tta: bool,
) -> Tuple[np.ndarray, str, float, str | None]:
    if image is None:
        raise gr.Error("Please upload a road / driving scene image.")

    service = get_service()
    img_bgr = preprocess_image(image)

    if tta:
        img_flip = cv2.flip(img_bgr, 1)
        det1 = service.predict(img_bgr)
        det2 = service.predict(img_flip)
        det = det1
    else:
        det = service.predict(img_bgr)

    vis_bgr = draw_detections_on_image(img_bgr, det)

    if show_heatmap:
        h, w, _ = vis_bgr.shape
        grid_h, grid_w = det.risk_grid.shape
        tile_h = h // grid_h
        tile_w = w // grid_w
        overlay = vis_bgr.copy()
        for gy in range(grid_h):
            for gx in range(grid_w):
                v = det.risk_grid[gy, gx]
                if v < 0.15:
                    continue
                x1 = gx * tile_w
                y1 = gy * tile_h
                x2 = min(w, (gx + 1) * tile_w)
                y2 = min(h, (gy + 1) * tile_h)
                color = (0, int(255 * (1 - v)), int(255 * v))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        vis_bgr = cv2.addWeighted(overlay, 0.25, vis_bgr, 0.75, 0)

    vis_rgb = postprocess_image(vis_bgr)
    explanation = det.scenario_summary
    instruction = det.driving_instruction

    audio_path = tts_to_file(
        f"{explanation} Driving instruction: {instruction}",
        "perception_guidance.mp3",
    )

    return vis_rgb, explanation, float(det.risk_score), audio_path


# ---------------------------------------------------------------------
# ASSISTANT TAB
# ---------------------------------------------------------------------


def assistant_chat(history: list[list[str]] | None, message: str) -> list[list[str]]:
    """
    history: [[user, assistant], ...]
    message: latest user message
    returns: updated history
    """
    history = history or []
    user_msg = message.strip()
    if not user_msg:
        return history

    # Handle "play music" locally by giving a YouTube link
    lower = user_msg.lower()
    if "play" in lower and "music" in lower:
        q = lower.replace("play", "").replace("music", "").strip() or "music"
        url = f"https://www.youtube.com/results?search_query={q.replace(' ', '+')}"
        bot = f"Here’s music for **{q}**: {url}"
        return history + [[user_msg, bot]]

    if not OPENAI_API_KEY:
        bot = (
            "Chat assistant is not configured with an API key right now. "
            "I can still handle simple commands like 'play music'."
        )
        return history + [[user_msg, bot]]

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an in-car assistant for a self-driving car demo website. "
                    "Explain how to use the Navigation, Perception, and Assistant tabs. "
                    "Keep answers short and friendly."
                ),
            }
        ]
        for u, a in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_msg})

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        bot = resp.choices[0].message.content
    except Exception as e:
        bot = f"Assistant error: {e}"

    return history + [[user_msg, bot]]


# ---------------------------------------------------------------------
# ENTERTAINMENT TAB
# ---------------------------------------------------------------------

def get_weather(location: str) -> str:
    """Get weather for location."""
    try:
        url = f"https://wttr.in/{location}?format=j1"
        response = requests.get(url, timeout=10)
        data = response.json()
        current = data['current_condition'][0]
        temp = current['temp_C']
        weather = current['weatherDesc'][0]['value']
        return f"Temperature: {temp}°C, Weather: {weather}"
    except Exception:
        return "Weather data not available. Check location name."

def recommend_entertainment(weather: str, query: str, language: str) -> str:
    """Use OpenAI to recommend entertainment based on weather and query."""
    if not OPENAI_API_KEY:
        return f"Recommended: {query} in {language}"
    prompt = f"Based on current weather '{weather}', user wants '{query}'. Recommend a YouTube search term for music or video in {language} language. Keep it short."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return f"Recommended: {query}"

def search_youtube(query: str) -> str:
    """Search YouTube and return embed HTML."""
    try:
        videos_search = VideosSearch(query, limit=1)
        results = videos_search.result().get('result', [])
        if results:
            video_id = results[0]['id']
            return f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
    except Exception:
        pass
    return "No video found. Try a different query."

def entertainment_pipeline(location: str, language: str, query: str) -> tuple[str, str, str]:
    """Pipeline for entertainment."""
    weather = get_weather(location)
    recommendation = recommend_entertainment(weather, query, language)
    video_html = search_youtube(recommendation)
    return weather, f"Recommendation: {recommendation}", video_html


# ---------------------------------------------------------------------
# BUILD UI WITH 4 TABS
# ---------------------------------------------------------------------


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Self-Driving Car Assistant") as demo:
        gr.Markdown(
            "## Self-Driving Car Assistant\n"
            "A three-tab demo: **Navigation**, **Perception**, and **Assistant**."
        )

        with gr.Tabs():
            # NAVIGATION TAB
            with gr.Tab("Navigation"):
                with gr.Row():
                    with gr.Column():
                        origin = gr.Textbox(
                            label="Current location (text)",
                            placeholder="e.g. IIT Delhi main gate",
                        )
                        destination = gr.Textbox(
                            label="Destination (text)",
                            placeholder="e.g. Delhi Airport T3",
                        )
                        voice_audio = gr.Audio(label="Voice destination input", sources=["microphone"])
                        transcribe_btn = gr.Button("Transcribe voice to destination")
                        nav_btn = gr.Button("Plan route", variant="primary")

                    with gr.Column():
                        nav_summary = gr.Markdown(label="Route summary")
                        nav_map = gr.HTML(label="Route map")
                        nav_audio = gr.Audio(label="Voice navigation", autoplay=True)

                transcribe_btn.click(
                    fn=transcribe_audio,
                    inputs=[voice_audio],
                    outputs=[destination],
                )

                nav_btn.click(
                    fn=navigation_pipeline,
                    inputs=[origin, destination],
                    outputs=[nav_summary, nav_map, nav_audio],
                )

            # PERCEPTION TAB
            with gr.Tab("Perception"):
                with gr.Row():
                    with gr.Column():
                        perc_image = gr.Image(
                            label="Upload / capture road image",
                            type="numpy",
                            sources=["upload", "webcam"],
                        )
                        show_heat = gr.Checkbox(value=True, label="Overlay risk heatmap")
                        use_tta = gr.Checkbox(value=False, label="Robust mode (TTA)")
                        perc_btn = gr.Button("Analyze scene", variant="primary")

                    with gr.Column():
                        perc_output = gr.Image(
                            label="Perception & risk visualization",
                            type="numpy",
                        )
                        perc_risk = gr.Number(
                            label="Global risk score (0 = safe, 1 = high risk)",
                            precision=3,
                        )
                        perc_text = gr.Markdown(label="Scene & risk explanation")
                        perc_audio = gr.Audio(label="Voice guidance", autoplay=True)

                perc_btn.click(
                    fn=perception_pipeline,
                    inputs=[perc_image, show_heat, use_tta],
                    outputs=[perc_output, perc_text, perc_risk, perc_audio],
                )

            # ASSISTANT TAB
            with gr.Tab("Assistant"):
                chatbot = gr.Chatbot(label="In-car assistant")
                msg = gr.Textbox(label="Ask me anything", placeholder="How do I use navigation?")
                send = gr.Button("Send", variant="primary")

                def chat_wrapper(history, message):
                    return assistant_chat(history, message), ""

                send.click(
                    fn=chat_wrapper,
                    inputs=[chatbot, msg],
                    outputs=[chatbot, msg],
                )

            # # ENTERTAINMENT TAB
            # with gr.Tab("Entertainment"):
            #     with gr.Row():
            #         with gr.Column():
            #             ent_location = gr.Textbox(
            #                 label="Your location",
            #                 placeholder="e.g. Delhi",
            #             )
            #             ent_language = gr.Dropdown(
            #                 choices=["en", "hi", "es", "fr", "de"],
            #                 value="en",
            #                 label="Preferred language",
            #             )
            #             ent_query = gr.Textbox(
            #                 label="What to play/watch",
            #                 placeholder="e.g. relaxing music, funny videos",
            #             )
            #             ent_btn = gr.Button("Get Entertainment", variant="primary")

            #         with gr.Column():
            #             ent_weather = gr.Markdown(label="Current Weather")
            #             ent_recommendation = gr.Markdown(label="AI Recommendation")
            #             ent_video = gr.HTML(label="YouTube Video")

            #     ent_btn.click(
            #         fn=entertainment_pipeline,
            #         inputs=[ent_location, ent_language, ent_query],
            #         outputs=[ent_weather, ent_recommendation, ent_video],
            #     )

        return demo

if __name__ == "__main__":
    app = build_interface()
    app.launch()