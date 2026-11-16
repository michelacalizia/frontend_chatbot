"""Entrypoint FastAPI: Dieta Mediterranea Anti-Spreco."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
import ollama
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import llm_service, rag_engine
from app.schemas import (
    FridgeRescueResponse,
    RescueResult,
    WeeklyMenuRequest,
    WeeklyMenuResponse,
)

OLLAMA_ERRORS = (ollama.ResponseError, ConnectionError, httpx.ConnectError)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Indicizza automaticamente il dataset iniziale se il DB è vuoto (best effort).
    try:
        if rag_engine.get_collection().count() == 0:
            rag_engine.ingest_recipes()
    except Exception as exc:  # Ollama può non essere ancora avviato
        print(f"[startup] Ingest automatico saltato: {exc}")
    yield


app = FastAPI(
    title="Dieta Mediterranea Anti-Spreco API",
    description="Backend RAG: ChromaDB + Ollama (Qwen) per menu settimanali e lista della spesa.",
    version="1.0.0",
    lifespan=lifespan,
)


def _ollama_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=f"Ollama non raggiungibile o modello non disponibile: {exc}",
    )


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok", "recipes_indexed": rag_engine.get_collection().count()}


@app.post("/api/v1/ingest", tags=["rag"])
def ingest() -> dict:
    """Carica/aggiorna ChromaDB leggendo data/seed_recipes.json."""
    try:
        count = rag_engine.ingest_recipes()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except OLLAMA_ERRORS as exc:
        raise _ollama_unavailable(exc)
    return {"status": "ok", "recipes_indexed": count}


@app.post("/api/v1/generate-menu", response_model=WeeklyMenuResponse, tags=["menu"])
def generate_menu(request: WeeklyMenuRequest) -> WeeklyMenuResponse:
    """Menu settimanale anti-spreco con lista della spesa aggregata."""
    try:
        candidates = rag_engine.retrieve_candidate_recipes(request)
        if not candidates:
            raise HTTPException(
                status_code=404,
                detail="Nessuna ricetta candidata: esegui /api/v1/ingest o allenta le restrizioni.",
            )
        llm_response = llm_service.generate_weekly_menu(request, candidates)
    except HTTPException:
        raise
    except OLLAMA_ERRORS as exc:
        raise _ollama_unavailable(exc)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

        menu = llm_service.normalize_menu(llm_response.menu, candidates, request.days)
    shopping_list = llm_service.compute_shopping_list(menu, candidates, request.portions)

    # Ricette usate nel menu, con quantità già scalate sulle porzioni richieste
    used_ids = {meal.recipe_id for day in menu for meal in (day.lunch, day.dinner)}
    recipes = []
    for r in candidates:
        if r.id not in used_ids:
            continue
        factor = request.portions / r.base_portions
        recipes.append(
            r.model_copy(
                update={
                    "base_portions": request.portions,
                    "ingredients": [
                        i.model_copy(update={"quantity": round(i.quantity * factor, 2)})
                        for i in r.ingredients
                    ],
                }
            )
        )

    return WeeklyMenuResponse(
        menu=menu,
        shopping_list=shopping_list,
        anti_waste_notes=llm_response.anti_waste_notes,
        recipes=recipes,
    )


@app.post("/api/v1/fridge-rescue", response_model=FridgeRescueResponse, tags=["menu"])
def fridge_rescue(
    ingredients: Annotated[list[str], Body(description="2-4 ingredienti in scadenza")],
) -> FridgeRescueResponse:
    """Le 3 migliori ricette per ingredienti in scadenza + suggerimenti di Qwen."""
    cleaned = [i.strip() for i in ingredients if i.strip()]
    if not 2 <= len(cleaned) <= 4:
        raise HTTPException(status_code=422, detail="Indica da 2 a 4 ingredienti.")

    try:
        recipes = rag_engine.retrieve_recipes_for_ingredients(cleaned, top_k=3)
        if not recipes:
            raise HTTPException(status_code=404, detail="Nessuna ricetta trovata: esegui /api/v1/ingest.")
        tips = llm_service.generate_rescue_tips(cleaned, recipes)
    except HTTPException:
        raise
    except OLLAMA_ERRORS as exc:
        raise _ollama_unavailable(exc)

    return FridgeRescueResponse(
        results=[
            RescueResult(recipe=r, suggestions=tips.get(r.id, "Nessun suggerimento disponibile."))
            for r in recipes
        ]
    )


# ---------------------------------------------------------------------------
# Frontend: la cartella "frontend" viene servita su http://localhost:8000/app/
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/app/")


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")