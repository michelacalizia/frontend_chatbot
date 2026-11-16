"""Interazione con Ollama (Qwen) con output JSON forzato + calcoli deterministici in Python."""
import json
import math
from collections import defaultdict
from functools import lru_cache

import ollama
from pydantic import ValidationError

from app.config import settings
from app.rag_engine import match_user_ingredients
from app.schemas import (
    DayMenu,
    MealEntry,
    MenuPlanLLM,
    Recipe,
    RescueTipsLLM,
    ShoppingListItem,
    WeeklyMenuRequest,
    WeeklyMenuResponse,
)

MONTHS_IT = [
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
]
CATEGORY_ORDER = {"Ortofrutta": 0, "Pescheria": 1, "Latticini": 2, "Dispensa": 3}

MENU_SYSTEM_PROMPT = """Sei un nutrizionista esperto di Dieta Mediterranea e di cucina anti-spreco.
Devi scegliere la ricetta di pranzo e di cena per ogni giorno richiesto.

REGOLE OBBLIGATORIE:
1. Usa ESCLUSIVAMENTE le ricette fornite. Indica solo gli id, copiati ESATTAMENTE dal contesto.
2. "days" contiene un elemento per ogni giorno, in ordine, con "lunch_id" e "dinner_id".
3. LOGICA ANTI-SPRECO: riutilizza gli ingredienti freschi e deperibili (verdure, pesce, ricotta,
   uova, erbe aromatiche) in più pasti ravvicinati, così da non avere avanzi. Metti i pasti con
   pesce e ingredienti molto deperibili nei primi giorni. Dai priorità agli ingredienti già in frigo.
4. Varia le ricette quando possibile, ma ripetere una ricetta è consentito se evita sprechi.
5. "anti_waste_notes": al massimo 3 frasi brevi in italiano, che citano giorni e ingredienti riutilizzati.
6. Rispondi SOLO con JSON valido conforme allo schema."""

RESCUE_SYSTEM_PROMPT = """Sei un cuoco esperto di cucina mediterranea e anti-spreco.
Per ogni ricetta fornita scrivi UN solo testo "tips" con 2 suggerimenti brevi (max 25 parole l'uno),
coerenti con QUELLA ricetta.
REGOLE:
- Parla solo di ingredienti presenti in "ingredienti_della_ricetta" o in "ingredienti_dell_utente".
- Non nominare altri ingredienti (niente legumi, formaggi o altro che non sia in elenco),
  tranne aggiunte semplici come erbe aromatiche, limone o pane.
- Se un ingrediente dell'utente è tra "ingredienti_utente_non_presenti", suggerisci come
  servirlo come contorno o come conservarlo per non sprecarlo.
- Non inventare ricette nuove.
Usa "recipe_id" copiato ESATTAMENTE dal contesto. Rispondi SOLO con JSON valido conforme allo schema."""


@lru_cache(maxsize=1)
def _client() -> ollama.Client:
    return ollama.Client(host=settings.ollama_host)


def _chat_structured(system: str, user: str, schema: dict, max_tokens: int) -> str:
    response = _client().chat(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format=schema,
        options={"temperature": settings.llm_temperature, "num_predict": max_tokens},
        keep_alive="30m",  # tiene il modello in memoria tra una richiesta e l'altra
    )
    return response.message.content


def _recipe_for_menu_prompt(recipe: Recipe) -> dict:
    # Contesto compatto: solo id, titolo, tag e ingredienti deperibili
    return {
        "id": recipe.id,
        "title": recipe.title,
        "tag": recipe.diet_tags,
        "deperibili": [i.name for i in recipe.ingredients if i.is_perishable],
    }


def generate_weekly_menu(
    request: WeeklyMenuRequest, candidates: list[Recipe]
) -> WeeklyMenuResponse:
    context = {
        "mese_corrente": MONTHS_IT[request.current_month - 1],
        "giorni": request.days,
        "ingredienti_in_frigo": request.fridge_ingredients,
        "restrizioni_alimentari": request.dietary_restrictions,
        "ricette_candidate": [_recipe_for_menu_prompt(r) for r in candidates],
    }
    user_prompt = (
        f"Scegli pranzo e cena per {request.days} giorni usando solo queste ricette.\n"
        + json.dumps(context, ensure_ascii=False)
    )
    schema = MenuPlanLLM.model_json_schema()

    last_error: Exception | None = None
    for _ in range(2):  # un secondo tentativo se il JSON non è valido
        raw = _chat_structured(MENU_SYSTEM_PROMPT, user_prompt, schema, max_tokens=1200)
        try:
            plan = MenuPlanLLM.model_validate_json(raw)
        except ValidationError as exc:
            last_error = exc
            continue

        by_id = {r.id: r for r in candidates}

        def entry(recipe_id: str) -> MealEntry:
            recipe = by_id.get(recipe_id)
            return MealEntry(recipe_id=recipe_id, recipe_title=recipe.title if recipe else "")

        menu = [
            DayMenu(day=i, lunch=entry(d.lunch_id), dinner=entry(d.dinner_id))
            for i, d in enumerate(plan.days[: request.days], start=1)
        ]
        return WeeklyMenuResponse(menu=menu, anti_waste_notes=plan.anti_waste_notes)

    raise ValueError(f"Il modello non ha restituito un JSON valido: {last_error}")


def generate_rescue_tips(ingredients: list[str], recipes: list[Recipe]) -> dict[str, str]:
    ricette_ctx = []
    for r in recipes:
        used, unused = match_user_ingredients(r, ingredients)
        ricette_ctx.append(
            {
                "id": r.id,
                "title": r.title,
                "ingredienti_della_ricetta": [i.name for i in r.ingredients],
                "ingredienti_utente_usati": used,
                "ingredienti_utente_non_presenti": unused,
            }
        )
    context = {"ingredienti_dell_utente": ingredients, "ricette": ricette_ctx}
    user_prompt = json.dumps(context, ensure_ascii=False)
    schema = RescueTipsLLM.model_json_schema()

    for _ in range(2):
        raw = _chat_structured(RESCUE_SYSTEM_PROMPT, user_prompt, schema, max_tokens=700)
        try:
            parsed = RescueTipsLLM.model_validate_json(raw)
            return {s.recipe_id: s.tips for s in parsed.suggestions}
        except ValidationError:
            continue
    return {}


# ---------------------------------------------------------------------------
# Post-elaborazione deterministica (Python puro, nessun calcolo delegato al LLM)
# ---------------------------------------------------------------------------
def normalize_menu(
    llm_menu: list[DayMenu], candidates: list[Recipe], days: int
) -> list[DayMenu]:
    """Garantisce `days` giorni e che ogni pasto punti a una ricetta realmente candidata."""
    by_id = {r.id: r for r in candidates}
    by_title = {r.title.strip().lower(): r for r in candidates}
    by_day = {d.day: d for d in llm_menu}
    fallback_cycle = 0

    def resolve(entry: MealEntry | None) -> MealEntry:
        nonlocal fallback_cycle
        recipe = None
        if entry is not None:
            recipe = by_id.get(entry.recipe_id) or by_title.get(
                entry.recipe_title.strip().lower()
            )
        if recipe is None:
            recipe = candidates[fallback_cycle % len(candidates)]
            fallback_cycle += 1
        return MealEntry(recipe_id=recipe.id, recipe_title=recipe.title)

    menu: list[DayMenu] = []
    for day in range(1, days + 1):
        original = by_day.get(day)
        menu.append(
            DayMenu(
                day=day,
                lunch=resolve(original.lunch if original else None),
                dinner=resolve(original.dinner if original else None),
            )
        )
    return menu


def compute_shopping_list(
    menu: list[DayMenu], candidates: list[Recipe], portions: int
) -> list[ShoppingListItem]:
    """Aggrega e somma le quantità (scalate sulle porzioni) per ingrediente, unità e corsia."""
    by_id = {r.id: r for r in candidates}
    quantities: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    display_names: dict[tuple[str, str, str], str] = {}

    for day in menu:
        for meal in (day.lunch, day.dinner):
            recipe = by_id[meal.recipe_id]
            factor = portions / recipe.base_portions
            for ing in recipe.ingredients:
                key = (ing.name.strip().lower(), ing.unit, ing.category)
                quantities[key].append(ing.quantity * factor)
                display_names.setdefault(key, ing.name.strip())

    items = [
        ShoppingListItem(
            name=display_names[key],
            total_quantity=round(math.fsum(values), 2),
            unit=key[1],
            category=key[2],
        )
        for key, values in quantities.items()
    ]
    items.sort(key=lambda i: (CATEGORY_ORDER.get(i.category, 99), i.name.lower()))
    return items