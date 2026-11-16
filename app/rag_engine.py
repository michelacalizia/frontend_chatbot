"""ChromaDB persistente + embeddings Ollama: indicizzazione e ricerca semantica."""
import json
from functools import lru_cache
from pathlib import Path

import chromadb
import ollama

from app.config import settings
from app.schemas import Ingredient, Recipe, WeeklyMenuRequest

@lru_cache(maxsize=1)
def _ollama_client() -> ollama.Client:
    return ollama.Client(host=settings.ollama_host)


@lru_cache(maxsize=1)
def _chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.chroma_path)


def get_collection():
    return _chroma_client().get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = _ollama_client().embed(model=settings.embedding_model, input=texts)
    return [list(v) for v in response["embeddings"]]


def _recipe_document(recipe: Recipe) -> str:
    ingredients = ", ".join(i.name for i in recipe.ingredients)
    tags = ", ".join(recipe.diet_tags)
    return (
        f"{recipe.title}. Ingredienti: {ingredients}. "
        f"Tag dieta: {tags}. Preparazione: {recipe.instructions}"
    )


def _recipe_metadata(recipe: Recipe) -> dict:
    meta: dict = {
        "prep_time_min": recipe.prep_time_min,
        "diet_tags": ",".join(t.lower() for t in recipe.diet_tags),
        "season_months": ",".join(str(m) for m in recipe.season_months),
        "recipe_json": recipe.model_dump_json(),
    }
    # Un flag booleano per mese: permette il pre-filtraggio stagionale in Chroma.
    for month in range(1, 13):
        meta[f"m{month}"] = month in recipe.season_months
    return meta


def ingest_recipes(json_path: str | None = None) -> int:
    """Carica (upsert) le ricette dal JSON in ChromaDB. Restituisce il numero indicizzato."""
    path = Path(json_path or settings.seed_path)
    if not path.exists():
        raise FileNotFoundError(f"File ricette non trovato: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    recipes = [Recipe.model_validate(item) for item in raw]
    if not recipes:
        return 0

    documents = [_recipe_document(r) for r in recipes]
    collection = get_collection()
    collection.upsert(
        ids=[r.id for r in recipes],
        embeddings=embed_texts(documents),
        documents=documents,
        metadatas=[_recipe_metadata(r) for r in recipes],
    )
    return len(recipes)


def _parse_results(result: dict) -> list[tuple[Recipe, float]]:
    metadatas = result.get("metadatas") or [[]]
    distances = result.get("distances") or [[]]
    parsed: list[tuple[Recipe, float]] = []
    for meta, dist in zip(metadatas[0], distances[0]):
        parsed.append((Recipe.model_validate_json(meta["recipe_json"]), float(dist)))
    return parsed


def _matches_restrictions(recipe: Recipe, restrictions: list[str]) -> bool:
    tags = {t.strip().lower() for t in recipe.diet_tags}
    return all(r.strip().lower() in tags for r in restrictions)


def retrieve_candidate_recipes(
    request: WeeklyMenuRequest, top_k: int | None = None
) -> list[Recipe]:
    """Ricette più pertinenti agli ingredienti in frigo, filtrate per mese e restrizioni."""
    top_k = top_k or settings.retrieval_top_k
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    if request.fridge_ingredients:
        query_text = "Ricetta mediterranea con " + ", ".join(request.fridge_ingredients)
    else:
        query_text = "Ricetta mediterranea di stagione equilibrata e salutare"
    query_embedding = embed_texts([query_text])[0]
    n_results = min(total, max(top_k * 3, top_k))

    def _query(where: dict | None) -> list[Recipe]:
        res = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["metadatas", "distances"],
        )
        return [r for r, _ in _parse_results(res)]

    # 1) Ricette di stagione, 2) a seguire le altre (solo se servono più candidate).
    candidates = _query({f"m{request.current_month}": True})
    seen = {r.id for r in candidates}
    if len(candidates) < 4:
        for r in _query(None):
            if r.id not in seen:
                candidates.append(r)
                seen.add(r.id)

    if request.dietary_restrictions:
        candidates = [
            r for r in candidates
            if _matches_restrictions(r, request.dietary_restrictions)
        ]
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Ricerca per ingredienti in scadenza (con parole generiche: pesce, verdure...)
# ---------------------------------------------------------------------------
def _is_vegetable(ing: Ingredient) -> bool:
    # Verdura "vera": ortofrutta a peso, almeno 100 g (esclude aglio, erbe, limone)
    return ing.category == "Ortofrutta" and ing.unit == "g" and ing.quantity >= 100


def _name_has(ing: Ingredient, words: list[str]) -> bool:
    name = ing.name.lower()
    return any(w in name for w in words)


def _is_fish(ing: Ingredient) -> bool:
    return ing.category == "Pescheria"


def _is_cheese(ing: Ingredient) -> bool:
    return _name_has(ing, ["ricotta", "parmigiano", "mozzarella"])


def _is_legume(ing: Ingredient) -> bool:
    return _name_has(ing, ["ceci", "lenticchie", "fagioli", "fave"])


def _is_dairy(ing: Ingredient) -> bool:
    return ing.category == "Latticini"


GENERIC_RULES = {
    "pesce": _is_fish,
    "pesci": _is_fish,
    "verdura": _is_vegetable,
    "verdure": _is_vegetable,
    "ortaggi": _is_vegetable,
    "formaggio": _is_cheese,
    "formaggi": _is_cheese,
    "legumi": _is_legume,
    "latticini": _is_dairy,
}


def _stem(word: str) -> str:
    # toglie l'ultima vocale: uovo/uova, zucchina/zucchine
    return word[:-1] if len(word) > 3 and word[-1] in "aeio" else word


def _ingredient_matches(ing: Ingredient, term: str) -> bool:
    term = term.strip().lower()
    rule = GENERIC_RULES.get(term)
    if rule:
        return rule(ing)
    name = ing.name.lower()
    return _stem(term) in name or _stem(name) in term


def match_user_ingredients(recipe: Recipe, wanted: list[str]) -> tuple[list[str], list[str]]:
    """Restituisce (ingredienti dell'utente usati dalla ricetta, ingredienti non presenti)."""
    used = [w for w in wanted if any(_ingredient_matches(i, w) for i in recipe.ingredients)]
    unused = [w for w in wanted if w not in used]
    return used, unused


def retrieve_recipes_for_ingredients(ingredients: list[str], top_k: int = 3) -> list[Recipe]:
    """Ricette ordinate per quanti ingredienti dell'utente usano (poi per similarità)."""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    wanted = [w.strip() for w in ingredients if w.strip()]
    query_text = "Ricetta mediterranea che usa: " + ", ".join(wanted)
    res = collection.query(
        query_embeddings=[embed_texts([query_text])[0]],
        n_results=total,
        include=["metadatas", "distances"],
    )
    scored = _parse_results(res)
    scored.sort(key=lambda item: (-len(match_user_ingredients(item[0], wanted)[0]), item[1]))
    return [r for r, _ in scored[:top_k]]