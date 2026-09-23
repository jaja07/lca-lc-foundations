# Module 1 — Fiche de révision

Fondations LangChain v1 : modèles, prompting, outils, mémoire, multimodal.
Établie à partir des sept notebooks de `notebooks/module-1/`.

---

## Carte du module

| Notebook | Notion centrale | À retenir absolument |
|---|---|---|
| `1.1_foundational_models` | Modèle vs agent | `init_chat_model` / `create_agent`, les deux formes d'`invoke` |
| `1.1_prompting` | Piloter le comportement | `system_prompt`, few-shot, `response_format` |
| `1.2_tools` | Donner des capacités | `@tool`, la docstring **est** la description |
| `1.2_web_search` | Dépasser le cutoff | Un outil qui appelle une API externe |
| `1.3_memory` | Persister l'état | `checkpointer` + `thread_id` |
| `1.4_multimodal_messages` | Au-delà du texte | `content` en liste de blocs typés |
| `1.5_personal_chef` | Synthèse | Les quatre briques assemblées |

---

## 1. Le partage fondamental : modèle ou agent ?

C'est la distinction structurante de tout le module. Tout le reste en découle.

### Le modèle — un appel, une réponse

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(model=MODEL)
response = model.invoke("What's the capital of the Moon?")

print(response.content)          # le texte
pprint(response.response_metadata)  # tokens, fournisseur, finish_reason
```

- Entrée : une **chaîne** (ou une liste de messages)
- Sortie : un **objet `AIMessage`**, dont on lit `.content`
- Aucune boucle, aucun outil, aucun état

### L'agent — une boucle de raisonnement

```python
from langchain.agents import create_agent
from langchain.messages import HumanMessage

agent = create_agent(model=MODEL)
response = agent.invoke(
    {"messages": [HumanMessage(content="What's the capital of the Moon?")]}
)

print(response['messages'][-1].content)
```

- Entrée : un **dictionnaire d'état**, `{"messages": [...]}`
- Sortie : un **dictionnaire**, dont `response["messages"]` est l'historique complet
- Peut boucler : appeler un outil, lire le résultat, rappeler le modèle

> **Le piège numéro un du module.** `model.invoke("texte")` accepte une chaîne.
> `agent.invoke("texte")` ne fonctionne pas — il exige `{"messages": [...]}`.
> Symétriquement, `response.content` sur un modèle, `response["messages"][-1].content`
> sur un agent.

### Les trois écritures équivalentes de `create_agent`

```python
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
agent = create_agent(model=model)   # un objet modèle déjà construit
agent = create_agent(model=MODEL)   # une chaîne "fournisseur:modèle", nommée
agent = create_agent(MODEL)         # la même, en positionnel
```

`create_agent` accepte indifféremment un **objet modèle** ou une **chaîne**. Dans le
second cas il appelle `init_chat_model` pour toi.

---

## 2. Choisir et paramétrer le modèle

### Deux voies vers le même objet

```python
# Voie générique — la chaîne porte le fournisseur
model = init_chat_model(model="google_genai:gemini-3.1-flash-lite")

# Voie spécifique — la classe porte le fournisseur
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
```

La première est **agnostique** : changer de fournisseur ne change qu'une chaîne. C'est
l'argument de vente de LangChain, et la raison pour laquelle ce dépôt a pu basculer
d'OpenAI vers Gemini sans réécrire la logique.

### Les paramètres du modèle passent en kwargs

```python
model = init_chat_model(
    model=MODEL,
    temperature=1.0,   # transmis tel quel au fournisseur
)
```

Tout kwargs non reconnu par LangChain est transmis au fournisseur sous-jacent.

### Le streaming

```python
for token, metadata in agent.stream(
    {"messages": [HumanMessage(content="Tell me about Luna City")]},
    stream_mode="messages"
):
    if token.content:
        print(token.content, end="", flush=True)
```

`stream_mode="messages"` émet un couple `(token, metadata)`. `metadata` indique **quel
nœud** du graphe a produit le token — utile dès qu'il y a plusieurs agents.

---

## 3. Le prompting — quatre niveaux de contrainte

Progression du plus souple au plus rigide. Sache les nommer et les distinguer.

### Niveau 1 — Le `system_prompt`

```python
agent = create_agent(
    model=MODEL,
    system_prompt="You are a science fiction writer, create a capital city at the users request."
)
```

Fixe le rôle et le ton. N'impose aucune forme.

### Niveau 2 — Le few-shot

Des exemples **dans** le prompt système :

```python
system_prompt = """
You are a science fiction writer, create a space capital city at the users request.

User: What is the capital of mars?
Scifi Writer: Marsialis

User: What is the capital of Venus?
Scifi Writer: Venusovia
"""
```

Le modèle infère le format depuis les exemples. Plus fiable que la consigne seule.

### Niveau 3 — Le prompt structuré

On **décrit** la structure en langage naturel :

```python
system_prompt = """
Please keep to the below structure.

Name: The name of the capital city
Location: Where it is based
Vibe: 2-3 words to describe its vibe
Economy: Main industries
"""
```

Toujours du texte libre en sortie. Rien ne **garantit** le respect du format.

### Niveau 4 — La sortie structurée (la seule garantie)

```python
from pydantic import BaseModel

class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str

agent = create_agent(
    model=MODEL,
    system_prompt="You are a science fiction writer...",
    response_format=CapitalInfo
)

response = agent.invoke({"messages": [question]})

info = response["structured_response"]   # une instance de CapitalInfo
print(info.name)                         # accès par attribut, pas par clé
```

> **À mémoriser.** `response_format=<Modèle Pydantic>` ajoute la clé
> `"structured_response"` au dictionnaire retourné. C'est un **objet Python typé**,
> pas un dictionnaire : on écrit `info.name`, jamais `info["name"]`.
> La clé `"messages"` reste présente à côté.

---

## 4. Les outils

### Définir — trois signatures du décorateur

```python
from langchain.tools import tool

# 1. Le nom vient de la fonction, la description de la docstring
@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

# 2. Nom explicite, description toujours issue de la docstring
@tool("square_root")
def tool1(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

# 3. Nom et description explicites, docstring devenue inutile
@tool("square_root", description="Calculate the square root of a number")
def tool1(x: float) -> float:
    return x ** 0.5
```

> **Le point que l'examen aimera.** Dans les formes 1 et 2, la **docstring devient la
> description** transmise au modèle. Elle n'est pas décorative : c'est sur elle que le
> modèle décide d'appeler l'outil ou non. Les **annotations de type** (`x: float`)
> construisent le schéma des arguments. Une docstring vague ou un type absent dégradent
> directement le taux d'appel correct.

### Tester un outil isolément

```python
tool1.invoke({"x": 467})   # un dict, dont les clés sont les noms de paramètres
```

Réflexe de débogage : si l'agent n'appelle pas l'outil, vérifie d'abord qu'il fonctionne seul.

### Brancher sur un agent

```python
agent = create_agent(
    model=MODEL,
    tools=[tool1],
    system_prompt="You are an arithmetic wizard. Use your tools..."
)
```

### Lire la trace d'exécution

```python
pprint(response['messages'])
print(response["messages"][1].tool_calls)
```

Avec un outil, `response["messages"]` contient typiquement **quatre** entrées :

| Index | Type | Contenu |
|---|---|---|
| 0 | `HumanMessage` | la question |
| 1 | `AIMessage` | `content` **vide**, mais `tool_calls` rempli |
| 2 | `ToolMessage` | le retour de l'outil |
| 3 | `AIMessage` | la réponse finale en langage naturel |

> **Piège d'indexation.** Le notebook `1.1_prompting` écrit
> `response['messages'][1].content` — correct **parce qu'il n'y a pas d'outil**
> (la liste vaut `[Human, AI]`). Dès qu'un outil entre en jeu, l'index `1` est le message
> d'appel d'outil, dont `content` est vide. **Prends l'habitude de `[-1]`.**

### L'outil de recherche web

```python
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)
```

La leçon n'est pas Tavily, c'est le **motif** : un outil est une fonction Python
ordinaire qui enveloppe n'importe quelle API externe. Il lève la limite du cutoff
d'entraînement — d'où la question de démonstration « How up to date is your training
knowledge? » posée avant, puis après l'ajout de l'outil.

---

## 5. La mémoire

### Par défaut, un agent est sans état

```python
agent = create_agent(MODEL)
agent.invoke({"messages": [HumanMessage(content="My favourite colour is green")]})
agent.invoke({"messages": [HumanMessage(content="What's my favourite colour?")]})
# → il ne sait pas
```

Chaque `invoke` part d'une ardoise vierge.

### Deux façons de donner de la mémoire

**A — Manuellement**, en renvoyant tout l'historique à chaque tour :

```python
response = agent.invoke({"messages": [
    HumanMessage(content="What's the capital of the Moon?"),
    AIMessage(content="The capital of the Moon is Luna City."),
    HumanMessage(content="Tell me more about Luna City"),
]})
```

C'est toi qui portes l'historique. Instructif, mais ingérable à l'échelle.

**B — Avec un checkpointer**, la vraie solution :

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(MODEL, checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": [HumanMessage(content="My favourite colour is green")]}, config)
agent.invoke({"messages": [HumanMessage(content="What's my favourite colour?")]}, config)
# → vert
```

> **Les deux moitiés indissociables.** Le `checkpointer` dit *où* stocker.
> Le `thread_id` dit *quelle conversation* reprendre. Un checkpointer **sans**
> `thread_id` lève une erreur ; changer de `thread_id` ouvre une conversation neuve.
> C'est ainsi qu'un même agent sert plusieurs utilisateurs isolément.

`InMemorySaver` vit dans la RAM du processus : redémarrer le kernel efface tout. Les
checkpointers persistants (SQLite, Postgres) viennent plus tard.

---

## 6. Le multimodal

Le `content` d'un `HumanMessage` peut être une **chaîne** ou une **liste de blocs typés**.

```python
# Texte seul — les deux écritures sont équivalentes
HumanMessage(content="What is the capital of The Moon?")
HumanMessage(content=[{"type": "text", "text": "What is the capital of The Moon?"}])

# Texte + image
HumanMessage(content=[
    {"type": "text",  "text": "Tell me about this capital"},
    {"type": "image", "base64": img_b64, "mime_type": "image/png"}
])

# Texte + audio
HumanMessage(content=[
    {"type": "text",  "text": "Tell me about this audio file"},
    {"type": "audio", "base64": aud_b64, "mime_type": "audio/wav"}
])
```

Le motif est constant : `type`, `base64`, `mime_type`. L'encodage se fait toujours en
base64 puis en chaîne UTF-8 :

```python
img_b64 = base64.b64encode(img_bytes).decode("utf-8")
```

> **Contrainte à retenir.** Tous les modèles n'acceptent pas toutes les modalités. Le
> notebook change explicitement de modèle pour l'audio — d'où la variable distincte
> `MODEL_AUDIO`. Vérifie les capacités du modèle avant de lui envoyer une image ou du son.

---

## 7. La synthèse — `1.5_personal_chef`

Le lab final n'introduit rien : il **assemble** les quatre briques.

```python
agent = create_agent(
    model=MODEL,                    # 1.1  le moteur
    tools=[web_search],             # 1.2  les capacités
    system_prompt=system_prompt,    # 1.1  le comportement
    checkpointer=InMemorySaver()    # 1.3  la continuité
)

config = {"configurable": {"thread_id": "1"}}
response = agent.invoke({"messages": [HumanMessage(content="...")]}, config)
```

**Retiens cette signature.** `model`, `tools`, `system_prompt`, `checkpointer`,
et `response_format` en cinquième : ce sont les cinq paramètres de `create_agent` vus
dans le module. Si tu ne devais mémoriser qu'un bloc de code, c'est celui-ci.

---

## 8. Récapitulatif des imports

Savoir **d'où** vient chaque symbole est typiquement testé.

| Symbole | Provenance |
|---|---|
| `init_chat_model` | `langchain.chat_models` |
| `create_agent` | `langchain.agents` |
| `HumanMessage`, `AIMessage` | `langchain.messages` |
| `tool` | `langchain.tools` |
| `InMemorySaver` | `langgraph.checkpoint.memory` |
| `ChatGoogleGenerativeAI` | `langchain_google_genai` |
| `BaseModel` | `pydantic` |

> Note le glissement : `InMemorySaver` vient de **langgraph**, pas de langchain. La
> persistance appartient à la couche graphe — un indice de ce qui arrive au module 2.

---

## 9. Auto-évaluation

Réponds avant de dérouler les réponses.

1. Quelle est la différence de signature entre `model.invoke()` et `agent.invoke()` ?
2. Où lit-on le texte de la réponse dans chaque cas ?
3. Pourquoi `response["messages"][1].content` peut-il être vide ?
4. Qu'est-ce qui devient la description d'un outil décoré par `@tool` sans argument ?
5. Que manque-t-il si un agent avec `checkpointer` ne se souvient de rien ?
6. Quelle clé apparaît dans la réponse quand on passe `response_format` ?
7. Quelle est la différence entre un prompt structuré et une sortie structurée ?
8. Comment passer une image à un agent ?
9. D'où vient `InMemorySaver`, et qu'est-ce que ça révèle ?
10. Cite les cinq paramètres de `create_agent` vus dans le module.

<details>
<summary>Réponses</summary>

1. `model.invoke()` prend une chaîne ; `agent.invoke()` prend `{"messages": [...]}`.
2. `response.content` pour le modèle ; `response["messages"][-1].content` pour l'agent.
3. C'est l'`AIMessage` qui porte un appel d'outil : le contenu textuel est vide, l'information est dans `.tool_calls`.
4. La **docstring** de la fonction. Les annotations de type produisent le schéma des arguments.
5. Le `config={"configurable": {"thread_id": "..."}}` à l'appel d'`invoke`.
6. `"structured_response"`, contenant une instance du modèle Pydantic — accès par attribut.
7. Le prompt structuré **demande** un format en langage naturel, sans garantie. La sortie structurée **impose** un schéma Pydantic et renvoie un objet typé.
8. Un `content` en liste de blocs, avec `{"type": "image", "base64": ..., "mime_type": ...}`.
9. De `langgraph.checkpoint.memory`. La persistance relève de la couche graphe, pas de langchain — annonce du module 2.
10. `model`, `tools`, `system_prompt`, `checkpointer`, `response_format`.

</details>

---

## 10. Spécificités de ce dépôt

Les notebooks ont été migrés d'OpenAI vers Gemini. Deux conséquences pour tes révisions :

- Les modèles sont lus depuis `.env` via `MODEL = os.getenv("COURSE_MODEL", ...)`.
  Dans la documentation officielle tu verras `model="gpt-5-nano"` en dur : c'est la même
  chose, la variable ne change rien à la sémantique.
- Avec les modèles `gemini-3.x`, `.content` peut être une **liste de blocs** plutôt
  qu'une chaîne. Si un `print` affiche `[{'type': 'text', ...}]`, ce n'est pas une erreur
  de ta part. Extraire le texte :

  ```python
  c = response["messages"][-1].content
  texte = c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))
  ```

Les sorties enregistrées dans les notebooks proviennent encore des exécutions OpenAI des
auteurs — elles mentionnent `'model_provider': 'openai'`. Elles seront remplacées par les
tiennes à la première exécution.

---

## Références

- Intégrations de modèles : <https://docs.langchain.com/oss/python/integrations/chat>
- Notebooks source : [notebooks/module-1/](../lca-lc-foundations/notebooks/module-1/)
