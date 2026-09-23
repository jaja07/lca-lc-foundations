# Module 2 — Fiche de révision

Agents avancés : MCP, contexte d'exécution, état personnalisé, multi-agent, RAG, SQL.
Établie à partir des huit notebooks de `notebooks/module-2/`.

---

## Carte du module

| Notebook | Notion centrale | À retenir absolument |
|---|---|---|
| `2.1_mcp` | Se connecter à des outils externes | `MultiServerMCPClient`, `stdio` vs distant |
| `2.1_travel_agent` | MCP distant en pratique | `transport="streamable_http"` + `url` |
| `2.2_runtime_context` | Configuration en lecture seule | `context_schema`, `ToolRuntime[Schema]` |
| `2.2_state` | État mutable partagé | `state_schema`, `Command(update={...})` |
| `2.3_multi_agent` | Composer des agents | Un sous-agent est appelé **depuis un outil** |
| `2.4_wedding_planners` | Synthèse | Coordinateur + sous-agents + état + retry MCP |
| `bonus_rag` | Recherche documentaire | Loader → splitter → embeddings → vector store |
| `bonus_sql` | Interroger une base | `SQLDatabase` enveloppée dans un `@tool` |

---

## 1. MCP — Model Context Protocol

MCP standardise la façon dont un agent découvre et appelle des capacités **externes au
code Python** : un serveur MCP expose des `tools`, des `resources` et des `prompts`,
peu importe le langage dans lequel il est écrit.

### Se connecter — deux transports

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

# Serveur local, lancé comme sous-processus (stdio)
client = MultiServerMCPClient({
    "local_server": {
        "transport": "stdio",
        "command": "python",
        "args": ["resources/2.1_mcp_server.py"],
    }
})

# Serveur distant, déjà en ligne (HTTP)
client = MultiServerMCPClient({
    "travel_server": {
        "transport": "streamable_http",
        "url": "https://mcp.kiwi.com",
    }
})
```

> **Le distinguo à connaître.** `stdio` démarre et pilote un **processus local** via
> `command`/`args` — le serveur vit et meurt avec le client. `streamable_http` se
> contente d'un `url` — le serveur tourne déjà, ailleurs, en permanence.

### Récupérer les trois primitives

```python
tools = await client.get_tools()                         # liste d'outils LangChain
resources = await client.get_resources("local_server")   # nécessite le nom du serveur
prompt = await client.get_prompt("local_server", "prompt")
prompt = prompt[0].content                                # une liste -> on prend le premier
```

`get_tools()` retourne directement des objets compatibles avec `create_agent(tools=...)`
— aucune conversion à faire, le adapter fait le travail. `get_prompt` renvoie une
**liste** de messages : dans le notebook, `prompt[0].content` est passé tel quel comme
`system_prompt`.

### Le côté serveur — pour comprendre ce qu'on consomme

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp_server")

@mcp.tool()
def search_web(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    ...

@mcp.resource("github://langchain-ai/langchain-mcp-adapters/main/README.md")
def github_file():
    """Resource for accessing a repo file"""
    ...

@mcp.prompt()
def prompt():
    """Analyze data from a langchain-ai repo file with comprehensive insights"""
    return "You are a helpful assistant..."

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Même logique que `@tool` côté LangChain : la **docstring** documente la capacité. Un
`@mcp.resource` est identifié par une **URI** (`github://...`), pas par un nom de
fonction.

> **Piège Windows, spécifique à ce dépôt.** Lancer un serveur MCP `stdio` en
> sous-processus depuis un notebook Jupyter sous Windows échoue silencieusement sans ce
> correctif, exécuté **avant** toute création de `MultiServerMCPClient` :
>
> ```python
> import sys, asyncio
> if sys.platform == "win32":
>     if not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
>         asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
>     if "ipykernel" in sys.modules:
>         sys.stderr = sys.__stderr__
> ```
>
> La policy par défaut (`SelectorEventLoop`) ne supporte pas les sous-processus ; il
> faut la `ProactorEventLoop`. Ce n'est **pas** dans la doc officielle LangChain, c'est
> une adaptation de ce dépôt pour Windows.

---

## 2. Le contexte d'exécution (`context_schema`)

Le contexte, c'est de la **configuration en lecture seule**, injectée à chaque appel
`invoke`, mais qui ne fait pas partie de l'historique de conversation.

```python
from dataclasses import dataclass
from langchain.agents import create_agent

@dataclass
class ColourContext:
    favourite_colour: str = "blue"
    least_favourite_colour: str = "yellow"

agent = create_agent(model=MODEL, context_schema=ColourContext)

response = agent.invoke(
    {"messages": [HumanMessage(content="What is my favourite colour?")]},
    context=ColourContext()          # passé à part, pas dans le dict d'état
)
```

### Le lire dans un outil

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_favourite_colour(runtime: ToolRuntime[ColourContext]) -> str:
    """Get the favourite colour of the user"""
    return runtime.context.favourite_colour
```

> **À mémoriser.** `ToolRuntime[ColourContext]` est un **paramètre spécial** : il
> n'apparaît **pas** dans le schéma d'arguments présenté au modèle — l'agent l'injecte
> automatiquement. Le modèle ne peut donc jamais halluciner sa valeur. C'est le mécanisme
> qui permet de donner à un outil des données que l'utilisateur ne fournit pas
> explicitement dans le message.

Changer `ColourContext(favourite_colour="green")` change la réponse **sans changer le
prompt** — le contexte est le canal propre pour des données comme un ID utilisateur, une
locale, des permissions.

---

## 3. L'état personnalisé (`state_schema`)

Le contexte est en lecture seule ; l'**état**, lui, est **mutable** et **partagé** entre
outils au sein d'un même graphe. C'est une extension du dictionnaire `messages` déjà vu
avec la mémoire (module 1), mais avec des clés supplémentaires.

### Déclarer le schéma

```python
from langchain.agents import AgentState

class CustomState(AgentState):
    favourite_colour: str
```

### Écrire dans l'état — `Command`

Un outil ordinaire ne peut pas modifier l'état par un simple `return`. Il doit renvoyer
un objet `Command` :

```python
from langgraph.types import Command
from langchain.messages import ToolMessage

@tool
def update_favourite_colour(favourite_colour: str, runtime: ToolRuntime) -> Command:
    """Update the favourite colour of the user in the state once they've revealed it."""
    return Command(update={
        "favourite_colour": favourite_colour,
        "messages": [ToolMessage("Successfully updated favourite colour",
                                  tool_call_id=runtime.tool_call_id)]
    })
```

> **Le piège numéro un de cette section.** Un `Command(update={...})` **doit** inclure
> une entrée `"messages"` avec un `ToolMessage` portant le `tool_call_id` du runtime.
> Sans ça, le graphe LangGraph ne considère pas l'appel d'outil comme terminé. Le
> `tool_call_id` vient toujours de `runtime.tool_call_id`, jamais inventé.

### Lire l'état

```python
@tool
def read_favourite_colour(runtime: ToolRuntime) -> str:
    """Read the favourite colour of the user from the state."""
    try:
        return runtime.state["favourite_colour"]
    except KeyError:
        return "No favourite colour found in state"
```

`runtime.state` est un **dictionnaire** — accès par clé, pas par attribut (contrairement
à `runtime.context`). D'où le `try/except KeyError` : la clé peut ne pas exister encore.

### Brancher le schéma et initialiser une valeur

```python
agent = create_agent(
    MODEL,
    tools=[update_favourite_colour, read_favourite_colour],
    checkpointer=InMemorySaver(),
    state_schema=CustomState
)

# On peut aussi injecter une valeur d'état directement à l'invoke :
agent.invoke(
    {"messages": [...], "favourite_colour": "green"},
    {"configurable": {"thread_id": "10"}}
)
```

> **Contexte vs État — la question d'examen.** Le contexte (`ColourContext`) est fixé
> à l'appel et ne change jamais pendant l'exécution — c'est de la config. L'état
> (`CustomState`) est modifié **pendant** l'exécution par les outils via `Command`, et
> **persiste** d'un appel à l'autre uniquement si un `checkpointer` + `thread_id` sont
> fournis — exactement le mécanisme de mémoire du module 1, étendu à des champs custom.

---

## 4. Le multi-agent — des sous-agents comme outils

L'idée clé : un agent LangChain **est déjà** un objet avec `.invoke()`. Rien n'empêche
de l'appeler **depuis l'intérieur d'un outil**, ce qui en fait un sous-agent invisible
pour l'agent principal.

```python
# Deux agents spécialisés, indépendants
subagent_1 = create_agent(model=MODEL, tools=[square_root])
subagent_2 = create_agent(model=MODEL, tools=[square])

# Chaque sous-agent est enveloppé dans un outil ordinaire
@tool
def call_subagent_1(x: float) -> float:
    """Call subagent 1 in order to calculate the square root of a number"""
    response = subagent_1.invoke({"messages": [HumanMessage(content=f"Calculate the square root of {x}")]})
    return response["messages"][-1].content

main_agent = create_agent(
    model=MODEL,
    tools=[call_subagent_1, call_subagent_2],
    system_prompt="You are a helpful assistant who can call subagents..."
)
```

> **À retenir.** Du point de vue de `main_agent`, `call_subagent_1` est un outil comme
> un autre : il ne voit ni le modèle, ni le prompt, ni les outils du sous-agent — juste
> sa description et sa sortie texte. C'est le même motif de composition qu'un outil qui
> appelle une API externe (module 1), mais l'« API » est elle-même un agent LangChain.
> Cette architecture permet d'isoler le contexte et les outils de chaque spécialiste.

---

## 5. La synthèse — `2.4_wedding_planners`

Le lab final assemble **tout** le module dans un coordinateur multi-agent avec état
partagé :

```python
class WeddingState(AgentState):
    origin: str
    destination: str
    guest_count: str
    genre: str

coordinator = create_agent(
    model=MODEL,
    tools=[search_flights, search_venues, suggest_playlist, update_state],
    state_schema=WeddingState,
    system_prompt="""
    You are a wedding coordinator.
    First find all the information you need to update the state. When you have the
    information, update the state.
    Once that has completed and returned, you can delegate the tasks
    to your specialists for flights, venues, and playlists.
    """
)
```

- `update_state` est un outil `Command` (section 3) qui remplit `origin`, `destination`,
  `guest_count`, `genre` — **il doit être appelé seul**, le prompt insiste dessus, sinon
  les autres outils liraient un état encore incomplet.
- `search_flights`, `search_venues`, `suggest_playlist` sont chacun un appel à un
  **sous-agent** (section 4) qui lit ses paramètres dans `runtime.state`.
- `travel_agent` utilise le MCP distant Kiwi (section 1) ; `venue_agent` utilise
  `web_search` (Tavily, module 1) ; `playlist_agent` utilise une base SQL (section 7).

### Fiabiliser un serveur MCP distant — l'intercepteur de retry

```python
RETRYABLE_MCP_CODES = {-32603}

class RetryMCPInterceptor:
    async def __call__(self, request, handler):
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(..., isError=False)   # erreur non-retryable : on répond quand même
            ...
            await asyncio.sleep(2 ** attempt)                    # backoff exponentiel

client = MultiServerMCPClient({...}, tool_interceptors=[RetryMCPInterceptor()])
```

> **Le principe à retenir, au-delà du code.** Un serveur MCP tiers (ici Kiwi, en ligne,
> hors de ton contrôle) peut échouer de façon transitoire. La bonne pratique n'est
> **pas** de laisser l'exception remonter et casser l'agent, mais de **renvoyer l'erreur
> à l'agent comme un `ToolMessage`** (`isError=False` avec un texte d'erreur) : le modèle
> peut alors décider d'ajuster sa requête et réessayer lui-même.

### Élargir le budget d'exécution

```python
response = await coordinator.ainvoke(
    {"messages": [...]},
    config={"tags": ["WP"], "recursion_limit": 40},
)
```

Un coordinateur qui enchaîne mise à jour d'état + trois sous-agents + leurs propres
boucles d'outils dépasse vite la limite par défaut de LangGraph. `recursion_limit`
relève ce plafond ; `tags` sert uniquement à retrouver la trace dans LangSmith.

---

## 6. Bonus — RAG (recherche documentaire)

Chaîne complète : charger → découper → vectoriser → indexer → chercher → envelopper en outil.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

data = PyPDFLoader("resources/acmecorp-employee-handbook.pdf").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("COURSE_EMBEDDINGS", "models/gemini-embedding-001"))
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search("How many days of vacation...?")
```

```python
@tool
def search_handbook(query: str) -> str:
    """Search the employee handbook for information"""
    results = vector_store.similarity_search(query)
    return results[0].page_content

agent = create_agent(model=MODEL, tools=[search_handbook], system_prompt="...")
```

> **À retenir.** `chunk_overlap=200` évite qu'une information soit coupée pile à la
> frontière entre deux morceaux. `add_start_index=True` conserve la position d'origine
> dans le document, utile pour citer une source. Le RAG n'est, au final, qu'**un outil
> de plus** — la seule nouveauté est ce qu'il y a derrière : une recherche par similarité
> plutôt qu'un appel d'API.

---

## 7. Bonus — SQL

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")

@tool
def sql_query(query: str) -> str:
    """Obtain information from the database using SQL queries"""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"

agent = create_agent(model=MODEL, tools=[sql_query])
response = agent.invoke({"messages": [HumanMessage(content="Who is the most popular artist beginning with 'S'?")]})
```

- Le modèle **génère lui-même** la requête SQL à partir de la question en langage
  naturel et du nom de l'outil — aucun texte SQL n'est écrit à la main.
- Le `try/except` est essentiel : une requête mal formée doit revenir comme un message
  d'erreur **texte** vers le modèle (qui peut se corriger), jamais comme une exception
  Python qui casse la boucle de l'agent — même principe que l'intercepteur MCP.

### Retrouver la requête générée

```python
print(response["messages"][-3].tool_calls[0]['args']['query'])
```

> **Piège d'indexation, variante SQL.** Avec un seul aller-retour d'outil, la séquence
> est `[Human, AI(tool_call), Tool, AI(final)]` (module 1) — mais le notebook indexe en
> `[-3]`, pas `[1]`, car le modèle a ici fait **plusieurs** appels d'outils (une requête
> exploratoire du schéma, puis la requête finale) avant de répondre. Ne suppose jamais
> une longueur fixe : inspecte `response["messages"]` avant d'indexer en dur.

---

## 8. Récapitulatif des imports

| Symbole | Provenance |
|---|---|
| `MultiServerMCPClient` | `langchain_mcp_adapters.client` |
| `ToolRuntime` | `langchain.tools` |
| `AgentState` | `langchain.agents` |
| `Command` | `langgraph.types` |
| `ToolMessage` | `langchain.messages` |
| `SQLDatabase` | `langchain_community.utilities` |
| `PyPDFLoader` | `langchain_community.document_loaders` |
| `RecursiveCharacterTextSplitter` | `langchain_text_splitters` |
| `GoogleGenerativeAIEmbeddings` | `langchain_google_genai` |
| `InMemoryVectorStore` | `langchain_core.vectorstores` |

> Remarque : `Command` vient de **langgraph.types**, pas de langchain — comme
> `InMemorySaver` au module 1, la mutation d'état est une notion de la couche graphe.

---

## 9. Auto-évaluation

Réponds avant de dérouler les réponses.

1. Quelle est la différence entre `transport="stdio"` et `transport="streamable_http"` ?
2. Que renvoie `client.get_prompt(...)`, et comment en extraire le texte utilisable ?
3. Pourquoi `ToolRuntime[ColourContext]` n'apparaît-il pas dans le schéma d'arguments présenté au modèle ?
4. Que doit obligatoirement contenir un `Command(update={...})` pour qu'un appel d'outil soit considéré comme terminé ?
5. Quelle est la différence fondamentale entre `context_schema` et `state_schema` ?
6. Comment un agent principal appelle-t-il un sous-agent ?
7. Pourquoi l'intercepteur MCP renvoie-t-il `isError=False` même en cas d'échec ?
8. Pourquoi ne faut-il jamais laisser une exception SQL remonter telle quelle depuis un outil ?
9. Sur quoi le modèle se base-t-il pour rédiger une requête SQL dans `bonus_sql` ?
10. Cite la chaîne complète des cinq étapes d'un pipeline RAG vu dans ce module.

<details>
<summary>Réponses</summary>

1. `stdio` lance un serveur **local** comme sous-processus via `command`/`args` ; `streamable_http` se connecte à un serveur **déjà en ligne** via `url`.
2. Une **liste** de messages de prompt ; on prend `prompt[0].content` pour obtenir le texte à passer en `system_prompt`.
3. C'est un paramètre spécial injecté automatiquement par l'agent — il est exclu du schéma d'outil transmis au modèle, qui ne peut donc ni le voir ni l'halluciner.
4. Une clé `"messages"` contenant un `ToolMessage` construit avec `tool_call_id=runtime.tool_call_id`.
5. Le contexte est une configuration **en lecture seule**, fixée à l'appel ; l'état est **mutable** pendant l'exécution et persiste entre appels via un checkpointer + `thread_id`.
6. En l'appelant depuis l'intérieur d'un `@tool` ordinaire, avec `sous_agent.invoke({"messages": [...]})`, puis en renvoyant `response["messages"][-1].content`.
7. Pour que l'échec revienne à l'agent comme un `ToolMessage` texte plutôt que comme une exception qui interromprait la boucle — le modèle peut alors ajuster sa requête et réessayer.
8. Parce que le modèle ne peut se corriger que s'il **voit** l'erreur ; une exception Python non interceptée casse l'exécution de l'agent avant qu'il ait pu réagir.
9. Sur la question en langage naturel et la description de l'outil `sql_query` — il n'y a pas de requête SQL écrite à la main dans le notebook.
10. Charger (`PyPDFLoader`) → découper (`RecursiveCharacterTextSplitter`) → vectoriser (`GoogleGenerativeAIEmbeddings`) → indexer (`InMemoryVectorStore`) → chercher (`similarity_search`).

</details>

---

## 10. Spécificités de ce dépôt

- Comme au module 1, `MODEL` (et ici aussi `COURSE_EMBEDDINGS`) est lu depuis `.env` ;
  la doc officielle affiche des chaînes en dur, c'est équivalent.
- Le correctif Windows pour `asyncio` (section 1) est indispensable **avant tout MCP
  stdio** dans un notebook — sans lui, le lancement du sous-processus échoue sur cette
  machine.
- Le quirk de `.content` en liste de blocs avec les modèles `gemini-3.x` (vu au module
  1) s'applique toujours ici, y compris dans les réponses des sous-agents.
- `2.4_wedding_planners` précise en tête de notebook que le lab a été **durci après le
  tournage** (gestion d'erreurs MCP, limite de recherches) : si tu compares à une vidéo
  du cours, attends-toi à des différences volontaires, pas à des erreurs.

---

## Références

- MCP avec LangChain : <https://docs.langchain.com/oss/python/langchain/mcp>
- Contexte et état d'un agent : <https://docs.langchain.com/oss/python/langchain/agents>
- Embeddings : <https://docs.langchain.com/oss/python/integrations/text_embedding>
- Notebooks source : [notebooks/module-2/](../lca-lc-foundations/notebooks/module-2/)
