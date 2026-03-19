import requests
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel

load_dotenv()

HEADERS = {"User-Agent": "AgenteViagens/1.0"}


# ── Ferramenta 1: Informações sobre o destino (Wikipedia) ─────────────────────
def buscar_informacoes_destino(cidade: str) -> str:
    """
    Busca informações históricas e culturais sobre uma cidade ou país.
    Use para obter contexto sobre o destino: história, cultura, pontos turísticos.
    """
    try:
        busca = requests.get(
            "https://pt.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": cidade,
                "format": "json",
                "srlimit": 1,
            },
            timeout=5,
            headers=HEADERS,
        )
        resultados = busca.json().get("query", {}).get("search", [])
        if not resultados:
            return f"Nenhuma informação encontrada para '{cidade}'."

        titulo = resultados[0]["title"]
        resumo = requests.get(
            f"https://pt.wikipedia.org/api/rest_v1/page/summary/{titulo.replace(' ', '_')}",
            timeout=5,
            headers=HEADERS,
        )
        texto = (
            resumo.json().get("extract", "Sem resumo disponível.")
            if resumo.status_code == 200
            else "Resumo não encontrado."
        )
        return f"[Wikipedia — {titulo}]\n{texto}"

    except Exception as e:
        return f"Erro ao buscar informações: {e}"


# ── Ferramenta 2: Clima atual do destino (wttr.in) ────────────────────────────
def buscar_clima_atual(cidade: str) -> str:
    """
    Retorna o clima atual de uma cidade: temperatura, condição e umidade.
    Use sempre para informar as condições meteorológicas do destino.
    """
    try:
        resposta = requests.get(
            f"https://wttr.in/{cidade.replace(' ', '+')}",
            params={"format": "j1"},  # ← removido "lang": "pt"
            timeout=5,
            headers=HEADERS,
        )
        if resposta.status_code != 200:
            return f"Não foi possível obter o clima de '{cidade}'."

        dados = resposta.json()
        atual = dados["current_condition"][0]

        temperatura = atual["temp_C"]
        sensacao   = atual["FeelsLikeC"]
        umidade    = atual["humidity"]
        descricao  = atual["weatherDesc"][0]["value"]  # ← direto, sem lang_pt
        vento_kmh  = atual["windspeedKmph"]

        return (
            f"[Clima atual — {cidade}]\n"
            f"Condição: {descricao}\n"
            f"Temperatura: {temperatura}°C (sensação: {sensacao}°C)\n"
            f"Umidade: {umidade}%\n"
            f"Vento: {vento_kmh} km/h"
        )

    except Exception as e:
        return f"Erro ao buscar clima: {e}"


# ── Agente de viagens com as duas ferramentas ─────────────────────────────────
agente_viagens = Agent(
    model=OpenRouterModel("openai/gpt-4o-mini"),
    tools=[buscar_informacoes_destino, buscar_clima_atual],
    system_prompt=(
        "Você é um assistente especializado em planejamento de viagens. "
        "Para qualquer destino mencionado, você DEVE:\n"
        "1. Usar 'buscar_informacoes_destino' para obter contexto cultural e histórico\n"
        "2. Usar 'buscar_clima_atual' para informar as condições meteorológicas\n"
        "Combine as duas informações em uma resposta organizada com seções claras: "
        "'Sobre o destino' e 'Clima atual'. "
        "Finalize com dicas práticas baseadas no clima encontrado."
    ),
)


# ── Execução ──────────────────────────────────────────────────────────────────
destinos = ["Kyoto", "Buenos Aires"]

for destino in destinos:
    print("=" * 60)
    print(f"DESTINO: {destino.upper()}")
    print("=" * 60)

    resultado = agente_viagens.run_sync(f"Quero viajar para {destino}. O que devo saber?")

    print(resultado.output)
    print(f"\nChamadas à API: {resultado.usage().requests}")
    print()




