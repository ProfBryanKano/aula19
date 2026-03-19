from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel


load_dotenv()


# Instrução vaga — o agente não tem identidade clara
agente_vago = Agent(
    model=OpenRouterModel("openai/gpt-4o-mini"),
    system_prompt="Você é um assistente inteligente.",
)


# Instrução focada — o agente tem uma responsabilidade única
agente_focado = Agent(
    model=OpenRouterModel("openai/gpt-4o-mini"),
    system_prompt=(
        "Você é um extrator de conclusões científicas. "
        "Dado um texto científico, sua única tarefa é identificar e listar as conclusões "
        "empiricamente sustentadas — ou seja, afirmações que o estudo demonstrou, mediu ou verificou. "
        "\n\n"
        "Formato obrigatório da resposta:\n"
        "- CONCLUSÕES: lista numerada, uma por linha, máximo 5 itens\n"
        "- LIMITAÇÕES MENCIONADAS: apenas se o texto citar explicitamente\n"
        "- TIPO DE ESTUDO: método identificado (ex: ensaio clínico, revisão sistemática, estudo observacional)\n"
        "\n"
        "Regras estritas:\n"
        "1. Use apenas o que está escrito — sem inferências ou conhecimento externo\n"
        "2. Linguagem técnica, frases curtas e diretas\n"
        "3. Se o texto não for científico, responda apenas: 'Texto não científico: [motivo em uma frase]'\n"
        "4. Não adicione introdução, opinião ou conclusão própria"
    ),
)

texto = """
Pesquisadores da USP identificaram que a exposição prolongada a telas
antes de dormir reduz em 40% a produção de melatonina, hormônio
responsável pela regulação do sono. O estudo acompanhou 200 voluntários
por 6 meses e concluiu que o uso de filtros de luz azul não elimina
completamente o problema.
"""


r1 = agente_vago.run_sync(f"Analise: {texto}")
r2 = agente_focado.run_sync(f"Analise: {texto}")


print("VAGO:")
print(r1.output)
print("\nFOCADO:")
print(r2.output)
