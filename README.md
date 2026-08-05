# Previsao de Preco do Tabaco — App Streamlit

Aplicativo Streamlit para previsao de preco do tabaco Virginia (Sul do Brasil).

Utiliza um modelo **CatBoost** treinado com dados de 93 paises (FAOSTAT + World Bank) para prever o preco internacional em USD/tonelada, convertendo para BRL com base no ultimo preco Afubra.

## Funcionalidades

- **Calculadora**: simule diferentes cenarios ajustando cambio, inflacao e producao, com calculo do valor total em arrobas e toneladas
- **Historico & Projecoes**: grafico interativo com dados historicos Afubra (2018–2024) e previsoes ate 2030
- Backtest 2024 com erro de apenas +0,4%

## Estrutura Esperada

O app espera os seguintes artefatos no caminho `../model_tabacco/modelagem/` (relativo ao diretorio do projeto):

```
../model_tabacco/modelagem/
├── catboost_modelo.cbm      # Modelo CatBoost treinado
└── contexto_brasil.pkl      # Contexto com series historicas (preco, producao, GDP, inflacao, cambio)
```

> **Nota**: esses arquivos nao fazem parte deste repositorio. Coloque-os manualmente no caminho indicado antes de executar o app.

## Como Usar

```bash
pip install streamlit pandas numpy catboost plotly
streamlit run app.py
```

## Metodo Hibrido

1. CatBoost preve o preco internacional em USD/tonne
2. Calcula a variacao relativa vs ano anterior
3. Aplica essa variacao ao ultimo preco real Afubra (2024: R$ 307,80/arroba)
4. Converte para BRL usando o cambio informado
