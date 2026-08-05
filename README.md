# Previsao de Preco do Tabaco — App Streamlit

Aplicativo Streamlit para previsao de preco do tabaco Virginia (Sul do Brasil).

Carrega o modelo CatBoost exportado e permite simular diferentes cenarios ajustando cambio, inflacao e producao.

## Como Usar

```bash
pip install streamlit pandas numpy catboost plotly
streamlit run app.py
```

O app carrega automaticamente o modelo de `./model_tabacco/modelagem/`.

## Como Retreinar o Modelo

```bash
cd model_tabacco
pip install pandas numpy matplotlib seaborn scikit-learn catboost requests
python retreinar.py
```
