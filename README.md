# 📊 Segmentação de Mercado por Candle Equivalente do mini-dólar

Este projeto implementa um sistema avançado de segmentação automática de mercado financeiro baseado em Candle Equivalente, com classificação dinâmica de regimes como:

📈 Tendência Ascendente (Fraca, Média, Forte)

📉 Tendência Descendente (Fraca, Média, Forte)

➖ Lateralidade

A abordagem combina análise de candles, ATR, amplitude real, filtros de consistência direcional, absorção estrutural e regras paramétricas para consolidar movimentos relevantes do preço.

O objetivo é transformar dados OHLC brutos em segmentos interpretáveis de estrutura de mercado.

# 🧠 Ideia Central

O pipeline funciona em etapas:

Leitura e preparação dos dados OHLCV.

Criação dos Candles Equivalentes puros.

Classificação paramétrica de convicção.

Filtro de consistência direcional.

Junção estrutural de segmentos.

Absorção de segmentos pequenos.

Filtro de exaustão.

Reclassificação conservadora de laterais.

Plotagem dos segmentos sobre o gráfico de candles.

O resultado é uma visão estrutural do mercado ao invés de apenas candle a candle.
