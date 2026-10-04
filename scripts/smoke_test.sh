#!/bin/bash
# smoke_test.sh — verifica se uma API já em execução responde corretamente
# Uso: ./scripts/smoke_test.sh [url-base]   (padrão: http://localhost:8000)
# Não sobe nem derruba container: isso é responsabilidade de quem chama
# (o workflow do CI, ou você manualmente).

URL="${1:-http://localhost:8000}"

# Espera a API ficar pronta: tenta /health até 30 vezes, 1s entre tentativas
# (o modelo é carregado no startup, então não responde na hora)
for i in $(seq 1 30); do
  if curl -sf "$URL/health" > /dev/null; then
    echo "API respondeu em /health (tentativa $i)"
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "ERRO: API não respondeu em 30s"
    exit 1
  fi
  sleep 1
done

# Chama /predict: prova que o modelo foi copiado e carregado
if ! RESPONSE=$(curl -sf -X POST "$URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "patient with cardiac chest pain and heart failure"}'); then
  echo "ERRO: /predict falhou"
  exit 1
fi

# Confere que a resposta tem o campo esperado
echo "$RESPONSE" | grep -q '"label"' || { echo "ERRO: resposta sem label: $RESPONSE"; exit 1; }

echo "Smoke test OK: $RESPONSE"