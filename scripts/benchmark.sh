#!/bin/bash
# benchmark.sh — mede a latência do endpoint /predict em ambiente local
# Uso: ./scripts/benchmark.sh
# Pré-requisito: container rodando com docker run -p 8000:8000 medical-triage

URL="http://localhost:8000/predict"
PAYLOAD='{"text": "patient with cardiac chest pain and heart failure"}'
RUNS=10

echo "Rodando $RUNS requisições para $URL..."
echo "---"

# Executa N requisições e coleta o tempo de cada uma
times=$(for i in $(seq 1 $RUNS); do
  curl -s -o /dev/null -w "%{time_total}\n" \
    -X POST "$URL" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD"
done)

# Exibe o tempo de cada requisição
echo "$times" | nl -w2 -v1 | awk '{printf "  req %s: %ss\n", $1, $2}'

echo "---"

# Calcula e exibe mínimo, máximo e média
echo "$times" | awk '
  BEGIN { min=9999; max=0; sum=0; count=0 }
  {
    sum += $1; count++;
    if ($1 < min) min = $1;
    if ($1 > max) max = $1;
  }
  END {
    printf "  Média:   %.6fs\n", sum/count
    printf "  Mínimo:  %.6fs\n", min
    printf "  Máximo:  %.6fs\n", max
  }
'