#!/usr/bin/env bash
# Teste de carga rápido do /predict, em degraus de concorrência.
#
# Pré-requisito: a ferramenta `hey`
#   macOS:  brew install hey
#   Linux:  sudo apt install hey   (ou: go install github.com/rakyll/hey@latest)
#
# Uso:
#   ./scripts/load_test.sh                          # API em http://localhost:8000
#   ./scripts/load_test.sh http://localhost:8000    # URL explícita
#   DURACAO=5s ./scripts/load_test.sh               # degraus mais curtos
#
# Dica: rode com a API no Docker e deixe nada pesado aberto na máquina,
# porque o gerador de carga e a API disputam a mesma CPU.

URL="${1:-http://localhost:8000}"
DURACAO="${DURACAO:-10s}"
DEGRAUS="${DEGRAUS:-1 2 4 6 8 12 20 50}"   # clientes simultâneos
CORPO='{"text": "patient with cardiac chest pain and heart failure"}'

if ! command -v hey >/dev/null 2>&1; then
  echo "hey não encontrado. Instale com: brew install hey" >&2
  exit 1
fi

# Aquecimento: a primeira requisição é mais lenta (cold start) e distorceria o degrau 1.
echo "Aquecendo..."
hey -z 3s -c 1 -m POST -H "Content-Type: application/json" -d "$CORPO" "$URL/predict" >/dev/null

printf "\n%-10s %-10s %-12s %-12s %-8s\n" "clientes" "req/s" "p95 (ms)" "p99 (ms)" "não-2xx"
printf "%-10s %-10s %-12s %-12s %-8s\n" "--------" "-----" "--------" "--------" "-------"

for C in $DEGRAUS; do
  SAIDA=$(hey -z "$DURACAO" -c "$C" -m POST \
    -H "Content-Type: application/json" -d "$CORPO" "$URL/predict")

  RPS=$(echo "$SAIDA" | awk '/Requests\/sec:/ {printf "%.1f", $2}')
  P95=$(echo "$SAIDA" | awk '/95% in/ {printf "%.1f", $3 * 1000}')
  P99=$(echo "$SAIDA" | awk '/99% in/ {printf "%.1f", $3 * 1000}')
  # Soma as respostas cujo status não é 200 (linhas no formato "[500]  12 responses").
  ERROS=$(echo "$SAIDA" | awk '/^ *\[[0-9]+\]/ && $1 != "[200]" {n += $2} END {print n + 0}')

  printf "%-10s %-10s %-12s %-12s %-8s\n" "$C" "$RPS" "$P95" "$P99" "$ERROS"
done

echo
echo "Limite = último degrau com p95 < 100 ms e sem erros."